"""
compute_lipschitz_lower_bounds.py

Evaluation-only script:
- loads trained PyTorch models from .pt files,
- computes empirical lower bounds on the global L2 Lipschitz constant,
- saves results to CSV.

Lower bounds:
1. Pairwise finite-difference lower bound:
       max_{i,j} ||f(x_i)-f(x_j)||_2 / ||x_i-x_j||_2

2. Jacobian spectral-norm lower bound:
       max_i ||J_f(x_i)||_2

Both are lower bounds on the true global Lipschitz constant.
"""

import os
import argparse
import itertools
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model_definitions import MODEL_FACTORIES


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data
# ============================================================

def make_mnist_test_loader(
    batch_size: int = 256,
    num_samples: int | None = None,
    seed: int = 0,
    data_root: str = "./data",
):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    if num_samples is not None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(test_dataset), size=num_samples, replace=False)
        test_dataset = Subset(test_dataset, indices.tolist())

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    return test_loader


# ============================================================
# Model loading
# ============================================================

def load_model(
    model_name: str,
    weights_path: str,
    device: torch.device,
) -> nn.Module:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Available models: {list(MODEL_FACTORIES.keys())}"
        )

    model = MODEL_FACTORIES[model_name]()
    state_dict = torch.load(weights_path, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# ============================================================
# Collect model outputs
# ============================================================

@torch.no_grad()
def collect_inputs_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    xs = []
    ys = []

    for x, _ in loader:
        x = x.to(device)
        y = model(x)

        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    return xs, ys


# ============================================================
# Pairwise finite-difference lower bound
# ============================================================

def pairwise_lipschitz_lower_bound(
    xs: torch.Tensor,
    ys: torch.Tensor,
    max_pairs: int | None = None,
    seed: int = 0,
    eps: float = 1e-12,
):
    """
    Computes

        max_{i,j} ||y_i - y_j||_2 / ||x_i - x_j||_2

    over either all pairs or a random subset of pairs.

    xs: shape [N, C, H, W]
    ys: shape [N, output_dim]
    """

    x_flat = xs.view(xs.size(0), -1)
    y_flat = ys.view(ys.size(0), -1)

    n = x_flat.size(0)

    if max_pairs is None:
        pairs = itertools.combinations(range(n), 2)
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)

        valid = i != j
        i = i[valid]
        j = j[valid]

        pairs = zip(i.tolist(), j.tolist())

    best_ratio = 0.0
    best_pair = None

    for i, j in pairs:
        dx = torch.linalg.vector_norm(x_flat[i] - x_flat[j], ord=2).item()
        dy = torch.linalg.vector_norm(y_flat[i] - y_flat[j], ord=2).item()

        if dx > eps:
            ratio = dy / dx

            if ratio > best_ratio:
                best_ratio = ratio
                best_pair = (i, j)

    return best_ratio, best_pair


# ============================================================
# Exact Jacobian spectral norm for one input
# ============================================================

def jacobian_spectral_norm_single(
    model: nn.Module,
    x: torch.Tensor,
):
    """
    Computes ||J_f(x)||_2 exactly by forming the full Jacobian.

    For MNIST classification:
        input dimension  = 1 * 28 * 28 = 784
        output dimension = 10

    So J has shape [10, 784], which is cheap enough.
    """

    model.eval()

    x = x.detach().clone()
    x.requires_grad_(True)

    def f_flat(x_single):
        # x_single has shape [1, 28, 28] or [C,H,W]
        out = model(x_single.unsqueeze(0))
        return out.squeeze(0)

    J = torch.autograd.functional.jacobian(
        f_flat,
        x,
        create_graph=False,
        strict=False,
        vectorize=True,
    )

    # J shape: [output_dim, C, H, W]
    J = J.reshape(J.shape[0], -1)

    # Spectral norm = largest singular value
    svals = torch.linalg.svdvals(J)
    return svals[0].item()


def jacobian_lipschitz_lower_bound(
    model: nn.Module,
    xs: torch.Tensor,
    device: torch.device,
    num_jacobian_samples: int = 100,
    seed: int = 0,
):
    """
    Computes

        max_i ||J_f(x_i)||_2

    over a subset of samples.
    """

    n = xs.size(0)
    rng = np.random.default_rng(seed)

    if num_jacobian_samples > n:
        num_jacobian_samples = n

    indices = rng.choice(n, size=num_jacobian_samples, replace=False)

    best_norm = 0.0
    best_index = None

    for count, idx in enumerate(indices, start=1):
        x = xs[idx].to(device)

        jac_norm = jacobian_spectral_norm_single(
            model=model,
            x=x,
        )

        if jac_norm > best_norm:
            best_norm = jac_norm
            best_index = int(idx)

        print(
            f"    Jacobian sample {count:4d}/{num_jacobian_samples} | "
            f"current {jac_norm:.6f} | best {best_norm:.6f}"
        )

    return best_norm, best_index


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights_dir",
        type=str,
        default="Versatility/models/weights_pt",
        help="Directory containing .pt weight files.",
    )

    parser.add_argument(
        "--model_names",
        nargs="+",
        default=["LeNet5ReLU", "2C2F", "FC-R18", "C-R18"],
        help="Model names to evaluate.",
    )

    parser.add_argument(
        "--instances",
        nargs="+",
        type=int,
        default=[1],
        help="Instance numbers to evaluate, e.g. 1 2 3 4 5.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of MNIST test samples used for lower-bound estimation.",
    )

    parser.add_argument(
        "--num_jacobian_samples",
        type=int,
        default=100,
        help="Number of samples used for Jacobian spectral-norm lower bound.",
    )

    parser.add_argument(
        "--max_pairs",
        type=int,
        default=200000,
        help=(
            "Number of random pairs for pairwise lower bound. "
            "Use -1 for all pairs."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="lipschitz_lower_bounds.csv",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader = make_mnist_test_loader(
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed,
        data_root="./data",
    )

    if args.max_pairs == -1:
        max_pairs = None
    else:
        max_pairs = args.max_pairs

    results = []

    for model_name in args.model_names:
        for instance in args.instances:
            run_name = f"{model_name}_instance{instance}"
            weights_path = os.path.join(
                args.weights_dir,
                f"{run_name}.pt",
            )

            print("\n" + "=" * 80)
            print(f"Evaluating {run_name}")
            print("=" * 80)

            if not os.path.exists(weights_path):
                print(f"Skipping: weights file not found: {weights_path}")
                continue

            model = load_model(
                model_name=model_name,
                weights_path=weights_path,
                device=device,
            )

            xs, ys = collect_inputs_outputs(
                model=model,
                loader=test_loader,
                device=device,
            )

            print(f"Collected {xs.size(0)} samples.")

            pair_lb, best_pair = pairwise_lipschitz_lower_bound(
                xs=xs,
                ys=ys,
                max_pairs=max_pairs,
                seed=args.seed,
            )

            print(f"Pairwise lower bound: {pair_lb:.6f}")
            print(f"Best pair: {best_pair}")

            jac_lb, best_jac_index = jacobian_lipschitz_lower_bound(
                model=model,
                xs=xs,
                device=device,
                num_jacobian_samples=args.num_jacobian_samples,
                seed=args.seed,
            )

            print(f"Jacobian lower bound: {jac_lb:.6f}")
            print(f"Best Jacobian sample index: {best_jac_index}")

            combined_lb = max(pair_lb, jac_lb)

            print(f"Combined empirical lower bound: {combined_lb:.6f}")

            results.append({
                "run_name": run_name,
                "model": model_name,
                "instance": instance,
                "weights_path": weights_path,
                "num_samples": xs.size(0),
                "max_pairs": "all" if max_pairs is None else max_pairs,
                "num_jacobian_samples": args.num_jacobian_samples,
                "pairwise_lower_bound": pair_lb,
                "best_pair": str(best_pair),
                "jacobian_lower_bound": jac_lb,
                "best_jacobian_sample_index": best_jac_index,
                "combined_lower_bound": combined_lb,
            })

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(df)

    print(f"\nSaved results to: {args.output_csv}")


if __name__ == "__main__":
    main()