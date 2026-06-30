import argparse
import csv
import math
import os
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_definitions import MODEL_FACTORIES

@torch.no_grad()
def certified_robust_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilon: float,
    lipschitz_bound: float,
    bound_type: str = "logit_vector",
) -> Dict[str, float]:
    """
    Compute certified robust accuracy for an l2 perturbation radius epsilon.

    Let y be the true label and let
        margin(x, y) = f_y(x) - max_{j != y} f_j(x).

    A point is counted as certified robust if margin(x, y) > L_margin * epsilon.

    bound_type controls how the user-provided Lipschitz bound K is interpreted:
      - logit_vector: K bounds ||f(x)-f(x')||_2 / ||x-x'||_2.
                      Then every pairwise margin f_y - f_j is sqrt(2) K-Lipschitz.
      - per_logit:    K bounds each scalar logit f_j individually.
                      Then every pairwise margin f_y - f_j is 2 K-Lipschitz.
      - margin:       K directly bounds all pairwise margins f_y - f_j.
                      Then L_margin = K.

    epsilon must be in the same input coordinates as lipschitz_bound.
    """
    if lipschitz_bound <= 0:
        raise ValueError("lipschitz_bound must be positive.")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative.")

    if bound_type == "logit_vector":
        margin_lipschitz = math.sqrt(2.0) * lipschitz_bound
    elif bound_type == "per_logit":
        margin_lipschitz = 2.0 * lipschitz_bound
    elif bound_type == "margin":
        margin_lipschitz = lipschitz_bound
    else:
        raise ValueError(f"Unknown bound_type: {bound_type}")

    threshold = margin_lipschitz * epsilon

    model.eval()
    total = 0
    natural_correct = 0
    certified_correct = 0
    sum_radius = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
        logits_without_true = logits.clone()
        logits_without_true.scatter_(1, y.view(-1, 1), float("-inf"))
        max_other_logits = logits_without_true.max(dim=1).values

        margins = true_logits - max_other_logits
        radii = margins / margin_lipschitz

        is_natural_correct = pred.eq(y)
        # margin > 0 already implies natural correctness up to tie cases, but keep both conditions explicit.
        is_certified = is_natural_correct & (margins > threshold)

        batch_size = x.size(0)
        total += batch_size
        natural_correct += is_natural_correct.sum().item()
        certified_correct += is_certified.sum().item()
        sum_radius += radii.clamp_min(0.0).sum().item()

    return {
        "epsilon": epsilon,
        "lipschitz_bound": lipschitz_bound,
        "bound_type": bound_type,
        "margin_lipschitz": margin_lipschitz,
        "natural_accuracy": natural_correct / total,
        "certified_robust_accuracy": certified_correct / total,
        "mean_certified_radius": sum_radius / total,
        "num_samples": total,
    }


def make_mnist_test_loader(batch_size: int, data_root: str, num_workers: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(model_name: str, weights_path: str, device: torch.device) -> torch.nn.Module:
    if model_name not in MODEL_FACTORIES:
        valid = ", ".join(MODEL_FACTORIES.keys())
        raise ValueError(f"Unknown model_name '{model_name}'. Valid names: {valid}")

    model = MODEL_FACTORIES[model_name]().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def parse_lipschitz_bounds(values: Iterable[str]) -> List[float]:
    bounds = [float(v) for v in values]
    if len(bounds) != 2:
        raise ValueError("Please provide exactly two Lipschitz bounds, e.g. --lipschitz_bounds 12.3 10.7")
    return bounds


def epsilon_in_bound_coordinates(epsilon: float, epsilon_space: str) -> float:
    """
    The trained networks receive normalized MNIST inputs.

    If the Lipschitz bound is for the normalized-input network, use epsilon_space='pixel'
    to enter a pixel-space perturbation radius. For MNIST normalization x_norm=(x-MNIST_MEAN)/MNIST_STD,
    ||delta_norm||_2 = ||delta_pixel||_2 / MNIST_STD.

    Use epsilon_space='normalized' if epsilon is already in normalized coordinates.
    """
    if epsilon_space == "normalized":
        return epsilon
    if epsilon_space == "pixel":
        return epsilon
    raise ValueError("epsilon_space must be 'pixel' or 'normalized'.")


def infer_model_name_from_run_name(run_name: str) -> str:
    if run_name.startswith("LeNet5ReLU"):
        return "LeNet5ReLU"
    if run_name.startswith("2C2F"):
        return "2C2F"
    if run_name.startswith("FC-R18"):
        return "FC-R18"
    if run_name.startswith("C-R18"):
        return "C-R18"
    raise ValueError(
        f"Could not infer model name from run_name '{run_name}'. "
        "Pass --model_name explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute certified robust accuracy for MNIST models using manually supplied Lipschitz bounds."
    )
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the saved .pt state_dict.")
    parser.add_argument("--model_name", type=str, default=None, choices=list(MODEL_FACTORIES.keys()))
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name used in the output CSV.")
    parser.add_argument("--epsilon", type=float, required=True, help="l2 perturbation radius.")
    parser.add_argument(
        "--epsilon_space",
        type=str,
        default="normalized",
        choices=["normalized", "pixel"],
        help="Whether --epsilon is in normalized MNIST coordinates or original pixel coordinates.",
    )
    parser.add_argument(
        "--lipschitz_bounds",
        nargs=2,
        required=True,
        help="Exactly two manually computed Lipschitz upper bounds.",
    )
    parser.add_argument(
        "--bound_type",
        type=str,
        default="logit_vector",
        choices=["logit_vector", "per_logit", "margin"],
        help="How to interpret each supplied Lipschitz bound.",
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_csv", type=str, default="certified_robust_accuracy.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or os.path.splitext(os.path.basename(args.weights_path))[0]
    model_name = args.model_name or infer_model_name_from_run_name(run_name)
    lipschitz_bounds = parse_lipschitz_bounds(args.lipschitz_bounds)
    epsilon_for_certificate = epsilon_in_bound_coordinates(args.epsilon, args.epsilon_space)

    loader = make_mnist_test_loader(
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
    )
    model = load_model(model_name=model_name, weights_path=args.weights_path, device=device)

    rows: List[Dict[str, float]] = []
    for idx, lip_bound in enumerate(lipschitz_bounds, start=1):
        result = certified_robust_accuracy(
            model=model,
            loader=loader,
            device=device,
            epsilon=epsilon_for_certificate,
            lipschitz_bound=lip_bound,
            bound_type=args.bound_type,
        )
        row = {
            "run_name": run_name,
            "model_name": model_name,
            "weights_path": args.weights_path,
            "bound_index": idx,
            "input_epsilon": args.epsilon,
            "epsilon_space": args.epsilon_space,
            "epsilon_used_for_certificate": epsilon_for_certificate,
            **result,
            "natural_accuracy_percent": 100.0 * result["natural_accuracy"],
            "certified_robust_accuracy_percent": 100.0 * result["certified_robust_accuracy"],
        }
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Weights: {args.weights_path}")
    print(f"Input epsilon: {args.epsilon} ({args.epsilon_space})")
    print(f"Epsilon used for certificate: {epsilon_for_certificate:.8g}")
    print(f"Bound type: {args.bound_type}")
    print("")

    for row in rows:
        print(
            f"bound {row['bound_index']}: K={row['lipschitz_bound']:.8g} | "
            f"natural acc={row['natural_accuracy_percent']:.2f}% | "
            f"certified robust acc={row['certified_robust_accuracy_percent']:.2f}% | "
            f"mean certified radius={row['mean_certified_radius']:.6g}"
        )

    print(f"\nSaved results to: {args.output_csv}")


if __name__ == "__main__":
    main()
