import math
import os
import random
import numpy as np
import pandas as pd
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_definitions import MODEL_FACTORIES


# ============================================================
# Reproducibility utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# MATLAB export utilities
# ============================================================

def sanitize_matlab_name(name: str) -> str:
    """
    MATLAB variable names cannot contain '.', '-' etc.
    Example:
        features.0.conv.weight -> features_0_conv_weight
    """
    return (
        name.replace(".", "_")
            .replace("-", "_")
            .replace(" ", "_")
            .replace("/", "_")
    )


def save_model_for_matlab(model: nn.Module, filepath: str) -> None:
    """
    Saves a PyTorch model state_dict as a MATLAB-readable .mat file.

    Each parameter tensor is saved as a separate MATLAB variable.

    PyTorch conv weights have shape:
        [out_channels, in_channels, kernel_height, kernel_width]

    MATLAB users should be aware that MATLAB Deep Learning Toolbox
    commonly expects conv weights as:
        [kernel_height, kernel_width, in_channels, out_channels]

    Therefore, for convenience, this function also saves transposed
    versions of 4D convolutional weights with suffix '_matlab_format'.
    """
    state_dict = model.state_dict()
    mat_dict = {}

    for name, tensor in state_dict.items():
        array = tensor.detach().cpu().numpy()
        clean_name = sanitize_matlab_name(name)

        mat_dict[clean_name] = array

        if array.ndim == 4:
            # PyTorch: [out_channels, in_channels, kh, kw]
            # MATLAB:  [kh, kw, in_channels, out_channels]
            mat_dict[clean_name + "_matlab_format"] = np.transpose(
                array,
                axes=(2, 3, 1, 0),
            )

        elif array.ndim == 2:
            # PyTorch Linear weight: [out_features, in_features]
            # MATLAB fullyConnectedLayer weights often use the same orientation,
            # but this transposed version can be useful for manual computations.
            mat_dict[clean_name + "_transpose"] = array.T

    sio.savemat(filepath, mat_dict)


def save_results_csv(final_results, summary_results, results_dir: str) -> None:
    final_df = pd.DataFrame(final_results)
    summary_df = pd.DataFrame(summary_results)

    final_df.to_csv(
        os.path.join(results_dir, "final_results.csv"),
        index=False,
    )

    summary_df.to_csv(
        os.path.join(results_dir, "summary_results.csv"),
        index=False,
    )

# ============================================================
# Training and evaluation
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
):
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            device=device,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {100.0 * train_acc:.2f}% | "
            f"test loss {test_loss:.4f} | "
            f"test acc {100.0 * test_acc:.2f}%"
        )

    return model, history


# ============================================================
# Data loaders
# ============================================================

def make_mnist_loaders(
    batch_size: int,
    seed: int,
    data_root: str = "./data",
):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


# ============================================================
# Main experiment:
# 4 architectures x 5 independently initialized instances
# ============================================================

def main():
    # -------------------------
    # Hyperparameters
    # -------------------------

    batch_size = 128
    epochs = 10
    lr = 1e-3

    weight_decays = {
        "LeNet5ReLU": 2e-3,
        "2C2F": 2e-3,
        "FC-R18": 1e-2,
        "C-R18": 8e-2,
    }

    num_instances = 1
    base_seed = 1005

    results_dir = "models"
    weights_pt_dir = os.path.join(results_dir, "weights_pt")
    weights_mat_dir = os.path.join(results_dir, "weights_mat")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(weights_pt_dir, exist_ok=True)
    os.makedirs(weights_mat_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # Model factories
    # -------------------------

    model_factories = MODEL_FACTORIES

    trained_models = {}
    histories = {}
    final_results = []

    # -------------------------
    # Train 5 instances of each model
    # -------------------------

    for model_name, make_model in model_factories.items():
        for instance_idx in range(num_instances):
            seed = base_seed + instance_idx

            run_name = f"{model_name}_instance{instance_idx + 1}"

            current_weight_decay = weight_decays[model_name]

            print("\n" + "=" * 80)
            print(f"Training {run_name}")
            print(f"Weight decay: {current_weight_decay}")
            print("=" * 80)

            set_seed(seed)

            train_loader, test_loader = make_mnist_loaders(
                batch_size=batch_size,
                seed=seed,
                data_root="./data",
            )

            model = make_model()

            trained_model, history = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=epochs,
                lr=lr,
                weight_decay=current_weight_decay,
            )

            final_test_loss, final_test_acc = evaluate(
                model=trained_model,
                loader=test_loader,
                device=device,
            )

            # -------------------------
            # Save weights
            # -------------------------

            pt_path = os.path.join(
                weights_pt_dir,
                f"{run_name}.pt",
            )

            mat_path = os.path.join(
                weights_mat_dir,
                f"{run_name}.mat",
            )

            torch.save(
                trained_model.state_dict(),
                pt_path,
            )

            save_model_for_matlab(
                model=trained_model,
                filepath=mat_path,
            )

            print(f"Saved PyTorch weights to: {pt_path}")
            print(f"Saved MATLAB weights to: {mat_path}")

            # -------------------------
            # Store results
            # -------------------------

            trained_models[run_name] = trained_model
            histories[run_name] = history

            final_results.append({
                "run_name": run_name,
                "model": model_name,
                "instance": instance_idx + 1,
                "seed": seed,
                "lr": lr,
                "weight_decay": current_weight_decay,
                "test_loss": final_test_loss,
                "test_acc": final_test_acc,
                "test_acc_percent": 100.0 * final_test_acc,
                "pytorch_weights_path": pt_path,
                "matlab_weights_path": mat_path,
            })

    # -------------------------
    # Compute summaries
    # -------------------------

    summary_results = []

    for model_name in model_factories.keys():
        accs = np.array([
            result["test_acc"]
            for result in final_results
            if result["model"] == model_name
        ])

        losses = np.array([
            result["test_loss"]
            for result in final_results
            if result["model"] == model_name
        ])

        summary_results.append({
            "model": model_name,
            "num_instances": num_instances,
            "mean_test_acc": accs.mean(),
            "std_test_acc": accs.std(ddof=1),
            "mean_test_acc_percent": 100.0 * accs.mean(),
            "std_test_acc_percent": 100.0 * accs.std(ddof=1),
            "mean_test_loss": losses.mean(),
            "std_test_loss": losses.std(ddof=1),
        })

    # -------------------------
    # Save CSV result files
    # -------------------------

    save_results_csv(
        final_results=final_results,
        summary_results=summary_results,
        results_dir=results_dir,
    )

    print(f"\nSaved final results to: {os.path.join(results_dir, 'final_results.csv')}")
    print(f"Saved summary results to: {os.path.join(results_dir, 'summary_results.csv')}")

    # -------------------------
    # Print final individual results
    # -------------------------

    print("\n" + "=" * 80)
    print("Final test accuracies for all trained models")
    print("=" * 80)

    for result in final_results:
        print(
            f"{result['run_name']:35s} | "
            f"test loss {result['test_loss']:.4f} | "
            f"test acc {result['test_acc_percent']:.2f}%"
        )

    # -------------------------
    # Print mean/std per architecture
    # -------------------------

    print("\n" + "=" * 80)
    print("Mean and standard deviation of final test accuracies")
    print("=" * 80)

    for result in summary_results:
        print(
            f"{result['model']:12s} | "
            f"mean test acc {result['mean_test_acc_percent']:.2f}% | "
            f"std test acc {result['std_test_acc_percent']:.2f}% | "
            f"mean test loss {result['mean_test_loss']:.4f} | "
            f"std test loss {result['std_test_loss']:.4f}"
        )

    return trained_models, histories, final_results, summary_results


if __name__ == "__main__":
    trained_models, histories, final_results, summary_results = main()