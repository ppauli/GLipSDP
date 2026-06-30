"""
Train small FNN and CNN models on scikit-learn Digits and export weights for MATLAB.

Models:
    - FNN2: Linear -> ReLU -> Linear
    - FNN3: Linear -> ReLU -> Linear -> ReLU -> Linear
    - CNN2: Conv2d -> ReLU -> Linear
    - CNN3: Conv2d -> ReLU -> Conv2d -> ReLU -> Linear

Dataset:
    sklearn.datasets.load_digits
    1797 samples, 8x8 grayscale images, 10 classes

Exports:
    exports/<model_name>.pt
    exports/<model_name>_weights.mat

Requirements:
    pip install torch scikit-learn scipy numpy
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.io import savemat
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def get_digits_loaders(batch_size: int = 64, test_size: float = 0.2, seed: int = 0):
    digits = load_digits()

    X = digits.data.astype(np.float32)       # shape: [N, 64]
    y = digits.target.astype(np.int64)       # shape: [N]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Standardize features for stable training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # FNN input: [N, 64]
    X_train_fnn = torch.tensor(X_train, dtype=torch.float32)
    X_test_fnn = torch.tensor(X_test, dtype=torch.float32)

    # CNN input: [N, 1, 8, 8]
    X_train_cnn = torch.tensor(X_train.reshape(-1, 1, 8, 8), dtype=torch.float32)
    X_test_cnn = torch.tensor(X_test.reshape(-1, 1, 8, 8), dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_fnn = TensorDataset(X_train_fnn, y_train)
    test_fnn = TensorDataset(X_test_fnn, y_test)

    train_cnn = TensorDataset(X_train_cnn, y_train)
    test_cnn = TensorDataset(X_test_cnn, y_test)

    train_loader_fnn = DataLoader(train_fnn, batch_size=batch_size, shuffle=True)
    test_loader_fnn = DataLoader(test_fnn, batch_size=batch_size, shuffle=False)

    train_loader_cnn = DataLoader(train_cnn, batch_size=batch_size, shuffle=True)
    test_loader_cnn = DataLoader(test_cnn, batch_size=batch_size, shuffle=False)

    return train_loader_fnn, test_loader_fnn, train_loader_cnn, test_loader_cnn, scaler


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class FNN2(nn.Module):
    """
    2 trainable layers:
        Linear(64 -> hidden)
        Linear(hidden -> 10)
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(64, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x


class FNN3(nn.Module):
    """
    3 trainable layers:
        Linear(64 -> hidden1)
        Linear(hidden1 -> hidden2)
        Linear(hidden2 -> 10)
    """

    def __init__(self, hidden_dim1: int = 32, hidden_dim2: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(64, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, 10)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN2(nn.Module):
    """
    2 trainable layers:
        Conv2d(1 -> channels)
        Linear(channels * 8 * 8 -> 10)

    Padding keeps spatial size 8x8.
    """

    def __init__(self, channels: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels * 8 * 8, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return x


class CNN3(nn.Module):
    """
    3 trainable layers:
        Conv2d(1 -> channels1)
        Conv2d(channels1 -> channels2)
        Linear(channels2 * 8 * 8 -> 10)

    Padding keeps spatial size 8x8.
    """

    def __init__(self, channels1: int = 8, channels2: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=channels1,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=channels1,
            out_channels=channels2,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.relu2 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels2 * 8 * 8, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return x


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == y_batch).sum().item()
            total_samples += x_batch.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        test_acc = evaluate_model(model, test_loader, device)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss = {train_loss:.4f} | "
                f"train acc = {train_acc:.4f} | "
                f"test acc = {test_acc:.4f}"
            )

    return model


@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()

    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        predictions = logits.argmax(dim=1)

        total_correct += (predictions == y_batch).sum().item()
        total_samples += x_batch.size(0)

    return total_correct / total_samples


# ---------------------------------------------------------------------
# MATLAB export
# ---------------------------------------------------------------------

def export_weights_for_matlab(
    model: nn.Module,
    model_name: str,
    export_dir: str,
    scaler: StandardScaler,
):
    """
    Export model weights once, in MATLAB-friendly format.

    Linear weights:
        PyTorch: [out_features, in_features]
        Exported: [in_features, out_features]

    Conv2d weights:
        PyTorch: [out_channels, in_channels, kernel_height, kernel_width]
        Exported: [kernel_height, kernel_width, in_channels, out_channels]

    Bias vectors are exported unchanged.
    """

    os.makedirs(export_dir, exist_ok=True)

    mat_dict = {}

    for name, param in model.state_dict().items():
        array = param.detach().cpu().numpy()

        # MATLAB field names cannot contain dots.
        safe_name = name.replace(".", "_")

        if "weight" in name:
            if array.ndim == 2:
                # Linear layer
                mat_dict[safe_name] = array.T

            elif array.ndim == 4:
                # Conv2d layer
                mat_dict[safe_name] = np.transpose(array, (2, 3, 1, 0))

            else:
                mat_dict[safe_name] = array
        else:
            # Bias vectors
            mat_dict[safe_name] = array

    # Export input normalization parameters.
    mat_dict["input_mean"] = scaler.mean_.astype(np.float32)
    mat_dict["input_scale"] = scaler.scale_.astype(np.float32)

    mat_path = os.path.join(export_dir, f"{model_name}_weights.mat")
    savemat(mat_path, mat_dict)

    print(f"Saved MATLAB weights to: {mat_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)

    seeds = list(range(10))  # seeds 0, 1, ..., 9

    model_configs = {
        "fnn2": {
            "make_model": lambda: FNN2(hidden_dim=32),
            "input_type": "fnn",
            "epochs": 100,
            "lr": 1e-3,
        },
        "fnn3": {
            "make_model": lambda: FNN3(hidden_dim1=32, hidden_dim2=32),
            "input_type": "fnn",
            "epochs": 100,
            "lr": 1e-3,
        },
        "cnn2": {
            "make_model": lambda: CNN2(channels=8),
            "input_type": "cnn",
            "epochs": 100,
            "lr": 1e-3,
        },
        "cnn3": {
            "make_model": lambda: CNN3(channels1=8, channels2=16),
            "input_type": "cnn",
            "epochs": 100,
            "lr": 1e-3,
        },
    }

    # Store individual final test accuracies for each model type.
    results = {
        model_name: []
        for model_name in model_configs.keys()
    }

    for seed in seeds:
        print("\n" + "#" * 70)
        print(f"Training all models with seed {seed}")
        print("#" * 70)

        set_seed(seed)

        train_loader_fnn, test_loader_fnn, train_loader_cnn, test_loader_cnn, scaler = (
            get_digits_loaders(batch_size=64, test_size=0.2, seed=seed)
        )

        for model_name, cfg in model_configs.items():
            print("\n" + "=" * 70)
            print(f"Training {model_name}, seed {seed}")
            print("=" * 70)

            # Reset RNG before creating each model.
            # This makes initialization seed-dependent and reproducible.
            set_seed(seed)

            model = cfg["make_model"]()

            if cfg["input_type"] == "fnn":
                train_loader = train_loader_fnn
                test_loader = test_loader_fnn
            elif cfg["input_type"] == "cnn":
                train_loader = train_loader_cnn
                test_loader = test_loader_cnn
            else:
                raise ValueError(f"Unknown input_type: {cfg['input_type']}")

            model = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=cfg["epochs"],
                lr=cfg["lr"],
                weight_decay=0.0,
            )

            final_acc = evaluate_model(
                model=model,
                data_loader=test_loader,
                device=device,
            )

            print(
                f"Final test accuracy for {model_name}, seed {seed}: "
                f"{final_acc:.4f}"
            )

            results[model_name].append(final_acc)

            instance_name = f"{model_name}_seed{seed}"

            export_weights_for_matlab(
                model=model,
                model_name=instance_name,
                export_dir=export_dir,
                scaler=scaler,
            )

    # -----------------------------------------------------------------
    # Accuracy summary
    # -----------------------------------------------------------------

    print("\n" + "#" * 70)
    print("Summary of final test accuracies")
    print("#" * 70)

    accuracy_export = {}

    accuracy_export["seeds"] = np.array(seeds, dtype=np.int64)

    for model_name, accuracies in results.items():
        accuracies = np.array(accuracies, dtype=np.float32)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)  # sample standard deviation

        print(
            f"{model_name}: "
            f"mean = {mean_acc:.4f}, "
            f"std = {std_acc:.4f}, "
            f"individual = {accuracies}"
        )

        accuracy_export[f"{model_name}_test_accuracies"] = accuracies
        accuracy_export[f"{model_name}_mean_test_accuracy"] = np.array(
            [mean_acc],
            dtype=np.float32,
        )
        accuracy_export[f"{model_name}_std_test_accuracy"] = np.array(
            [std_acc],
            dtype=np.float32,
        )

    accuracy_path = os.path.join(export_dir, "test_accuracy_summary.mat")
    savemat(accuracy_path, accuracy_export)

    print(f"\nSaved test accuracy summary to: {accuracy_path}")


if __name__ == "__main__":
    main()