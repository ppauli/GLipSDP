import os
import itertools
from turtle import width
from xml.parsers.expat import model
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.nn.utils.parametrizations import spectral_norm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3

EXPORT_DIR = "exported_mnist_models"
os.makedirs(EXPORT_DIR, exist_ok=True)

MLP_HIDDEN_LAYERS = [3, 5, 9, 17, 33, 65]
MLP_WIDTHS = [16, 32, 64]

CNN_CONV_LAYERS = [3, 5, 9, 17]
CNN_CHANNELS = [8, 16, 32]

NUM_CLASSES = 10
MNIST_INPUT_DIM = 28 * 28


# ============================================================
# Data
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# ============================================================
# Model definitions
# ============================================================

class ScaledSpectralLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        spectral_scale: float = 1.1,
        n_power_iterations: int = 1,
        is_output_layer: bool = False,
    ):
        super().__init__()

        linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

        if is_output_layer:
            nn.init.xavier_normal_(linear.weight)
        else:
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")

        if bias:
            nn.init.zeros_(linear.bias)

        self.linear = spectral_norm(
            linear,
            name="weight",
            n_power_iterations=n_power_iterations,
        )

        self.spectral_scale = spectral_scale

    def forward(self, x):
        return self.spectral_scale * self.linear(x)


class ScaledSpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        spectral_scale: float = 1.1,
        n_power_iterations: int = 1,
    ):
        super().__init__()

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")

        if bias:
            nn.init.zeros_(conv.bias)

        self.conv = spectral_norm(
            conv,
            name="weight",
            n_power_iterations=n_power_iterations,
        )

        self.spectral_scale = spectral_scale

    def forward(self, x):
        return self.spectral_scale * self.conv(x)


class MLP(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        width: int,
        spectral_scale: float = 1.1,
        n_power_iterations: int = 1,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Flatten())

        in_features = 28 * 28

        for _ in range(num_hidden_layers):
            layers.append(
                ScaledSpectralLinear(
                    in_features=in_features,
                    out_features=width,
                    spectral_scale=spectral_scale,
                    n_power_iterations=n_power_iterations,
                    is_output_layer=False,
                )
            )
            layers.append(nn.LeakyReLU(0.01))

            in_features = width

        layers.append(
            ScaledSpectralLinear(
                in_features=width,
                out_features=10,
                spectral_scale=1.1,
                n_power_iterations=n_power_iterations,
                is_output_layer=True,
            )
        )

        self.net = nn.Sequential(*layers)

        self.num_hidden_layers = num_hidden_layers
        self.width = width
        self.spectral_scale = spectral_scale
        self.n_power_iterations = n_power_iterations

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(
        self,
        num_conv_layers: int,
        channels: int,
        spectral_scale: float = 1.1,
        n_power_iterations: int = 1,
    ):
        super().__init__()

        layers = []

        in_channels = 1

        for _ in range(num_conv_layers):
            layers.append(
                ScaledSpectralConv2d(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    spectral_scale=spectral_scale,
                    n_power_iterations=n_power_iterations,
                )
            )
            layers.append(nn.LeakyReLU(0.01))

            in_channels = channels

        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = ScaledSpectralLinear(
            in_features=channels,
            out_features=10,
            spectral_scale=1.0,
            n_power_iterations=n_power_iterations,
            is_output_layer=True,
        )

        self.num_conv_layers = num_conv_layers
        self.channels = channels
        self.spectral_scale = spectral_scale
        self.n_power_iterations = n_power_iterations

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ============================================================
# Training and evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()

        #torch.nn.utils.clip_grad_norm_(
        #    model.parameters(),
        #    max_norm=1.0,
        #)

        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train_model(model, model_name):
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining {model_name}")
    print("-" * 80)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_acc)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} | "
            f"test acc {test_acc:.4f}"
        )

    return model, history


# ============================================================
# Export functions
# ============================================================

def export_to_onnx(model, filepath):
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28, device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def export_to_mat(model, filepath, metadata=None, history=None):
    """
    Export effective weights and biases to MATLAB.

    This avoids exporting PyTorch spectral-normalization internals such as
    parametrizations.weight.original, _u, and _v.
    """

    params = {}

    linear_idx = 0
    conv_idx = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            W = module.weight.detach().cpu().numpy()
            b = module.bias.detach().cpu().numpy() if module.bias is not None else None

            params[f"linear_{linear_idx}_weight"] = W

            if b is not None:
                params[f"linear_{linear_idx}_bias"] = b

            linear_idx += 1

        elif isinstance(module, nn.Conv2d):
            W = module.weight.detach().cpu().numpy()
            b = module.bias.detach().cpu().numpy() if module.bias is not None else None

            params[f"conv_{conv_idx}_weight"] = W

            if b is not None:
                params[f"conv_{conv_idx}_bias"] = b

            conv_idx += 1

    export_dict = {
        "params": params,
    }

    if metadata is not None:
        export_dict["metadata"] = metadata

    if history is not None:
        export_dict["history"] = {
            key: np.asarray(value)
            for key, value in history.items()
        }

    sio.savemat(
        filepath,
        export_dict,
        long_field_names=True,
    )


def export_model(model, model_name, metadata, history):
    onnx_path = os.path.join(EXPORT_DIR, f"{model_name}.onnx")
    mat_path = os.path.join(EXPORT_DIR, f"{model_name}.mat")
    pt_path = os.path.join(EXPORT_DIR, f"{model_name}.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "history": history,
        },
        pt_path,
    )

    export_to_onnx(model, onnx_path)
    export_to_mat(model, mat_path, metadata=metadata, history=history)

    print(f"Saved PyTorch model: {pt_path}")
    print(f"Saved ONNX model:    {onnx_path}")
    print(f"Saved MAT file:      {mat_path}")


# ============================================================
# Main experiment loop
# ============================================================

def run_all_experiments():
    results = []

    # --------------------------------------------------------
    # Fully connected networks
    # --------------------------------------------------------

    for num_hidden_layers, width in itertools.product(
        MLP_HIDDEN_LAYERS,
        MLP_WIDTHS,
    ):
        model_name = f"mlp_L{num_hidden_layers}_W{width}"

        model = MLP(
            num_hidden_layers=num_hidden_layers,
            width=width,
            spectral_scale=2.0,
            n_power_iterations=1,
        )   
        metadata = {
            "model_type": "MLP",
            "num_hidden_layers": num_hidden_layers,
            "width": width,
            "input_shape": np.array([1, 28, 28]),
            "num_classes": NUM_CLASSES,
        }

        model, history = train_model(model, model_name)

        export_model(
            model=model,
            model_name=model_name,
            metadata=metadata,
            history=history,
        )

        results.append({
            "model_name": model_name,
            "model_type": "MLP",
            "num_hidden_layers": num_hidden_layers,
            "width": width,
            "final_test_accuracy": history["test_accuracy"][-1],
        })

    # --------------------------------------------------------
    # Convolutional networks
    # --------------------------------------------------------

    for num_conv_layers, channels in itertools.product(
        CNN_CONV_LAYERS,
        CNN_CHANNELS,
    ):
        model_name = f"cnn_C{num_conv_layers}_CH{channels}"

        model = CNN(
            num_conv_layers=num_conv_layers,
            channels=channels,
            spectral_scale=2.0,
            n_power_iterations=1,
        )   
        metadata = {
            "model_type": "CNN",
            "num_conv_layers": num_conv_layers,
            "channels": channels,
            "kernel_size": 3,
            "padding": 1,
            "input_shape": np.array([1, 28, 28]),
            "num_classes": NUM_CLASSES,
            "fully_connected_layers_after_conv": 1,
        }

        model, history = train_model(model, model_name)

        export_model(
            model=model,
            model_name=model_name,
            metadata=metadata,
            history=history,
        )

        results.append({
            "model_name": model_name,
            "model_type": "CNN",
            "num_conv_layers": num_conv_layers,
            "channels": channels,
            "kernel_size": 3,
            "final_test_accuracy": history["test_accuracy"][-1],
        })

    results_path = os.path.join(EXPORT_DIR, "summary_results.mat")

    sio.savemat(
        results_path,
        {
            "results": results,
        },
    )

    print("\nAll experiments completed.")
    print(f"Summary saved to: {results_path}")


if __name__ == "__main__":
    run_all_experiments()