import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv2d(nn.Module):
    """
    TensorFlow-style 'same' padding for Conv2d.

    For input spatial size H x W and stride S, the output size is ceil(H / S) x ceil(W / S).
    This works for stride > 1 as well.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_height, input_width = x.shape[-2:]
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride

        output_height = math.ceil(input_height / stride_height)
        output_width = math.ceil(input_width / stride_width)

        pad_height = max((output_height - 1) * stride_height + kernel_height - input_height, 0)
        pad_width = max((output_width - 1) * stride_width + kernel_width - input_width, 0)

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        x = F.pad(x, pad=(pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)


class LeNet5ReLU(nn.Module):
    """Classical LeNet-5-style network for MNIST with ReLU activations."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, pad=(2, 2, 2, 2))  # 28 x 28 -> 32 x 32
        return self.net(x)


class TwoCTwoF(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            SamePadConv2d(1, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            SamePadConv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class FCResidualBlock(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = out + identity
        out = F.relu(out)
        return out


class FCR18(nn.Module):
    def __init__(self, num_blocks: int = 8):
        super().__init__()
        self.input_layer = nn.Linear(28 * 28, 64)
        self.blocks = nn.Sequential(*[FCResidualBlock(width=64) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.blocks(x)
        return self.output_layer(x)


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int = 16, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv1 = SamePadConv2d(channels, channels, kernel_size=kernel_size, stride=stride)
        self.conv2 = SamePadConv2d(channels, channels, kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = out + identity
        out = F.relu(out)
        return out


class CR18(nn.Module):
    def __init__(self, num_blocks: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            SamePadConv2d(1, 16, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.blocks = nn.Sequential(
            *[ConvResidualBlock(channels=16, kernel_size=3, stride=1) for _ in range(num_blocks)]
        )
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


MODEL_FACTORIES = {
    "LeNet5ReLU": lambda: LeNet5ReLU(),
    "2C2F": lambda: TwoCTwoF(),
    "FC-R18": lambda: FCR18(num_blocks=8),
    "C-R18": lambda: CR18(num_blocks=8),
}
