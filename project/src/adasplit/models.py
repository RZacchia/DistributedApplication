import torch
import torch.nn as nn


class LeNet5FeatureExtractor(nn.Module):
    """
    LeNet-5 feature extractor producing an 84-dim representation.
    Works for both:
      - CIFAR-10: 3x32x32
      - KronoDroid: 3x18x18 (paper)
    """
    def __init__(self, in_channels: int = 3, input_size: int = 32):
        super().__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # infer flatten dim for given input_size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self._forward_conv(dummy)
            flat_dim = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 120)
        self.fc2 = nn.Linear(120, 84)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class LeNet5ClientHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(84, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)
