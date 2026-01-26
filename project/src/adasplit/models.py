import torch.nn as nn
import torch

# ------------------------------
# Paper model: LeNet-5 split (feature extractor + client head)
# Model split: f(θ) and h(Φ)
# ------------------------------



class LeNet5FeatureExtractor(nn.Module):
    """
    LeNet-5 feature extractor f(θ_i;·) used in the paper (Section V-B-2 "Model and Training").
    Paper detail: "Each client utilizes the LeNet-5 architecture, where the first four layers
    are shared with the server for feature extraction, and the final fully connected (FC) layer
    acts as the client-specific decision layer." 

    We implement the standard LeNet-5 up to the 84-dim feature vector:
      conv1 -> pool -> conv2 -> pool -> fc1(120) -> fc2(84)
    The client-specific classifier head is a separate Linear(84, num_classes).
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)   # layer 1
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)      # layer 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)            # layer 3
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)      # layer 4

        # For 32x32 CIFAR: output after pool2 is (16, 5, 5) => 400 features
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)  # paper: FC layer input dimension = 84 for CIFAR-10 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x  # 84-dim representation


class LeNet5ClientHead(nn.Module):
    """
    Client-specific decision layer h(Φ_i;·): Linear(84, C).
    default value = 10 since I will only train on CIFAR10
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(84, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)