import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10Model(nn.Module):
    """
    Convolutional neural network for CIFAR-10 image classification.

    This model uses convolutional layers to effectively process
    32x32 RGB images from the CIFAR-10 dataset.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
    ):
        """
        Initialize CIFAR10Model.

        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.use_batch_norm = use_batch_norm

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate flattened dimension after convolution and pooling
        # Input: 32x32 -> after 3 pools: 4x4
        self.flattened_dim = 128 * 4 * 4

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256) if use_batch_norm else nn.Identity()
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128) if use_batch_norm else nn.Identity()
        self.fc3 = nn.Linear(128, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with appropriate schemes."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        # Flatten and fully connected layers
        x = x.flatten(1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)

        return x
