import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    """
    Basic ResNet building block with skip connections.

    This can be used as a component in larger architectures
    or for building custom ResNet-style models.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batch_norm: bool = True,
    ):
        """
        Initialize ResNetBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not use_batch_norm,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_batch_norm,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

        # Skip connection
        self.skip_connection = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        identity = self.skip_connection(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)

        return out
