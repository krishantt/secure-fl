import torch
import torch.nn as nn


class MNISTModel(nn.Module):
    """
    Optimized neural network for MNIST digit classification.

    This model is specifically designed for 28x28 grayscale images
    and achieves good accuracy on MNIST dataset.
    """

    def __init__(
        self,
        hidden_dims: list[int] = None,
        output_dim: int = 10,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
    ):
        """
        Initialize MNISTModel.

        Args:
            hidden_dims: Hidden layer dimensions
            output_dim: Number of output classes
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        if hidden_dims is None:
            hidden_dims = [128, 64]
        super().__init__()

        self.flatten = nn.Flatten()
        self.use_batch_norm = use_batch_norm

        # Build network layers
        layers = []
        prev_dim = 28 * 28  # MNIST image size

        for _i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.extend([nn.ReLU(inplace=True), nn.Dropout(dropout_rate)])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.classifier = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with appropriate schemes."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)

        Returns:
            Output logits of shape (batch_size, output_dim)
        """
        x = self.flatten(x)

        # Handle batch_size=1 case for batch normalization
        if x.shape[0] == 1 and self.use_batch_norm and self.training:
            # Temporarily switch to eval mode for single sample inference
            # This preserves gradients while avoiding BatchNorm issues
            was_training = self.training
            self.eval()
            output = self.classifier(x)
            if was_training:
                self.train()
            return output
        else:
            return self.classifier(x)
