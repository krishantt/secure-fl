import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexibleMLP(nn.Module):
    """
    Flexible multi-layer perceptron with configurable architecture.

    This model allows for easy experimentation with different
    network architectures and hyperparameters.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        final_activation: str | None = None,
    ):
        """
        Initialize FlexibleMLP.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function name
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            final_activation: Final activation function (if any)
        """
        super().__init__()

        # Activation function mapping
        activation_map = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
        }

        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")

        self.activation = activation_map[activation]

        # Final activation
        self.final_activation = None
        if final_activation is not None:
            if final_activation not in activation_map:
                raise ValueError(f"Unsupported final activation: {final_activation}")
            self.final_activation = activation_map[final_activation]

        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))

            layers.append(self.activation)

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(dims[-1], output_dim))

        if self.final_activation is not None:
            layers.append(self.final_activation)

        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() > 2:
            x = x.flatten(1)
        return self.network(x)
