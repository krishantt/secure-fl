import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """
    Simple fully connected neural network for testing and basic applications.

    This model is designed to be lightweight and fast for testing purposes,
    with configurable dimensions.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] = [128, 64],
        output_dim: int = 10,
        dropout_rate: float = 0.1,
        activation: str = "relu",
    ):
        """
        Initialize SimpleModel.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'tanh', 'gelu')
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # Select activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    self.activation,
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.flatten(1)

        return self.network(x)
