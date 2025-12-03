"""
Shared Model Definitions for Secure Federated Learning

This module provides common neural network models used throughout the secure-fl
framework, eliminating code duplication and ensuring consistency across
different components.

Models included:
- SimpleModel: Basic fully connected network for testing
- MNISTModel: Optimized model for MNIST dataset
- CIFAR10Model: CNN model for CIFAR-10 dataset
- ResNetBlock: Basic ResNet building block
- FlexibleMLP: Configurable multi-layer perceptron

Usage:
    from secure_fl.models import MNISTModel, CIFAR10Model, SimpleModel

    # Create models
    mnist_model = MNISTModel()
    cifar_model = CIFAR10Model()
    simple_model = SimpleModel(input_dim=784, output_dim=10)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MNISTModel(nn.Module):
    """
    Optimized neural network for MNIST digit classification.

    This model is specifically designed for 28x28 grayscale images
    and achieves good accuracy on MNIST dataset.
    """

    def __init__(
        self,
        hidden_dims: list[int] = [128, 64],
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
        super().__init__()

        self.flatten = nn.Flatten()
        self.use_batch_norm = use_batch_norm

        # Build network layers
        layers = []
        prev_dim = 28 * 28  # MNIST image size

        for i, hidden_dim in enumerate(hidden_dims):
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


# Model factory functions
def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create models by name.

    Args:
        model_name: Name of the model to create
        **kwargs: Model-specific arguments

    Returns:
        Initialized model

    Raises:
        ValueError: If model name is not recognized
    """
    model_registry = {
        "simple": SimpleModel,
        "mnist": MNISTModel,
        "cifar10": CIFAR10Model,
        "flexible_mlp": FlexibleMLP,
    }

    if model_name not in model_registry:
        available_models = ", ".join(model_registry.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available_models}"
        )

    return model_registry[model_name](**kwargs)


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024**2)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": model_size_mb,
        "model_class": model.__class__.__name__,
    }


# Export all public classes and functions
__all__ = [
    "SimpleModel",
    "MNISTModel",
    "CIFAR10Model",
    "ResNetBlock",
    "FlexibleMLP",
    "create_model",
    "get_model_info",
]
