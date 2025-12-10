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

import torch.nn as nn

from secure_fl.models.cifar10 import CIFAR10Model
from secure_fl.models.flexible_mlp import FlexibleMLP
from secure_fl.models.mnist import MNISTModel
from secure_fl.models.resnet import ResNetBlock
from secure_fl.models.simple import SimpleModel


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
        "synthetic": SimpleModel,
        "mnist": MNISTModel,
        "cifar10": CIFAR10Model,
        "resnet": ResNetBlock,
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
