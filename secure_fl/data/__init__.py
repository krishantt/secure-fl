"""
Data handling module for Secure Federated Learning

This module provides data loading, preprocessing, and partitioning utilities
for federated learning experiments. It supports multiple datasets and
provides consistent interfaces for data distribution across clients.

Supported datasets:
- MNIST: Handwritten digits classification
- CIFAR-10: Natural image classification
- MedMNIST: Medical image classification
- Synthetic: Generated data for testing

Key components:
- FederatedDataLoader: Main class for loading and partitioning datasets
- Dataset utilities and transformations
- Client data distribution strategies
"""

from .dataloader import FederatedDataLoader
from .utils import create_data_transforms, get_dataset_info

__all__ = [
    "FederatedDataLoader",
    "get_dataset_info",
    "create_data_transforms",
]
