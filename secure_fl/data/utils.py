"""
Data utilities for federated learning datasets

This module provides utility functions for data transformations,
dataset information, and common preprocessing operations used
across different federated learning experiments.
"""

import logging
from typing import Any

import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


def create_data_transforms(
    dataset_name: str,
    train: bool = True,
    augment: bool = False,
    normalize: bool = True,
) -> transforms.Compose:
    """
    Create appropriate data transforms for a given dataset.

    Args:
        dataset_name: Name of the dataset
        train: Whether transforms are for training (affects augmentation)
        augment: Whether to apply data augmentation
        normalize: Whether to apply normalization

    Returns:
        Composed transforms for the dataset
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        return _create_mnist_transforms(train, augment, normalize)
    elif dataset_name == "cifar10":
        return _create_cifar10_transforms(train, augment, normalize)
    elif dataset_name == "medmnist":
        return _create_medmnist_transforms(train, augment, normalize)
    else:
        # Default transforms for synthetic or unknown datasets
        return _create_default_transforms()


def _create_mnist_transforms(
    train: bool = True, augment: bool = False, normalize: bool = True
) -> transforms.Compose:
    """Create transforms for MNIST dataset."""
    transform_list = []

    # Convert PIL to tensor
    transform_list.append(transforms.ToTensor())

    # Data augmentation for training
    if train and augment:
        transform_list.extend(
            [
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        )

    # Normalization (MNIST mean and std)
    if normalize:
        transform_list.append(transforms.Normalize(mean=(0.1307,), std=(0.3081,)))

    return transforms.Compose(transform_list)


def _create_cifar10_transforms(
    train: bool = True, augment: bool = False, normalize: bool = True
) -> transforms.Compose:
    """Create transforms for CIFAR-10 dataset."""
    transform_list = []

    # Data augmentation for training
    if train and augment:
        transform_list.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
            ]
        )

    # Convert PIL to tensor
    transform_list.append(transforms.ToTensor())

    # Normalization (CIFAR-10 mean and std)
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            )
        )

    return transforms.Compose(transform_list)


def _create_medmnist_transforms(
    train: bool = True, augment: bool = False, normalize: bool = True
) -> transforms.Compose:
    """Create transforms for MedMNIST dataset."""
    transform_list = []

    # Convert to tensor (MedMNIST data is already numpy arrays)
    transform_list.append(transforms.ToTensor())

    # Data augmentation for training
    if train and augment:
        transform_list.extend(
            [
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]
        )

    # Normalization (use ImageNet stats as approximation for medical images)
    if normalize:
        transform_list.append(transforms.Normalize(mean=(0.485,), std=(0.229,)))

    return transforms.Compose(transform_list)


def _create_default_transforms() -> transforms.Compose:
    """Create default transforms for synthetic or unknown datasets."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


def get_dataset_info(dataset_name: str) -> dict[str, Any]:
    """
    Get comprehensive information about a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary containing dataset information
    """
    dataset_info = {
        "mnist": {
            "name": "MNIST",
            "input_shape": (1, 28, 28),
            "num_classes": 10,
            "num_samples": {"train": 60000, "test": 10000},
            "description": "MNIST handwritten digits (0-9)",
            "task_type": "classification",
            "data_type": "grayscale_image",
            "class_names": [str(i) for i in range(10)],
            "mean": (0.1307,),
            "std": (0.3081,),
            "download_size": "~11MB",
        },
        "cifar10": {
            "name": "CIFAR-10",
            "input_shape": (3, 32, 32),
            "num_classes": 10,
            "num_samples": {"train": 50000, "test": 10000},
            "description": "CIFAR-10 natural images (10 classes)",
            "task_type": "classification",
            "data_type": "rgb_image",
            "class_names": [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
            "download_size": "~170MB",
        },
        "medmnist": {
            "name": "MedMNIST (PathMNIST)",
            "input_shape": (1, 28, 28),
            "num_classes": 9,
            "num_samples": {"train": 89996, "test": 10004},
            "description": "PathMNIST medical pathology images",
            "task_type": "classification",
            "data_type": "grayscale_image",
            "class_names": [
                "adipose",
                "background",
                "debris",
                "lymphocytes",
                "mucus",
                "smooth_muscle",
                "normal_colon_mucosa",
                "cancer_associated_stroma",
                "colorectal_adenocarcinoma",
            ],
            "mean": (0.485,),
            "std": (0.229,),
            "download_size": "~7MB",
        },
        "synthetic": {
            "name": "Synthetic",
            "input_shape": (784,),
            "num_classes": 10,
            "num_samples": {"train": 1000, "test": 200},
            "description": "Synthetic random data for testing",
            "task_type": "classification",
            "data_type": "vector",
            "class_names": [f"class_{i}" for i in range(10)],
            "mean": (0.0,),
            "std": (1.0,),
            "download_size": "Generated",
        },
    }

    return dataset_info.get(
        dataset_name.lower(),
        {
            "name": f"Unknown ({dataset_name})",
            "input_shape": "unknown",
            "num_classes": "unknown",
            "num_samples": "unknown",
            "description": f"Unknown dataset: {dataset_name}",
            "task_type": "unknown",
            "data_type": "unknown",
            "class_names": [],
            "mean": (0.0,),
            "std": (1.0,),
            "download_size": "unknown",
        },
    )


def calculate_dataset_stats(dataset) -> dict[str, float]:
    """
    Calculate mean and standard deviation statistics for a dataset.

    Args:
        dataset: PyTorch dataset

    Returns:
        Dictionary with mean and std statistics
    """
    from torch.utils.data import DataLoader

    # Create temporary dataloader
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Calculate statistics
    mean = torch.zeros(3)  # Assume max 3 channels
    std = torch.zeros(3)
    total_samples = 0

    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)

        # Update running mean and std
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    # Return only non-zero channels
    num_channels = data.size(1)
    return {
        "mean": mean[:num_channels].tolist(),
        "std": std[:num_channels].tolist(),
        "num_channels": num_channels,
        "total_samples": total_samples,
    }


def validate_dataset_compatibility(
    dataset_name: str, model_input_shape: tuple[int, ...]
) -> bool:
    """
    Validate that a dataset is compatible with a model's input shape.

    Args:
        dataset_name: Name of the dataset
        model_input_shape: Expected input shape of the model

    Returns:
        True if compatible, False otherwise
    """
    dataset_info = get_dataset_info(dataset_name)
    dataset_shape = dataset_info.get("input_shape")

    if dataset_shape == "unknown":
        logger.warning(f"Unknown dataset shape for {dataset_name}")
        return True  # Assume compatible if unknown

    # For vector data, check if shapes match or can be flattened
    if isinstance(dataset_shape, tuple) and isinstance(model_input_shape, tuple):
        # Calculate total elements
        dataset_elements = 1
        for dim in dataset_shape:
            dataset_elements *= dim

        model_elements = 1
        for dim in model_input_shape:
            model_elements *= dim

        if dataset_elements == model_elements:
            return True

        # Check if shapes match exactly
        if dataset_shape == model_input_shape:
            return True

    logger.warning(
        f"Dataset shape {dataset_shape} may not be compatible with "
        f"model input shape {model_input_shape}"
    )
    return False


def create_federated_split_strategy(
    strategy: str = "iid",
    alpha: float = 0.5,
    min_samples: int = 10,
) -> dict[str, Any]:
    """
    Create configuration for federated data splitting strategies.

    Args:
        strategy: Splitting strategy ('iid', 'non_iid', 'dirichlet')
        alpha: Concentration parameter for Dirichlet distribution
        min_samples: Minimum samples per client

    Returns:
        Configuration dictionary for the splitting strategy
    """
    strategies = {
        "iid": {
            "name": "Independent and Identically Distributed",
            "description": "Random uniform distribution across clients",
            "parameters": {"min_samples": min_samples},
        },
        "non_iid": {
            "name": "Non-IID Class-based",
            "description": "Each client gets data from limited classes",
            "parameters": {"min_samples": min_samples},
        },
        "dirichlet": {
            "name": "Dirichlet Distribution",
            "description": "Class distribution follows Dirichlet distribution",
            "parameters": {"alpha": alpha, "min_samples": min_samples},
        },
    }

    if strategy not in strategies:
        logger.warning(f"Unknown strategy '{strategy}', using 'iid'")
        strategy = "iid"

    return {
        "strategy": strategy,
        **strategies[strategy],
    }


def print_dataset_summary(dataset_name: str, num_clients: int = 1) -> None:
    """
    Print a comprehensive summary of dataset information.

    Args:
        dataset_name: Name of the dataset
        num_clients: Number of federated clients (for partition info)
    """
    info = get_dataset_info(dataset_name)

    print(f"\n{'=' * 60}")
    print(f"DATASET SUMMARY: {info['name']}")
    print(f"{'=' * 60}")
    print(f"Description: {info['description']}")
    print(f"Task Type: {info['task_type']}")
    print(f"Data Type: {info['data_type']}")
    print(f"Input Shape: {info['input_shape']}")
    print(f"Number of Classes: {info['num_classes']}")

    if isinstance(info["num_samples"], dict):
        print(f"Training Samples: {info['num_samples'].get('train', 'N/A')}")
        print(f"Test Samples: {info['num_samples'].get('test', 'N/A')}")
    else:
        print(f"Total Samples: {info['num_samples']}")

    print(f"Download Size: {info['download_size']}")

    if num_clients > 1:
        train_samples = info["num_samples"].get("train", info["num_samples"])
        if isinstance(train_samples, int):
            samples_per_client = train_samples // num_clients
            print("\nFEDERATED SETUP:")
            print(f"Clients: {num_clients}")
            print(f"Samples per Client: ~{samples_per_client}")

    if info["class_names"] and len(info["class_names"]) <= 20:
        print(f"\nClass Names: {', '.join(info['class_names'])}")

    print(f"{'=' * 60}\n")


# Legacy compatibility functions
def load_dataset(*args, **kwargs):
    """Legacy function for backward compatibility."""
    logger.warning(
        "load_dataset function is deprecated. Use FederatedDataLoader class instead."
    )
    from .dataloader import FederatedDataLoader

    # Extract parameters from legacy call
    dataset_name = args[0] if args else kwargs.get("dataset_name", "synthetic")
    partition_id = args[1] if len(args) > 1 else kwargs.get("partition_id")
    num_partitions = kwargs.get("num_partitions", 5)
    val_split = kwargs.get("val_split", 0.2)

    # Create FederatedDataLoader
    loader = FederatedDataLoader(
        dataset_name=dataset_name,
        num_clients=num_partitions,
        val_split=val_split,
    )

    if partition_id is not None:
        # Return specific client data
        client_datasets = loader.create_client_datasets()
        return client_datasets[partition_id]
    else:
        # Return full dataset
        return loader.load_dataset()
