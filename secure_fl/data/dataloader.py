"""
Federated Data Loading and Partitioning Module

This module provides the FederatedDataLoader class for loading, preprocessing,
and partitioning datasets for federated learning experiments. It supports
multiple popular datasets and provides flexible partitioning strategies.

Features:
- Multiple dataset support (MNIST, CIFAR-10, MedMNIST, Synthetic)
- IID and Non-IID data partitioning
- Automatic data downloading and preprocessing
- Consistent interfaces for different datasets
- Client-specific data distribution strategies
"""

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

logger = logging.getLogger(__name__)


class FederatedDataLoader:
    """
    Main class for loading and partitioning datasets for federated learning.

    This class provides a unified interface for loading different datasets,
    partitioning them across federated clients, and creating data loaders
    with appropriate transformations.
    """

    SUPPORTED_DATASETS = ["mnist", "cifar10", "medmnist", "synthetic"]

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "./_data",
        num_clients: int = 5,
        iid: bool = True,
        val_split: float = 0.2,
        batch_size: int = 32,
        seed: int = 42,
    ):
        """
        Initialize FederatedDataLoader.

        Args:
            dataset_name: Name of dataset to load
            data_dir: Directory to store/load data
            num_clients: Number of federated clients
            iid: Whether to use IID (True) or Non-IID (False) partitioning
            val_split: Validation split ratio (0.0 to 1.0)
            batch_size: Default batch size for data loaders
            seed: Random seed for reproducible partitioning
        """
        if dataset_name.lower() not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                f"Supported: {self.SUPPORTED_DATASETS}"
            )

        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.num_clients = num_clients
        self.iid = iid
        self.val_split = val_split
        self.batch_size = batch_size
        self.seed = seed

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed for reproducibility
        torch.manual_seed(seed)

        # Cache for loaded datasets
        self._train_dataset = None
        self._test_dataset = None
        self._client_datasets = None

        logger.info(
            f"Initialized FederatedDataLoader for {dataset_name} with "
            f"{num_clients} clients ({'IID' if iid else 'Non-IID'})"
        )

    def load_dataset(self) -> tuple[Dataset, Dataset | None]:
        """
        Load the full dataset (train and optional test).

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if self._train_dataset is None:
            logger.info(f"Loading {self.dataset_name} dataset...")

            if self.dataset_name == "mnist":
                self._train_dataset, self._test_dataset = self._load_mnist()
            elif self.dataset_name == "cifar10":
                self._train_dataset, self._test_dataset = self._load_cifar10()
            elif self.dataset_name == "medmnist":
                self._train_dataset, self._test_dataset = self._load_medmnist()
            elif self.dataset_name == "synthetic":
                self._train_dataset, self._test_dataset = self._load_synthetic()

            logger.info(
                f"Loaded {self.dataset_name}: "
                f"Train={len(self._train_dataset)}, Test={len(self._test_dataset) if self._test_dataset else 0}"
            )

        return self._train_dataset, self._test_dataset

    def create_client_datasets(self) -> list[tuple[Dataset, Dataset | None]]:
        """
        Create datasets for each federated client.

        Returns:
            List of (train_dataset, val_dataset) tuples for each client
        """
        if self._client_datasets is None:
            train_dataset, _ = self.load_dataset()

            if self.iid:
                self._client_datasets = self._create_iid_partitions(train_dataset)
            else:
                self._client_datasets = self._create_non_iid_partitions(train_dataset)

            logger.info(f"Created datasets for {len(self._client_datasets)} clients")

        return self._client_datasets

    def create_client_dataloaders(
        self,
        client_id: int | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
    ) -> (
        list[tuple[DataLoader, DataLoader | None]]
        | tuple[DataLoader, DataLoader | None]
    ):
        """
        Create data loaders for clients.

        Args:
            client_id: Specific client ID (if None, returns all clients)
            batch_size: Override default batch size
            shuffle: Whether to shuffle training data

        Returns:
            DataLoader(s) for specified client(s)
        """
        client_datasets = self.create_client_datasets()
        batch_size = batch_size or self.batch_size

        if client_id is not None:
            # Return loaders for specific client
            if client_id >= len(client_datasets):
                raise ValueError(f"Client ID {client_id} >= {len(client_datasets)}")

            train_data, val_data = client_datasets[client_id]
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )
            val_loader = (
                DataLoader(val_data, batch_size=batch_size, shuffle=False)
                if val_data
                else None
            )

            return train_loader, val_loader

        # Return loaders for all clients
        client_loaders = []
        for train_data, val_data in client_datasets:
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )
            val_loader = (
                DataLoader(val_data, batch_size=batch_size, shuffle=False)
                if val_data
                else None
            )
            client_loaders.append((train_loader, val_loader))

        return client_loaders

    def get_dataset_info(self) -> dict[str, Any]:
        """Get information about the loaded dataset."""
        info_map = {
            "mnist": {
                "input_shape": (1, 28, 28),
                "num_classes": 10,
                "num_samples": 60000,
                "description": "MNIST handwritten digits",
            },
            "cifar10": {
                "input_shape": (3, 32, 32),
                "num_classes": 10,
                "num_samples": 50000,
                "description": "CIFAR-10 natural images",
            },
            "medmnist": {
                "input_shape": (1, 28, 28),
                "num_classes": 9,
                "num_samples": 89996,
                "description": "PathMNIST medical images",
            },
            "synthetic": {
                "input_shape": (784,),
                "num_classes": 10,
                "num_samples": 1000,
                "description": "Synthetic random data",
            },
        }

        return info_map.get(
            self.dataset_name,
            {
                "input_shape": "unknown",
                "num_classes": "unknown",
                "num_samples": "unknown",
                "description": f"Unknown dataset: {self.dataset_name}",
            },
        )

    def _load_mnist(self) -> tuple[Dataset, Dataset]:
        """Load MNIST dataset."""
        try:
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError("torchvision is required for MNIST dataset")

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.MNIST(
            root=str(self.data_dir), train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=str(self.data_dir), train=False, download=True, transform=transform
        )

        return train_dataset, test_dataset

    def _load_cifar10(self) -> tuple[Dataset, Dataset]:
        """Load CIFAR-10 dataset."""
        try:
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError("torchvision is required for CIFAR-10 dataset")

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_dataset = datasets.CIFAR10(
            root=str(self.data_dir), train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=str(self.data_dir), train=False, download=True, transform=transform
        )

        return train_dataset, test_dataset

    def _load_medmnist(self) -> tuple[Dataset, Dataset]:
        """Load MedMNIST dataset."""
        try:
            import medmnist
        except ImportError:
            raise ImportError(
                "medmnist is required for MedMNIST dataset. Install with: pip install medmnist"
            )

        # Use PathMNIST as default MedMNIST dataset
        DataClass = medmnist.PathMNIST

        train_dataset = DataClass(split="train", download=True, root=str(self.data_dir))
        test_dataset = DataClass(split="test", download=True, root=str(self.data_dir))

        # Convert to tensor datasets for consistency
        def convert_medmnist_to_tensor(dataset):
            data = torch.tensor(dataset.imgs, dtype=torch.float32) / 255.0
            labels = torch.tensor(dataset.labels.squeeze(), dtype=torch.long)

            # Add channel dimension if missing
            if len(data.shape) == 3:
                data = data.unsqueeze(1)

            return TensorDataset(data, labels)

        train_tensor_dataset = convert_medmnist_to_tensor(train_dataset)
        test_tensor_dataset = convert_medmnist_to_tensor(test_dataset)

        return train_tensor_dataset, test_tensor_dataset

    def _load_synthetic(
        self, num_samples: int = 1000, input_dim: int = 784, num_classes: int = 10
    ) -> tuple[Dataset, Dataset]:
        """Load synthetic dataset."""
        # Generate synthetic training data
        torch.manual_seed(self.seed)
        X_train = torch.randn(num_samples, input_dim)
        y_train = torch.randint(0, num_classes, (num_samples,))
        train_dataset = TensorDataset(X_train, y_train)

        # Generate synthetic test data
        X_test = torch.randn(num_samples // 5, input_dim)
        y_test = torch.randint(0, num_classes, (num_samples // 5,))
        test_dataset = TensorDataset(X_test, y_test)

        return train_dataset, test_dataset

    def _create_iid_partitions(
        self, dataset: Dataset
    ) -> list[tuple[Dataset, Dataset | None]]:
        """Create IID data partitions for clients."""
        total_size = len(dataset)
        partition_size = total_size // self.num_clients

        # Create partition sizes
        partition_sizes = [partition_size] * self.num_clients
        # Add remaining samples to last partition
        partition_sizes[-1] += total_size % self.num_clients

        # Split dataset into partitions
        partitions = random_split(dataset, partition_sizes)

        # Split each partition into train/val
        client_datasets = []
        for partition in partitions:
            if self.val_split > 0:
                partition_size = len(partition)
                train_size = int((1 - self.val_split) * partition_size)
                val_size = partition_size - train_size

                train_data, val_data = random_split(partition, [train_size, val_size])
                client_datasets.append((train_data, val_data))
            else:
                client_datasets.append((partition, None))

        return client_datasets

    def _create_non_iid_partitions(
        self, dataset: Dataset
    ) -> list[tuple[Dataset, Dataset | None]]:
        """Create Non-IID data partitions for clients."""
        # For Non-IID, we'll create class-based partitions
        # This is a simplified Non-IID strategy where each client gets data from limited classes

        try:
            # Get labels from dataset
            if hasattr(dataset, "targets"):
                labels = torch.tensor(dataset.targets)
            elif hasattr(dataset, "tensors"):
                labels = dataset.tensors[1]  # For TensorDataset
            else:
                # Fallback: extract labels manually
                labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])

            num_classes = len(torch.unique(labels))
            classes_per_client = max(1, num_classes // self.num_clients)

            client_datasets = []

            for client_id in range(self.num_clients):
                # Assign classes to this client
                start_class = (client_id * classes_per_client) % num_classes
                client_classes = []

                for i in range(classes_per_client):
                    class_id = (start_class + i) % num_classes
                    client_classes.append(class_id)

                # Get indices for client's classes
                client_indices = []
                for class_id in client_classes:
                    class_indices = torch.where(labels == class_id)[0]
                    client_indices.extend(class_indices.tolist())

                # Create subset for this client
                client_data = torch.utils.data.Subset(dataset, client_indices)

                # Split into train/val
                if self.val_split > 0:
                    client_size = len(client_data)
                    train_size = int((1 - self.val_split) * client_size)
                    val_size = client_size - train_size

                    train_data, val_data = random_split(
                        client_data, [train_size, val_size]
                    )
                    client_datasets.append((train_data, val_data))
                else:
                    client_datasets.append((client_data, None))

            return client_datasets

        except Exception as e:
            logger.warning(
                f"Failed to create Non-IID partitions: {e}. Falling back to IID."
            )
            return self._create_iid_partitions(dataset)

    def get_client_stats(self) -> dict[str, Any]:
        """Get statistics about client data distribution."""
        client_datasets = self.create_client_datasets()

        stats = {
            "num_clients": len(client_datasets),
            "partitioning": "IID" if self.iid else "Non-IID",
            "val_split": self.val_split,
            "client_sizes": [],
        }

        for i, (train_data, val_data) in enumerate(client_datasets):
            train_size = len(train_data) if train_data else 0
            val_size = len(val_data) if val_data else 0

            stats["client_sizes"].append(
                {
                    "client_id": i,
                    "train_samples": train_size,
                    "val_samples": val_size,
                    "total_samples": train_size + val_size,
                }
            )

        return stats

    def __repr__(self) -> str:
        return (
            f"FederatedDataLoader(dataset='{self.dataset_name}', "
            f"num_clients={self.num_clients}, "
            f"iid={self.iid}, "
            f"val_split={self.val_split})"
        )
