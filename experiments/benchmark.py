"""
Enhanced Multi-Dataset Benchmark for Secure FL

This module provides comprehensive benchmarking utilities to evaluate
the Secure FL framework across multiple datasets and their corresponding models.

Supported datasets and models:
- MNIST: MNISTModel (fully connected)
- CIFAR-10: CIFAR10Model (CNN)
- Synthetic: SimpleModel (configurable MLP)

Features:
- Multi-dataset support with appropriate models
- Performance comparison across configurations
- ZKP overhead analysis
- Convergence analysis
- Rich visualizations
"""

import argparse
import json
import logging

# Import Secure FL components
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent.parent))

from secure_fl.client import create_client
from secure_fl.models import (
    CIFAR10Model,
    FlexibleMLP,
    MNISTModel,
    SimpleModel,
)
from secure_fl.server import create_server_strategy
from secure_fl.utils import (
    ndarrays_to_torch,
)

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages multiple datasets and their corresponding models"""

    def __init__(self, data_dir: str = "./_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.dataset_configs = {
            # Image datasets
            "mnist": {
                "model_class": MNISTModel,
                "model_kwargs": {"hidden_dims": [128, 64], "output_dim": 10},
                "input_shape": (1, 28, 28),
                "num_classes": 10,
                "data_loader": self._load_mnist_data,
                "description": "MNIST handwritten digits (28x28 grayscale)",
            },
            "fashion_mnist": {
                "model_class": MNISTModel,
                "model_kwargs": {"hidden_dims": [256, 128], "output_dim": 10},
                "input_shape": (1, 28, 28),
                "num_classes": 10,
                "data_loader": self._load_fashion_mnist_data,
                "description": "Fashion-MNIST clothing items (28x28 grayscale)",
            },
            "cifar10": {
                "model_class": CIFAR10Model,
                "model_kwargs": {"num_classes": 10},
                "input_shape": (3, 32, 32),
                "num_classes": 10,
                "data_loader": self._load_cifar10_data,
                "description": "CIFAR-10 natural images (32x32 RGB)",
            },
            "cifar100": {
                "model_class": CIFAR10Model,  # Reuse architecture but with 100 classes
                "model_kwargs": {"num_classes": 100},
                "input_shape": (3, 32, 32),
                "num_classes": 100,
                "data_loader": self._load_cifar100_data,
                "description": "CIFAR-100 natural images (32x32 RGB, 100 classes)",
            },
            # Synthetic datasets with different models
            "synthetic": {
                "model_class": SimpleModel,
                "model_kwargs": {
                    "input_dim": 784,
                    "hidden_dims": [128, 64],
                    "output_dim": 10,
                },
                "input_shape": (784,),
                "num_classes": 10,
                "data_loader": self._load_synthetic_data,
                "description": "Synthetic data for basic testing",
            },
            "synthetic_large": {
                "model_class": FlexibleMLP,
                "model_kwargs": {
                    "input_dim": 1024,
                    "hidden_dims": [512, 256, 128],
                    "output_dim": 20,
                    "activation": "relu",
                    "dropout_rate": 0.2,
                    "use_batch_norm": True,
                },
                "input_shape": (1024,),
                "num_classes": 20,
                "data_loader": self._load_synthetic_large_data,
                "description": "Large synthetic dataset with 20 classes",
            },
            "synthetic_small": {
                "model_class": FlexibleMLP,
                "model_kwargs": {
                    "input_dim": 100,
                    "hidden_dims": [64, 32],
                    "output_dim": 5,
                    "activation": "tanh",
                    "dropout_rate": 0.1,
                },
                "input_shape": (100,),
                "num_classes": 5,
                "data_loader": self._load_synthetic_small_data,
                "description": "Small synthetic dataset for quick testing",
            },
            # Text-like datasets (using flexible MLP)
            "text_classification": {
                "model_class": FlexibleMLP,
                "model_kwargs": {
                    "input_dim": 300,  # Word embedding dimension
                    "hidden_dims": [256, 128, 64],
                    "output_dim": 4,  # Sentiment classes
                    "activation": "gelu",
                    "dropout_rate": 0.3,
                    "use_batch_norm": True,
                },
                "input_shape": (300,),
                "num_classes": 4,
                "data_loader": self._load_text_classification_data,
                "description": "Simulated text classification (4 sentiment classes)",
            },
            # Medical-like datasets
            "medical": {
                "model_class": FlexibleMLP,
                "model_kwargs": {
                    "input_dim": 256,
                    "hidden_dims": [128, 64, 32],
                    "output_dim": 3,  # Disease classes
                    "activation": "relu",
                    "dropout_rate": 0.4,
                    "use_batch_norm": True,
                },
                "input_shape": (256,),
                "num_classes": 3,
                "data_loader": self._load_medical_data,
                "description": "Simulated medical diagnosis (3 disease classes)",
            },
            # Financial-like datasets
            "financial": {
                "model_class": FlexibleMLP,
                "model_kwargs": {
                    "input_dim": 50,  # Financial features
                    "hidden_dims": [128, 64],
                    "output_dim": 2,  # Binary classification
                    "activation": "leaky_relu",
                    "dropout_rate": 0.2,
                    "final_activation": "sigmoid",
                },
                "input_shape": (50,),
                "num_classes": 2,
                "data_loader": self._load_financial_data,
                "description": "Simulated financial fraud detection (binary)",
            },
        }

    def list_available_datasets(self) -> dict[str, str]:
        """List all available datasets with descriptions"""
        return {
            name: config["description"] for name, config in self.dataset_configs.items()
        }

    def get_dataset_categories(self) -> dict[str, list[str]]:
        """Get datasets organized by category"""
        categories = {
            "Image Datasets": ["mnist", "fashion_mnist", "cifar10", "cifar100"],
            "Synthetic Datasets": ["synthetic", "synthetic_large", "synthetic_small"],
            "Domain-Specific": ["text_classification", "medical", "financial"],
        }
        return categories

    def get_model(self, dataset_name: str) -> nn.Module:
        """Get appropriate model for dataset"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = self.dataset_configs[dataset_name]
        return config["model_class"](**config["model_kwargs"])

    def get_model_fn(self, dataset_name: str):
        """Get model function for dataset"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = self.dataset_configs[dataset_name]

        def model_fn():
            return config["model_class"](**config["model_kwargs"])

        return model_fn

    def get_dataset_info(self, dataset_name: str) -> dict[str, Any]:
        """Get dataset configuration information"""
        info = self.dataset_configs[dataset_name].copy()
        # Make JSON serializable by removing non-serializable items
        info["model_class"] = info["model_class"].__name__
        # Remove function references that can't be serialized
        info.pop("data_loader", None)
        return info

    def load_federated_data(
        self, dataset_name: str, num_clients: int = 3, iid: bool = True
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Load federated data splits for specified dataset"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        loader_func = self.dataset_configs[dataset_name]["data_loader"]
        return loader_func(num_clients, iid)

    def _load_mnist_data(
        self, num_clients: int, iid: bool
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Load MNIST data split across clients"""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Download MNIST if not exists
        train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=transform
        )

        return self._create_federated_splits(
            train_dataset, test_dataset, num_clients, iid
        )

    def _load_fashion_mnist_data(
        self, num_clients: int, iid: bool
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Load Fashion-MNIST data split across clients"""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        )

        # Download Fashion-MNIST if not exists
        train_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=False, download=True, transform=transform
        )

        return self._create_federated_splits(
            train_dataset, test_dataset, num_clients, iid
        )

    def _load_cifar100_data(
        self, num_clients: int, iid: bool
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Load CIFAR-100 data split across clients"""
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)
                ),
            ]
        )

        train_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=transform_test
        )

        return self._create_federated_splits(
            train_dataset, test_dataset, num_clients, iid, num_classes=100
        )

    def _load_cifar10_data(
        self, num_clients: int, iid: bool
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Load CIFAR-10 data split across clients"""
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=transform_test
        )

        return self._create_federated_splits(
            train_dataset, test_dataset, num_clients, iid
        )

    def _load_synthetic_data(
        self, num_clients: int, iid: bool
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Generate synthetic federated data"""
        # Generate synthetic data
        total_samples = 2000
        samples_per_client = total_samples // num_clients
        input_dim = 784
        num_classes = 10

        client_loaders = []

        for client_id in range(num_clients):
            if iid:
                # IID: each client gets random samples from all classes
                X = torch.randn(samples_per_client, input_dim)
                y = torch.randint(0, num_classes, (samples_per_client,))
            else:
                # Non-IID: each client gets 2-3 classes
                classes_per_client = 2 + (client_id % 2)  # 2 or 3 classes
                start_class = (client_id * 2) % num_classes
                client_classes = [
                    (start_class + i) % num_classes for i in range(classes_per_client)
                ]

                X = torch.randn(samples_per_client, input_dim)
                y = torch.tensor(
                    [
                        client_classes[i % len(client_classes)]
                        for i in range(samples_per_client)
                    ]
                )

            # Split into train/val
            split_idx = int(0.8 * samples_per_client)

            train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
            val_dataset = TensorDataset(X[split_idx:], y[split_idx:])

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            client_loaders.append((train_loader, val_loader))

        return client_loaders

    def _load_synthetic_large_data(
        self, num_clients: int, iid: bool
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Generate large synthetic federated data"""
        total_samples = 5000
        samples_per_client = total_samples // num_clients
        input_dim = 1024
        num_classes = 20

        client_loaders = []

        for client_id in range(num_clients):
            if iid:
                # IID: each client gets random samples from all classes
                X = torch.randn(samples_per_client, input_dim) * 2 + 1
                y = torch.randint(0, num_classes, (samples_per_client,))
            else:
                # Non-IID: each client gets 4-6 classes
                classes_per_client = 4 + (client_id % 3)  # 4, 5, or 6 classes
                start_class = (client_id * 3) % num_classes
                client_classes = [
                    (start_class + i) % num_classes for i in range(classes_per_client)
                ]

                X = torch.randn(samples_per_client, input_dim) * 1.5 + 0.5
                y = torch.tensor(
                    [
                        client_classes[i % len(client_classes)]
                        for i in range(samples_per_client)
                    ]
                )

            # Split into train/val
            split_idx = int(0.8 * samples_per_client)

            train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
            val_dataset = TensorDataset(X[split_idx:], y[split_idx:])

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            client_loaders.append((train_loader, val_loader))

        return client_loaders

    def _load_synthetic_small_data(
        self, num_clients: int, iid: bool
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Generate small synthetic federated data for quick testing"""
        total_samples = 500
        samples_per_client = max(50, total_samples // num_clients)
        input_dim = 100
        num_classes = 5

        client_loaders = []

        for client_id in range(num_clients):
            if iid:
                X = torch.randn(samples_per_client, input_dim)
                y = torch.randint(0, num_classes, (samples_per_client,))
            else:
                # Non-IID: each client gets 2-3 classes
                classes_per_client = 2 + (client_id % 2)
                start_class = (client_id * 2) % num_classes
                client_classes = [
                    (start_class + i) % num_classes for i in range(classes_per_client)
                ]

                X = torch.randn(samples_per_client, input_dim) * 0.8
                y = torch.tensor(
                    [
                        client_classes[i % len(client_classes)]
                        for i in range(samples_per_client)
                    ]
                )

            # Split into train/val
            split_idx = int(0.8 * samples_per_client)

            train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
            val_dataset = TensorDataset(X[split_idx:], y[split_idx:])

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

            client_loaders.append((train_loader, val_loader))

        return client_loaders

    def _load_text_classification_data(
        self, num_clients: int, iid: bool
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Generate simulated text classification data (e.g., sentiment analysis)"""
        total_samples = 3000
        samples_per_client = total_samples // num_clients
        input_dim = 300  # Simulating word embeddings
        num_classes = 4  # Positive, Negative, Neutral, Mixed

        client_loaders = []

        for client_id in range(num_clients):
            if iid:
                # Random word embeddings
                X = torch.randn(samples_per_client, input_dim) * 0.5
                y = torch.randint(0, num_classes, (samples_per_client,))
            else:
                # Non-IID: clients might specialize in certain sentiments
                primary_classes = [
                    (client_id * 2) % num_classes,
                    (client_id * 2 + 1) % num_classes,
                ]

                X = torch.randn(samples_per_client, input_dim) * 0.3
                # 70% from primary classes, 30% from others
                primary_samples = int(0.7 * samples_per_client)
                y_primary = torch.randint(0, len(primary_classes), (primary_samples,))
                y_primary = torch.tensor([primary_classes[i] for i in y_primary])
                y_others = torch.randint(
                    0, num_classes, (samples_per_client - primary_samples,)
                )
                y = torch.cat([y_primary, y_others])

            # Split into train/val
            split_idx = int(0.8 * samples_per_client)

            train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
            val_dataset = TensorDataset(X[split_idx:], y[split_idx:])

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            client_loaders.append((train_loader, val_loader))

        return client_loaders

    def _load_medical_data(
        self, num_clients: int, iid: bool
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Generate simulated medical diagnosis data"""
        total_samples = 2000
        samples_per_client = total_samples // num_clients
        input_dim = 256  # Medical features (symptoms, test results, etc.)
        num_classes = 3  # Healthy, Disease A, Disease B

        client_loaders = []

        for client_id in range(num_clients):
            if iid:
                # Random medical features
                X = torch.abs(torch.randn(samples_per_client, input_dim)) * 2
                y = torch.randint(0, num_classes, (samples_per_client,))
            else:
                # Non-IID: different hospitals might see different disease prevalence
                if client_id % 3 == 0:
                    # Hospital specializing in Disease A
                    disease_dist = [0.3, 0.6, 0.1]
                elif client_id % 3 == 1:
                    # Hospital specializing in Disease B
                    disease_dist = [0.3, 0.1, 0.6]
                else:
                    # General hospital
                    disease_dist = [0.5, 0.25, 0.25]

                X = torch.abs(torch.randn(samples_per_client, input_dim)) * 1.5
                y = torch.multinomial(
                    torch.tensor(disease_dist), samples_per_client, replacement=True
                )

            # Split into train/val
            split_idx = int(0.8 * samples_per_client)

            train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
            val_dataset = TensorDataset(X[split_idx:], y[split_idx:])

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            client_loaders.append((train_loader, val_loader))

        return client_loaders

    def _load_financial_data(
        self, num_clients: int, iid: bool
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Generate simulated financial fraud detection data"""
        total_samples = 4000
        samples_per_client = total_samples // num_clients
        input_dim = 50  # Financial features

        client_loaders = []

        for client_id in range(num_clients):
            if iid:
                X = torch.randn(samples_per_client, input_dim)
                y = torch.bernoulli(
                    torch.full((samples_per_client,), 0.1)
                ).long()  # 10% fraud
            else:
                # Non-IID: different institutions might have different fraud rates
                if client_id % 3 == 0:
                    # High-risk institution
                    fraud_rate = 0.2
                elif client_id % 3 == 1:
                    # Medium-risk institution
                    fraud_rate = 0.1
                else:
                    # Low-risk institution
                    fraud_rate = 0.05

                X = torch.randn(samples_per_client, input_dim)
                y = torch.bernoulli(
                    torch.full((samples_per_client,), fraud_rate)
                ).long()

            # Split into train/val
            split_idx = int(0.8 * samples_per_client)

            train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
            val_dataset = TensorDataset(X[split_idx:], y[split_idx:])

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            client_loaders.append((train_loader, val_loader))

        return client_loaders

    def _create_federated_splits(
        self,
        train_dataset,
        test_dataset,
        num_clients: int,
        iid: bool,
        num_classes: int = 10,
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Create federated data splits from datasets"""
        total_train = len(train_dataset)
        samples_per_client = total_train // num_clients

        client_loaders = []

        if iid:
            # IID: randomly distribute samples
            indices = torch.randperm(total_train)

            for i in range(num_clients):
                start_idx = i * samples_per_client
                end_idx = start_idx + samples_per_client
                client_indices = indices[start_idx:end_idx]

                client_train = torch.utils.data.Subset(train_dataset, client_indices)

                # Use same test set for all clients (could be split too)
                train_loader = DataLoader(client_train, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                client_loaders.append((train_loader, test_loader))
        else:
            # Non-IID: distribute by class

            # Group samples by class
            class_indices = {i: [] for i in range(num_classes)}
            for idx, (_, label) in enumerate(train_dataset):
                class_indices[label].append(idx)

            # Distribute classes to clients
            classes_per_client = max(1, num_classes // num_clients)

            for i in range(num_clients):
                client_indices = []

                # Assign classes to this client
                start_class = (i * classes_per_client) % num_classes
                client_classes = [
                    (start_class + j) % num_classes for j in range(classes_per_client)
                ]

                # Add additional classes if needed
                if len(client_classes) < 2:  # Ensure at least 2 classes per client
                    additional_class = (start_class + classes_per_client) % num_classes
                    client_classes.append(additional_class)

                # Collect indices for assigned classes
                for class_id in client_classes:
                    class_size = len(class_indices[class_id])
                    samples_from_class = min(
                        class_size, samples_per_client // len(client_classes)
                    )
                    client_indices.extend(class_indices[class_id][:samples_from_class])

                client_train = torch.utils.data.Subset(train_dataset, client_indices)

                train_loader = DataLoader(client_train, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                client_loaders.append((train_loader, test_loader))

        return client_loaders


class MultiDatasetBenchmarkRunner:
    """Enhanced benchmark runner supporting multiple datasets and models"""

    def __init__(self, output_dir: str = "results/multi_dataset_benchmark"):
        self.base_output_dir = Path(output_dir)
        self.output_dir = None  # Will be set based on datasets and configs
        self.dataset_manager = DatasetManager()
        self.results = {}

    def run_comprehensive_benchmark(
        self, datasets: list[str] = None, configs: list[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Run comprehensive benchmark across multiple datasets using actual SecureFL"""

        if datasets is None:
            datasets = ["mnist", "cifar10", "synthetic"]

        if configs is None:
            configs = self.get_default_benchmark_configs()

        # Create organized output directory based on datasets and configs
        self.output_dir = self._create_output_directory(datasets, configs)

        logger.info(f"Running multi-dataset SecureFL benchmark on {datasets}")
        logger.info(f"Configurations: {[c['name'] for c in configs]}")
        logger.info(f"Results will be saved to: {self.output_dir}")

        all_results = {}

        for dataset in datasets:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"BENCHMARKING DATASET: {dataset.upper()}")
            logger.info(f"{'=' * 60}")

            dataset_results = {}

            for config in configs:
                config_name = f"{dataset}_{config['name']}"
                logger.info(f"\nRunning config: {config_name}")

                try:
                    # Always use secure-fl benchmark with actual components
                    result = self._run_secure_fl_single_dataset(dataset, config)
                    dataset_results[config["name"]] = result
                    logger.info(f"✓ Completed {config_name}")

                except Exception as e:
                    logger.error(f"✗ Failed {config_name}: {e}")
                    dataset_results[config["name"]] = {"error": str(e)}

            all_results[dataset] = dataset_results

        # Save results
        self.results = all_results
        self._save_results(all_results, datasets, configs)

        # Generate comprehensive visualizations
        self._generate_multi_dataset_plots(all_results)

        return all_results

    def _create_output_directory(
        self, datasets: list[str], configs: list[dict[str, Any]]
    ) -> Path:
        """Create organized output directory based on datasets and configs"""
        import datetime

        # Create timestamp for uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create dataset string (limit length for filesystem compatibility)
        if len(datasets) <= 3:
            datasets_str = "_".join(datasets)
        else:
            datasets_str = f"{len(datasets)}datasets"

        # Create config string (limit length)
        config_names = [c["name"] for c in configs]
        if len(config_names) <= 3:
            configs_str = "_".join(config_names)
        else:
            configs_str = f"{len(config_names)}configs"

        # Create organized directory structure
        run_name = f"{datasets_str}_{configs_str}_{timestamp}"
        output_dir = self.base_output_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organization
        (output_dir / "plots").mkdir(exist_ok=True)
        (output_dir / "raw_data").mkdir(exist_ok=True)

        return output_dir

    def _run_secure_fl_single_dataset(
        self, dataset: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Run benchmark using actual SecureFL components"""

        # Extract config parameters
        num_clients = config.get("num_clients", 3)
        num_rounds = config.get("num_rounds", 5)
        local_epochs = config.get("local_epochs", 3)
        learning_rate = config.get("learning_rate", 0.01)
        enable_zkp = config.get("enable_zkp", False)
        proof_rigor = config.get("proof_rigor", "medium")
        iid = config.get("iid", True)

        logger.info(f"SecureFL Benchmark - Dataset: {dataset}")
        logger.info(f"Clients: {num_clients}, Rounds: {num_rounds}, ZKP: {enable_zkp}")

        # Get dataset and model using DatasetManager
        dataset_info = self.dataset_manager.get_dataset_info(dataset)
        model_fn = self.dataset_manager.get_model_fn(dataset)

        # Load federated data
        federated_data = self.dataset_manager.load_federated_data(
            dataset, num_clients=num_clients, iid=iid
        )

        # Create server strategy
        strategy = create_server_strategy(
            model_fn=model_fn,
            momentum=0.9,
            learning_rate=learning_rate,
            enable_zkp=enable_zkp,
            proof_rigor=proof_rigor,
            min_available_clients=num_clients,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
        )

        # Create SecureFL clients
        clients = []
        for i in range(num_clients):
            train_loader, test_loader = federated_data[i]
            client = create_client(
                client_id=f"client_{i}",
                model_fn=model_fn,
                train_data=train_loader.dataset,
                val_data=test_loader.dataset if test_loader else None,
                batch_size=32,
                enable_zkp=enable_zkp,
                proof_rigor=proof_rigor,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                quantize_weights=False,
            )
            clients.append(client)

        # Initialize results tracking
        results = {
            "dataset": dataset,
            "model_class": dataset_info["model_class"],
            "config": config.copy(),
            "dataset_info": dataset_info,
            "round_accuracies": [],
            "round_losses": [],
            "communication_overhead": [],
            "proof_times": [],
            "training_time": 0,
            "memory_usage": [],
        }

        start_time = time.time()

        # Get initial parameters
        initial_model = model_fn()
        global_params = [
            param.detach().cpu().numpy() for param in initial_model.parameters()
        ]

        # Run federated learning rounds with actual SecureFL
        for round_num in range(num_rounds):
            logger.info(f"  Round {round_num + 1}/{num_rounds}")
            round_start = time.time()

            # Client training phase with actual SecureFL clients
            client_results = []
            round_proof_times = []

            for i, client in enumerate(clients):
                client_start = time.time()

                try:
                    # Use actual SecureFL client.fit method
                    fit_config = {"round": round_num + 1, "epochs": local_epochs}
                    updated_params, num_examples, client_metrics = client.fit(
                        global_params, fit_config
                    )

                    client_time = time.time() - client_start
                    training_time = client_metrics.get("training_time", client_time)
                    proof_time = client_metrics.get("proof_time", 0.0)

                    round_proof_times.append(proof_time)

                    client_results.append(
                        {
                            "client_id": f"client_{i}",
                            "num_examples": num_examples,
                            "parameters": updated_params,
                            "training_time": training_time,
                            "proof_time": proof_time,
                            "metrics": client_metrics,
                        }
                    )

                except Exception as e:
                    logger.error(f"Client {i} failed: {e}")
                    continue

            if not client_results:
                logger.error(f"No successful clients in round {round_num + 1}")
                continue

            # Server aggregation using strategy
            parameters_list = [
                (res["parameters"], res["num_examples"]) for res in client_results
            ]

            try:
                aggregated_params, server_metrics = strategy.aggregate_fit(
                    server_round=round_num + 1, results=parameters_list, failures=[]
                )
                global_params = aggregated_params
            except Exception as e:
                logger.error(f"Aggregation failed: {e}")
                continue

            # Evaluate global model
            try:
                # Create temporary model for evaluation
                eval_model = model_fn()
                ndarrays_to_torch(eval_model, global_params)
                eval_model.eval()

                # Use first client's test data for evaluation
                _, test_loader = federated_data[0]

                correct = 0
                total = 0
                total_loss = 0.0
                criterion = nn.CrossEntropyLoss()

                with torch.no_grad():
                    for data, target in test_loader:
                        if dataset == "synthetic":
                            data = data.view(data.size(0), -1)

                        outputs = eval_model(data)
                        loss = criterion(outputs, target)
                        total_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

                accuracy = 100 * correct / total if total > 0 else 0.0
                avg_loss = (
                    total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
                )

            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                accuracy = 0.0
                avg_loss = float("inf")

            # Calculate communication overhead
            comm_overhead = sum(param.nbytes for param in global_params) / 1024  # KB
            avg_proof_time = np.mean(round_proof_times) if round_proof_times else 0.0

            # Record results
            results["round_accuracies"].append(accuracy)
            results["round_losses"].append(avg_loss)
            results["communication_overhead"].append(comm_overhead)
            results["proof_times"].append(avg_proof_time)

            round_time = time.time() - round_start
            logger.info(f"    Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
            logger.info(
                f"    Proof time: {avg_proof_time:.3f}s, Round time: {round_time:.3f}s"
            )

        results["training_time"] = time.time() - start_time
        results["final_accuracy"] = (
            results["round_accuracies"][-1] if results["round_accuracies"] else 0.0
        )
        results["convergence_round"] = self._calculate_convergence_round(
            results["round_accuracies"]
        )

        return results

    def _calculate_convergence_round(
        self, accuracies: list[float], threshold: float = 0.95
    ) -> int:
        """Calculate round at which model converged"""
        if len(accuracies) < 2:
            return len(accuracies)

        max_accuracy = max(accuracies)
        target_accuracy = max_accuracy * threshold

        for i, acc in enumerate(accuracies):
            if acc >= target_accuracy:
                return i + 1

        return len(accuracies)

    def get_default_benchmark_configs(self) -> list[dict[str, Any]]:
        """Get default benchmark configurations for SecureFL"""
        return [
            {
                "name": "baseline_iid",
                "num_clients": 3,
                "num_rounds": 5,
                "local_epochs": 3,
                "learning_rate": 0.01,
                "enable_zkp": False,
                "iid": True,
                "description": "IID baseline without ZKP",
            },
            {
                "name": "baseline_non_iid",
                "num_clients": 3,
                "num_rounds": 5,
                "local_epochs": 3,
                "learning_rate": 0.01,
                "enable_zkp": False,
                "iid": False,
                "description": "Non-IID baseline without ZKP",
            },
            {
                "name": "zkp_medium_rigor",
                "num_clients": 3,
                "num_rounds": 5,
                "local_epochs": 3,
                "learning_rate": 0.01,
                "enable_zkp": True,
                "proof_rigor": "medium",
                "iid": False,
                "description": "SecureFL with medium ZKP rigor",
            },
            {
                "name": "zkp_low_rigor",
                "num_clients": 3,
                "num_rounds": 5,
                "local_epochs": 3,
                "learning_rate": 0.01,
                "enable_zkp": True,
                "proof_rigor": "low",
                "iid": False,
                "description": "SecureFL with low ZKP rigor",
            },
            {
                "name": "quick_test",
                "num_clients": 2,
                "num_rounds": 3,
                "local_epochs": 2,
                "learning_rate": 0.01,
                "enable_zkp": True,
                "proof_rigor": "low",
                "iid": True,
                "description": "Quick SecureFL test",
            },
        ]

    def _save_results(
        self,
        results: dict[str, Any],
        datasets: list[str],
        configs: list[dict[str, Any]],
    ):
        """Save benchmark results to JSON with organized structure"""

        # Save main results file
        results_file = self.output_dir / "raw_data" / "benchmark_results.json"

        # Create metadata
        metadata = {
            "experiment_info": {
                "timestamp": self.output_dir.name.split("_")[-1],
                "datasets": datasets,
                "configurations": [c["name"] for c in configs],
                "total_datasets": len(datasets),
                "total_configs": len(configs),
            },
            "results": {},
        }

        # Convert numpy arrays to lists for JSON serialization
        for dataset, dataset_configs in results.items():
            metadata["results"][dataset] = {}
            for config_name, result in dataset_configs.items():
                if isinstance(result, dict):
                    serialized_result = {}
                    for k, v in result.items():
                        if isinstance(v, np.ndarray):
                            serialized_result[k] = v.tolist()
                        elif isinstance(v, type):
                            # Convert class types to string
                            serialized_result[k] = v.__name__
                        elif hasattr(v, "__dict__"):
                            # Skip complex objects that can't be serialized
                            serialized_result[k] = str(v)
                        else:
                            serialized_result[k] = v
                    metadata["results"][dataset][config_name] = serialized_result
                else:
                    metadata["results"][dataset][config_name] = result

        with open(results_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Also save individual dataset results for easier analysis
        for dataset, dataset_results in results.items():
            dataset_file = self.output_dir / "raw_data" / f"{dataset}_results.json"
            dataset_metadata = {
                "dataset": dataset,
                "timestamp": metadata["experiment_info"]["timestamp"],
                "configurations": list(dataset_results.keys()),
                "results": metadata["results"][dataset],
            }
            with open(dataset_file, "w") as f:
                json.dump(dataset_metadata, f, indent=2)

        # Create summary file
        summary_file = self.output_dir / "experiment_summary.txt"
        self._create_summary_file(summary_file, datasets, configs, results)

        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"  - Main results: {results_file.relative_to(self.output_dir)}")
        logger.info(f"  - Individual datasets: raw_data/{dataset}_results.json")
        logger.info("  - Summary: experiment_summary.txt")

    def _create_summary_file(
        self,
        summary_file: Path,
        datasets: list[str],
        configs: list[dict[str, Any]],
        results: dict[str, Any],
    ):
        """Create a human-readable summary file"""
        with open(summary_file, "w") as f:
            f.write("SECURE FL BENCHMARK EXPERIMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Timestamp: {self.output_dir.name.split('_')[-1]}\n")
            f.write(f"Total Datasets: {len(datasets)}\n")
            f.write(f"Total Configurations: {len(configs)}\n\n")

            f.write("DATASETS:\n")
            for dataset in datasets:
                dataset_info = self.dataset_manager.get_dataset_info(dataset)
                f.write(
                    f"  • {dataset:<15} → {dataset_info['model_class']:<12} | {dataset_info['description']}\n"
                )
            f.write("\n")

            f.write("CONFIGURATIONS:\n")
            for config in configs:
                f.write(
                    f"  • {config['name']:<18} | {config.get('description', 'No description')}\n"
                )
            f.write("\n")

            f.write("RESULTS SUMMARY:\n")
            for dataset, dataset_results in results.items():
                f.write(f"\n{dataset.upper()} Dataset:\n")
                for config_name, result in dataset_results.items():
                    if isinstance(result, dict) and "final_accuracy" in result:
                        accuracy = result["final_accuracy"]
                        time = result.get("training_time", 0)
                        f.write(
                            f"  {config_name:<20}: {accuracy:6.2f}% ({time:5.1f}s)\n"
                        )
                    else:
                        f.write(f"  {config_name:<20}: ERROR or incomplete\n")

            f.write(
                "\nDetailed results available in: raw_data/benchmark_results.json\n"
            )
            f.write("Plots and visualizations in: plots/\n")

    def _generate_multi_dataset_plots(self, results: dict[str, Any]):
        """Generate comprehensive visualization plots"""
        logger.info("Generating multi-dataset visualizations...")

        try:
            # 1. Accuracy comparison across datasets and configurations
            self._plot_accuracy_comparison(results)

            # 2. ZKP overhead analysis
            self._plot_zkp_overhead(results)

            # 3. Convergence analysis
            self._plot_convergence_analysis(results)

            # 4. Model complexity vs performance
            self._plot_model_complexity(results)

            logger.info("All visualizations saved!")
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            logger.info("Continuing without visualizations...")

    def _plot_accuracy_comparison(self, results: dict[str, Any]):
        """Plot accuracy comparison across datasets and configs"""
        # Set beautiful aesthetic style
        plt.style.use("default")
        sns.set_theme()
        sns.set_palette("husl")

        fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 8))
        if len(results) == 1:
            axes = [axes]

        for idx, (dataset, configs) in enumerate(results.items()):
            ax = axes[idx]

            config_names = []
            final_accuracies = []
            colors = []

            # Enhanced color scheme
            color_map = {
                "baseline_iid": "#2E8B57",  # Sea green
                "baseline_non_iid": "#4682B4",  # Steel blue
                "zkp_high_rigor": "#DC143C",  # Crimson
                "zkp_medium_rigor": "#FF8C00",  # Dark orange
                "zkp_low_rigor": "#32CD32",  # Lime green
                "scaled_up": "#9932CC",  # Dark orchid
                "quick_test": "#20B2AA",  # Light sea green
                "performance_focused": "#696969",  # Dim gray
            }

            for config_name, result in configs.items():
                if "error" not in result:
                    config_names.append(config_name.replace("_", "\n"))
                    final_accuracies.append(result["final_accuracy"])
                    colors.append(color_map.get(config_name, "#666666"))

            bars = ax.bar(
                config_names,
                final_accuracies,
                color=colors,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.8,
            )

            # Add value labels with better formatting
            for bar, acc in zip(bars, final_accuracies, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(final_accuracies) * 0.02,
                    f"{acc:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=11,
                )

            ax.set_title(
                f"{dataset.upper()} Dataset\nAccuracy Comparison",
                fontweight="bold",
                fontsize=16,
                pad=20,
            )
            ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
            ax.set_xlabel("Configuration", fontsize=14, fontweight="bold")
            if final_accuracies:
                ax.set_ylim(0, max(final_accuracies) * 1.15)
            else:
                ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=11)

        plt.suptitle(
            "Secure FL Framework - Multi-Configuration Benchmark Results",
            fontsize=18,
            fontweight="bold",
            y=0.95,
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "plots" / "accuracy_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_zkp_overhead(self, results: dict[str, Any]):
        """Plot ZKP overhead analysis"""
        # Set beautiful aesthetic style
        plt.style.use("default")
        sns.set_theme()
        sns.set_palette("husl")

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        if isinstance(axes, plt.Axes):
            axes = [axes, axes]
        ax1, ax2 = axes

        # Collect ZKP data
        datasets = []
        proof_times = []
        rigor_levels = []

        for dataset, configs in results.items():
            for _config_name, result in configs.items():
                if "error" not in result and result.get("config", {}).get(
                    "enable_zkp", False
                ):
                    datasets.append(dataset)
                    proof_times.append(np.mean(result.get("proof_times", [0])))

                    rigor = result.get("config", {}).get("proof_rigor", "medium")
                    rigor_levels.append(rigor)

        # Plot 1: Proof times by rigor level
        rigor_colors = {"high": "#DC143C", "medium": "#FF8C00", "low": "#32CD32"}

        for rigor in ["low", "medium", "high"]:
            rigor_times = [
                pt
                for pt, rl in zip(proof_times, rigor_levels, strict=False)
                if rl == rigor
            ]
            rigor_datasets = [
                ds
                for ds, rl in zip(datasets, rigor_levels, strict=False)
                if rl == rigor
            ]

            if rigor_times:
                ax1.scatter(
                    rigor_datasets,
                    rigor_times,
                    label=f"{rigor.title()} Rigor",
                    color=rigor_colors[rigor],
                    s=100,
                    alpha=0.7,
                )

        ax1.set_title("ZKP Proof Generation Time by Rigor Level", fontweight="bold")
        ax1.set_ylabel("Proof Time (seconds)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Communication overhead
        overhead_data = []
        overhead_labels = []

        for dataset, configs in results.items():
            baseline_comm = None
            zkp_comm = None

            for _config_name, result in configs.items():
                if "error" not in result:
                    avg_comm = np.mean(result.get("communication_overhead", [0]))

                    if not result.get("config", {}).get("enable_zkp", False):
                        baseline_comm = avg_comm
                    else:
                        zkp_comm = avg_comm

            if baseline_comm and zkp_comm:
                overhead_pct = ((zkp_comm - baseline_comm) / baseline_comm) * 100
                overhead_data.append(overhead_pct)
                overhead_labels.append(dataset)

        if overhead_data:
            bars = ax2.bar(
                overhead_labels,
                overhead_data,
                color="#FF6347",
                alpha=0.7,
                edgecolor="black",
            )

            for bar, overhead in zip(bars, overhead_data, strict=False):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{overhead:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )
        else:
            ax2.text(
                0.5,
                0.5,
                "No ZKP overhead data available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        ax2.set_title("Communication Overhead from ZKP", fontweight="bold")
        ax2.set_ylabel("Overhead (%)")
        if overhead_data:
            ax2.set_ylim(0, max(overhead_data) * 1.1 + 1)
        ax2.grid(True, alpha=0.3)

        plt.suptitle(
            "ZKP Performance Analysis - Real Benchmark Data",
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "plots" / "zkp_overhead_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_convergence_analysis(self, results: dict[str, Any]):
        """Plot convergence analysis for each dataset"""
        fig, axes = plt.subplots(len(results), 2, figsize=(15, 5 * len(results)))
        if len(results) == 1:
            axes = axes.reshape(1, -1)

        config_colors = {
            "baseline_iid": "#2E8B57",
            "baseline_non_iid": "#4682B4",
            "zkp_high_rigor": "#DC143C",
            "zkp_medium_rigor": "#FF8C00",
            "scaled_up": "#9932CC",
        }

        for idx, (dataset, configs) in enumerate(results.items()):
            ax_acc, ax_loss = axes[idx] if len(results) > 1 else axes

            # Plot accuracy convergence
            for config_name, result in configs.items():
                if "error" not in result and result.get("round_accuracies"):
                    rounds = range(1, len(result["round_accuracies"]) + 1)
                    color = config_colors.get(config_name, "#666666")

                    ax_acc.plot(
                        rounds,
                        result["round_accuracies"],
                        marker="o",
                        label=config_name.replace("_", " ").title(),
                        color=color,
                        linewidth=2,
                    )

            ax_acc.set_title(
                f"{dataset.upper()} - Accuracy Convergence", fontweight="bold"
            )
            ax_acc.set_xlabel("Round")
            ax_acc.set_ylabel("Accuracy (%)")
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)

            # Plot loss convergence
            for config_name, result in configs.items():
                if "error" not in result and result.get("round_losses"):
                    rounds = range(1, len(result["round_losses"]) + 1)
                    color = config_colors.get(config_name, "#666666")

                    ax_loss.plot(
                        rounds,
                        result["round_losses"],
                        marker="s",
                        label=config_name.replace("_", " ").title(),
                        color=color,
                        linewidth=2,
                    )

            ax_loss.set_title(
                f"{dataset.upper()} - Loss Convergence", fontweight="bold"
            )
            ax_loss.set_xlabel("Round")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "plots" / "convergence_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_model_complexity(self, results: dict[str, Any]):
        """Plot model complexity vs performance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Collect model complexity data
        datasets = []
        accuracies = []
        model_names = []
        training_times = []

        for dataset, configs in results.items():
            # Use baseline_iid for fair comparison
            baseline_result = configs.get(
                "baseline_iid", configs.get("baseline_non_iid")
            )
            if baseline_result and "error" not in baseline_result:
                datasets.append(dataset)
                accuracies.append(baseline_result["final_accuracy"])
                model_names.append(baseline_result["model_class"])
                training_times.append(baseline_result["training_time"])

        # Check if we have any valid data
        if not datasets:
            logger.warning("No valid results found for model complexity analysis")
            # Create empty plots with message
            for ax in [ax1, ax2]:
                ax.text(
                    0.5,
                    0.5,
                    "No valid results available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            return

        # Plot 1: Model complexity (estimated by name)
        model_complexity = {"SimpleModel": 1, "MNISTModel": 2, "CIFAR10Model": 3}

        complexity_scores = [model_complexity.get(name, 2) for name in model_names]
        dataset_colors = {
            "mnist": "#FF6B6B",
            "cifar10": "#4ECDC4",
            "synthetic": "#45B7D1",
        }

        for i, (dataset, acc, complexity) in enumerate(
            zip(datasets, accuracies, complexity_scores, strict=False)
        ):
            ax1.scatter(
                complexity,
                acc,
                label=dataset.upper(),
                color=dataset_colors.get(dataset, "#666666"),
                s=150,
                alpha=0.7,
            )
            ax1.annotate(
                model_names[i],
                (complexity, acc),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        ax1.set_title("Model Complexity vs Accuracy", fontweight="bold")
        ax1.set_xlabel("Model Complexity (Relative)")
        ax1.set_ylabel("Final Accuracy (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Training time vs accuracy
        for i, (dataset, acc, training_time) in enumerate(
            zip(datasets, accuracies, training_times, strict=False)
        ):
            ax2.scatter(
                training_time,
                acc,
                label=dataset.upper(),
                color=dataset_colors.get(dataset, "#666666"),
                s=150,
                alpha=0.7,
            )
            ax2.annotate(
                model_names[i],
                (time, acc),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        ax2.set_title("Training Time vs Accuracy", fontweight="bold")
        ax2.set_xlabel("Training Time (seconds)")
        ax2.set_ylabel("Final Accuracy (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if hasattr(self, "output_dir") and self.output_dir:
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                plots_dir / "model_complexity_analysis.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close()

    def run_secure_fl_benchmark(
        self,
        dataset_name: str,
        model_name: str,
        num_clients: int = 3,
        num_rounds: int = 5,
        enable_zkp: bool = False,  # Start with False for initial testing
        proof_rigor: str = "medium",
        server_host: str = "localhost",
        server_port: int = 8080,
    ):
        """Run benchmark using actual secure-fl server and clients"""

        print(f"🚀 Running Secure-FL benchmark for {dataset_name} with {model_name}")
        print(f"   Clients: {num_clients}, Rounds: {num_rounds}, ZKP: {enable_zkp}")

        start_time = time.time()

        # Get dataset and model
        dataset_manager = DatasetManager()
        federated_data = dataset_manager.load_federated_data(
            dataset_name, num_clients=num_clients
        )

        # Extract train and test datasets from federated data
        [client_data[0].dataset for client_data in federated_data]
        (
            [client_data[1].dataset for client_data in federated_data]
            if federated_data[0][1]
            else None
        )

        model_fn = dataset_manager.get_model_fn(model_name)

        print(f"✅ Loaded federated data for {num_clients} clients")

        # Create server strategy
        create_server_strategy(
            model_fn=model_fn,
            momentum=0.9,
            learning_rate=0.01,
            enable_zkp=enable_zkp,
            proof_rigor=proof_rigor,
            min_available_clients=num_clients,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
        )

        print(f"✅ Created server strategy with ZKP={enable_zkp}")

        # Create clients using the DataLoaders from federated data
        clients = []
        for i in range(num_clients):
            train_loader, test_loader = federated_data[i]
            client = create_client(
                client_id=f"client_{i}",
                model_fn=model_fn,
                train_data=train_loader.dataset,
                val_data=test_loader.dataset if test_loader else None,
                batch_size=32,
                enable_zkp=enable_zkp,
                proof_rigor=proof_rigor,
                local_epochs=2,
                learning_rate=0.01,
                quantize_weights=False,  # Disable quantization to fix dtype errors
            )
            clients.append(client)

        print(f"✅ Created {len(clients)} secure FL clients")

        # Simulate federated learning process
        # Note: This is a simplified simulation for benchmarking
        # In production, server and clients would run in separate processes

        results = {
            "dataset": dataset_name,
            "model": model_name,
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "enable_zkp": enable_zkp,
            "proof_rigor": proof_rigor,
            "rounds": [],
            "clients_info": [],
            "total_time": 0,
        }

        # Simulate FL training rounds
        print(f"🔄 Starting {num_rounds} federated learning rounds...")

        # Get initial parameters from model
        initial_model = model_fn()
        global_params = [
            param.detach().cpu().numpy() for param in initial_model.parameters()
        ]

        round_metrics = []

        for round_num in range(1, num_rounds + 1):
            print(f"  📍 Round {round_num}/{num_rounds}")
            round_start = time.time()

            # Client training phase
            client_results = []
            round_training_times = []
            round_proof_times = []

            for i, client in enumerate(clients):
                client_start = time.time()

                # Simulate client fit
                config = {
                    "server_round": round_num,
                    "local_epochs": 2,
                    "learning_rate": 0.01,
                    "proof_rigor": proof_rigor,
                    "enable_zkp": enable_zkp,
                }

                try:
                    updated_params, num_examples, client_metrics = client.fit(
                        global_params, config
                    )

                    client_time = time.time() - client_start
                    training_time = client_metrics.get("training_time", client_time)
                    proof_time = client_metrics.get("proof_time", 0)

                    round_training_times.append(training_time)
                    round_proof_times.append(proof_time)

                    client_results.append(
                        {
                            "client_id": f"client_{i}",
                            "num_examples": num_examples,
                            "training_time": training_time,
                            "proof_time": proof_time,
                            "train_loss": client_metrics.get("train_loss", 0),
                            "train_accuracy": client_metrics.get("train_accuracy", 0),
                        }
                    )

                    # Update global parameters (simple averaging for simulation)
                    if i == 0:
                        global_params = updated_params
                    else:
                        # Simple parameter averaging
                        for j in range(len(global_params)):
                            global_params[j] = (
                                global_params[j] * i + updated_params[j]
                            ) / (i + 1)
                            # Ensure correct dtype
                            global_params[j] = global_params[j].astype(np.float32)

                except Exception as e:
                    print(f"    ❌ Client {i} failed: {e}")
                    continue

            # Server aggregation phase (simulated)
            round_time = time.time() - round_start

            # Evaluate global model
            if federated_data[0][1] is not None:
                # Use the first client's test loader
                test_loader = federated_data[0][1]
                test_model = model_fn()

                # Set global parameters
                for param, array in zip(
                    test_model.parameters(), global_params, strict=False
                ):
                    param.data = torch.tensor(array, dtype=param.dtype)

                test_model.eval()
                correct = 0
                total = 0

                with torch.no_grad():
                    for data, target in test_loader:
                        output = test_model(data)
                        pred = output.argmax(dim=1)
                        correct += pred.eq(target).sum().item()
                        total += data.size(0)

                global_accuracy = correct / total if total > 0 else 0.0
            else:
                global_accuracy = 0.0

            round_metrics.append(
                {
                    "round": round_num,
                    "round_time": round_time,
                    "avg_training_time": np.mean(round_training_times)
                    if round_training_times
                    else 0,
                    "avg_proof_time": np.mean(round_proof_times)
                    if round_proof_times
                    else 0,
                    "total_proof_time": sum(round_proof_times),
                    "global_accuracy": global_accuracy,
                    "num_successful_clients": len(client_results),
                    "client_results": client_results,
                }
            )

            print(
                f"    ✅ Round {round_num} completed - Accuracy: {global_accuracy:.3f}, Time: {round_time:.2f}s"
            )

        # Collect final results
        total_time = time.time() - start_time
        results["total_time"] = total_time
        results["rounds"] = round_metrics

        # Get client info
        for client in clients:
            results["clients_info"].append(client.get_client_info())

        # Calculate summary metrics
        final_accuracy = round_metrics[-1]["global_accuracy"] if round_metrics else 0.0
        total_training_time = sum(
            r["avg_training_time"] * r["num_successful_clients"] for r in round_metrics
        )
        total_proof_time = sum(r["total_proof_time"] for r in round_metrics)

        results["summary"] = {
            "final_accuracy": final_accuracy,
            "total_training_time": total_training_time,
            "total_proof_time": total_proof_time,
            "zkp_overhead_ratio": total_proof_time / total_training_time
            if total_training_time > 0
            else 0,
            "avg_round_time": np.mean([r["round_time"] for r in round_metrics])
            if round_metrics
            else 0,
        }

        print("🎉 Secure-FL benchmark completed!")
        print(f"   Final accuracy: {final_accuracy:.3f}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   ZKP overhead: {results['summary']['zkp_overhead_ratio']:.2%}")

        return results

    def run_secure_fl_comparison(self, datasets=None, zkp_configs=None):
        """Run comparison between different ZKP configurations"""

        if datasets is None:
            datasets = ["mnist", "fashion_mnist"]

        if zkp_configs is None:
            zkp_configs = [
                {"enable_zkp": False, "name": "No ZKP"},
                {"enable_zkp": True, "proof_rigor": "low", "name": "ZKP Low"},
                {"enable_zkp": True, "proof_rigor": "medium", "name": "ZKP Medium"},
                {"enable_zkp": True, "proof_rigor": "high", "name": "ZKP High"},
            ]

        print(
            f"🔬 Running Secure-FL comparison across {len(datasets)} datasets and {len(zkp_configs)} configurations"
        )

        all_results = []

        DatasetManager()

        for dataset in datasets:
            model_name = dataset  # Assuming dataset name matches model name

            print(f"\n📊 Testing dataset: {dataset}")

            for config in zkp_configs:
                print(f"  🧪 Configuration: {config['name']}")

                try:
                    result = self.run_secure_fl_benchmark(
                        dataset_name=dataset,
                        model_name=model_name,
                        num_clients=3,
                        num_rounds=3,  # Shorter for comparison
                        enable_zkp=config["enable_zkp"],
                        proof_rigor=config.get("proof_rigor", "medium"),
                    )

                    result["config_name"] = config["name"]
                    all_results.append(result)

                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    continue

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"secure_fl_comparison_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {results_file}")

        # Generate comparison plots
        self._plot_secure_fl_comparison(all_results)

        return all_results

    def _plot_secure_fl_comparison(self, results):
        """Generate comparison plots for secure-FL results"""

        # Set beautiful aesthetic style
        plt.style.use("default")
        sns.set_theme()
        sns.set_palette("husl")

        # Extract data for plotting
        datasets = list(set(r["dataset"] for r in results))
        configs = list(set(r["config_name"] for r in results))

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        accuracy_data = {}
        for config in configs:
            accuracy_data[config] = []
            for dataset in datasets:
                dataset_results = [
                    r
                    for r in results
                    if r["dataset"] == dataset and r["config_name"] == config
                ]
                if dataset_results:
                    accuracy_data[config].append(
                        dataset_results[0]["summary"]["final_accuracy"]
                    )
                else:
                    accuracy_data[config].append(0)

        x = np.arange(len(datasets))
        width = 0.2
        for i, (config, accuracies) in enumerate(accuracy_data.items()):
            ax1.bar(x + i * width, accuracies, width, label=config)

        ax1.set_xlabel("Dataset")
        ax1.set_ylabel("Final Accuracy")
        ax1.set_title("Accuracy Comparison Across ZKP Configurations")
        ax1.set_xticks(x + width * (len(configs) - 1) / 2)
        ax1.set_xticklabels(datasets)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. ZKP Overhead
        ax2 = axes[0, 1]
        overhead_data = {}
        for config in configs:
            overhead_data[config] = []
            for dataset in datasets:
                dataset_results = [
                    r
                    for r in results
                    if r["dataset"] == dataset and r["config_name"] == config
                ]
                if dataset_results:
                    overhead_data[config].append(
                        dataset_results[0]["summary"]["zkp_overhead_ratio"] * 100
                    )
                else:
                    overhead_data[config].append(0)

        for i, (config, overheads) in enumerate(overhead_data.items()):
            if "ZKP" in config:  # Only show ZKP configs for overhead
                ax2.bar(x + i * width, overheads, width, label=config)

        ax2.set_xlabel("Dataset")
        ax2.set_ylabel("ZKP Overhead (%)")
        ax2.set_title("ZKP Overhead Comparison")
        ax2.set_xticks(x + width * (len(configs) - 1) / 2)
        ax2.set_xticklabels(datasets)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Training Time
        ax3 = axes[1, 0]
        time_data = {}
        for config in configs:
            time_data[config] = []
            for dataset in datasets:
                dataset_results = [
                    r
                    for r in results
                    if r["dataset"] == dataset and r["config_name"] == config
                ]
                if dataset_results:
                    time_data[config].append(dataset_results[0]["total_time"])
                else:
                    time_data[config].append(0)

        for i, (config, times) in enumerate(time_data.items()):
            ax3.bar(x + i * width, times, width, label=config)

        ax3.set_xlabel("Dataset")
        ax3.set_ylabel("Total Time (seconds)")
        ax3.set_title("Total Training Time Comparison")
        ax3.set_xticks(x + width * (len(configs) - 1) / 2)
        ax3.set_xticklabels(datasets)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Round-by-round accuracy for one dataset
        ax4 = axes[1, 1]
        sample_dataset = None
        if results and datasets:
            sample_dataset = datasets[0]
            for config in configs:
                config_results = [
                    r
                    for r in results
                    if r["dataset"] == sample_dataset and r["config_name"] == config
                ]
                if config_results:
                    rounds = config_results[0]["rounds"]
                    accuracies = [r["global_accuracy"] for r in rounds]
                    round_nums = [r["round"] for r in rounds]
                    ax4.plot(round_nums, accuracies, marker="o", label=config)

        ax4.set_xlabel("Round")
        ax4.set_ylabel("Accuracy")
        ax4.set_title(f"Convergence Comparison ({sample_dataset})")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if hasattr(self, "output_dir") and self.output_dir:
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                plots_dir / "secure_fl_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close()


def list_datasets():
    """List all available datasets"""
    dm = DatasetManager()

    print("\n🔍 AVAILABLE DATASETS FOR SECURE FL BENCHMARK")
    print("=" * 60)

    categories = dm.get_dataset_categories()
    for category, datasets in categories.items():
        print(f"\n📂 {category}:")
        for dataset in datasets:
            if dataset in dm.dataset_configs:
                config = dm.dataset_configs[dataset]
                model_name = config["model_class"].__name__
                description = config["description"]
                input_shape = config["input_shape"]
                num_classes = config["num_classes"]

                print(f"  • {dataset:15s} → {model_name:12s} | {description}")
                print(
                    f"    {'':<17s}   Shape: {str(input_shape):<12s} | Classes: {num_classes}"
                )

    print(f"\nTotal: {len(dm.dataset_configs)} datasets available")
    print(
        f"Usage: python benchmark.py --datasets {' '.join(list(dm.dataset_configs.keys())[:3])}"
    )


def main():
    """Main benchmark execution function"""
    parser = argparse.ArgumentParser(
        description="Multi-Dataset Benchmark for Secure FL"
    )

    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "mnist",
            "fashion_mnist",
            "cifar10",
            "cifar100",
            "synthetic",
            "synthetic_large",
            "synthetic_small",
            "text_classification",
            "medical",
            "financial",
        ],
        default=["mnist", "cifar10", "synthetic"],
        help="Datasets to benchmark",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/multi_dataset_benchmark",
        help="Output directory for results",
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark (fewer rounds)"
    )

    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        choices=[
            "baseline_iid",
            "baseline_non_iid",
            "zkp_medium_rigor",
            "zkp_low_rigor",
            "quick_test",
        ],
        help="Specific configs to run",
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List all available configurations and exit",
    )

    parser.add_argument(
        "--secure-fl",
        action="store_true",
        help="Run secure-fl benchmark with actual server and clients",
    )

    parser.add_argument(
        "--secure-fl-comparison",
        action="store_true",
        help="Run secure-fl comparison across ZKP configurations",
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=3,
        help="Number of federated learning clients",
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="Number of federated learning rounds",
    )

    parser.add_argument(
        "--enable-zkp",
        action="store_true",
        help="Enable zero-knowledge proofs",
    )

    parser.add_argument(
        "--proof-rigor",
        choices=["low", "medium", "high"],
        default="medium",
        help="ZKP proof rigor level",
    )

    args = parser.parse_args()

    # Handle listing requests
    if args.list_datasets:
        list_datasets()
        return

    if args.list_configs:
        runner = MultiDatasetBenchmarkRunner()
        configs = runner.get_default_benchmark_configs()

        print("\n⚙️  AVAILABLE BENCHMARK CONFIGURATIONS")
        print("=" * 60)

        for config in configs:
            name = config["name"]
            desc = config.get("description", "No description")
            clients = config["num_clients"]
            rounds = config["num_rounds"]
            zkp = "✓" if config["enable_zkp"] else "✗"
            rigor = config.get("proof_rigor", "N/A")
            iid = "IID" if config["iid"] else "Non-IID"

            print(f"  • {name:18s} | {desc}")
            print(
                f"    {'':20s} Clients: {clients}, Rounds: {rounds}, ZKP: {zkp}, Rigor: {rigor}, Data: {iid}"
            )
            print()

        print(f"Total: {len(configs)} configurations available")
        print(
            f"Usage: python benchmark.py --configs {' '.join([c['name'] for c in configs[:3]])}"
        )
        return

    # Handle secure-fl specific runs
    if args.secure_fl or args.secure_fl_comparison:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        runner = MultiDatasetBenchmarkRunner(args.output_dir)

        if args.secure_fl:
            # Run single secure-fl benchmark
            dataset = args.datasets[0] if args.datasets else "mnist"
            model_name = dataset  # Assuming dataset name matches model name

            print(f"🚀 Running Secure-FL benchmark for {dataset}")

            result = runner.run_secure_fl_benchmark(
                dataset_name=dataset,
                model_name=model_name,
                num_clients=args.num_clients,
                num_rounds=args.num_rounds,
                enable_zkp=args.enable_zkp,
                proof_rigor=args.proof_rigor,
            )

            print("\n🎉 Secure-FL benchmark completed successfully!")
            return

        elif args.secure_fl_comparison:
            # Run secure-fl comparison
            datasets = (
                args.datasets[:2]
                if len(args.datasets) >= 2
                else ["mnist", "fashion_mnist"]
            )

            print(f"🔬 Running Secure-FL comparison for datasets: {datasets}")

            results = runner.run_secure_fl_comparison(datasets=datasets)

            print(
                f"\n🎉 Secure-FL comparison completed! Tested {len(results)} configurations."
            )
            return

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize benchmark runner
    runner = MultiDatasetBenchmarkRunner(args.output_dir)

    # Get configurations
    if args.configs:
        all_configs = runner.get_default_benchmark_configs()
        configs = [c for c in all_configs if c["name"] in args.configs]
    else:
        configs = runner.get_default_benchmark_configs()

    # Modify for quick run
    if args.quick:
        for config in configs:
            config["num_rounds"] = min(3, config["num_rounds"])
            config["local_epochs"] = min(2, config["local_epochs"])

    # Run benchmark
    logger.info("Starting Multi-Dataset Secure FL Benchmark")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Output: {args.output_dir}")

    try:
        results = runner.run_comprehensive_benchmark(args.datasets, configs)

        # Print summary
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        for dataset, configs in results.items():
            print(f"\n{dataset.upper()} Dataset:")
            for config_name, result in configs.items():
                if "error" not in result:
                    final_acc = result.get("final_accuracy", 0)
                    training_time = result.get("training_time", 0)
                    print(
                        f"  {config_name:20s}: {final_acc:6.2f}% ({training_time:5.1f}s)"
                    )
                else:
                    print(f"  {config_name:20s}: ERROR - {result['error']}")

        print(f"\nDetailed results and plots saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
