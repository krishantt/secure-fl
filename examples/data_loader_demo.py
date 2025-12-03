#!/usr/bin/env python3
"""
FederatedDataLoader Demo Script

This script demonstrates how to use the FederatedDataLoader class for
loading and partitioning datasets in federated learning experiments.

Usage:
    python data_loader_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from secure_fl.data import FederatedDataLoader
from secure_fl.data.utils import print_dataset_summary


def demo_basic_usage():
    """Demonstrate basic FederatedDataLoader usage."""
    print("=" * 60)
    print("BASIC FEDERATED DATA LOADER USAGE")
    print("=" * 60)

    # Create a federated data loader for MNIST
    loader = FederatedDataLoader(
        dataset_name="mnist",
        num_clients=3,
        iid=True,
        val_split=0.2,
        batch_size=32,
    )

    print(f"Created loader: {loader}")

    # Load the full dataset
    train_dataset, test_dataset = loader.load_dataset()
    print(f"Full dataset - Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Create client datasets
    client_datasets = loader.create_client_datasets()
    print(f"Created {len(client_datasets)} client datasets")

    for i, (train_data, val_data) in enumerate(client_datasets):
        train_size = len(train_data) if train_data else 0
        val_size = len(val_data) if val_data else 0
        print(f"Client {i}: Train={train_size}, Val={val_size}")

    # Get dataset statistics
    stats = loader.get_client_stats()
    print(f"Client stats: {stats}")


def demo_different_datasets():
    """Demonstrate different dataset types."""
    print("\n" + "=" * 60)
    print("DIFFERENT DATASET TYPES")
    print("=" * 60)

    datasets = ["mnist", "cifar10", "synthetic"]

    for dataset_name in datasets:
        print(f"\n--- {dataset_name.upper()} Dataset ---")

        # Print dataset summary
        print_dataset_summary(dataset_name, num_clients=5)

        # Create loader
        try:
            loader = FederatedDataLoader(
                dataset_name=dataset_name,
                num_clients=2,
                batch_size=64,
            )

            # Load and show basic info
            train_dataset, test_dataset = loader.load_dataset()
            print(f"Loaded successfully - Train: {len(train_dataset)}")

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")


def demo_iid_vs_non_iid():
    """Demonstrate IID vs Non-IID data partitioning."""
    print("\n" + "=" * 60)
    print("IID VS NON-IID PARTITIONING")
    print("=" * 60)

    for iid in [True, False]:
        partition_type = "IID" if iid else "Non-IID"
        print(f"\n--- {partition_type} Partitioning ---")

        loader = FederatedDataLoader(
            dataset_name="mnist",
            num_clients=3,
            iid=iid,
            val_split=0.2,
            batch_size=32,
        )

        stats = loader.get_client_stats()
        print(f"Partitioning: {stats['partitioning']}")
        print("Client distribution:")
        for client_stat in stats["client_sizes"]:
            print(
                f"  Client {client_stat['client_id']}: "
                f"{client_stat['total_samples']} samples"
            )


def demo_data_loaders():
    """Demonstrate creating PyTorch DataLoaders."""
    print("\n" + "=" * 60)
    print("PYTORCH DATALOADERS")
    print("=" * 60)

    loader = FederatedDataLoader(
        dataset_name="synthetic",  # Use synthetic for fast demo
        num_clients=2,
        batch_size=16,
    )

    # Get data loaders for all clients
    all_client_loaders = loader.create_client_dataloaders()
    print(f"Created data loaders for {len(all_client_loaders)} clients")

    for i, (train_loader, val_loader) in enumerate(all_client_loaders):
        print(f"\nClient {i}:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader) if val_loader else 0}")

        # Show first batch
        if len(train_loader) > 0:
            data, target = next(iter(train_loader))
            print(f"  Batch shape: {data.shape}, Labels: {target.shape}")

    # Get data loader for specific client
    print("\n--- Specific Client DataLoader ---")
    train_loader, val_loader = loader.create_client_dataloaders(client_id=0)
    print(f"Client 0 - Train batches: {len(train_loader)}")


def demo_configuration_options():
    """Demonstrate various configuration options."""
    print("\n" + "=" * 60)
    print("CONFIGURATION OPTIONS")
    print("=" * 60)

    configs = [
        {
            "name": "Small Scale",
            "params": {
                "dataset_name": "synthetic",
                "num_clients": 2,
                "batch_size": 8,
                "val_split": 0.1,
            },
        },
        {
            "name": "Medium Scale",
            "params": {
                "dataset_name": "mnist",
                "num_clients": 5,
                "batch_size": 64,
                "val_split": 0.2,
                "iid": False,
            },
        },
        {
            "name": "Large Batch",
            "params": {
                "dataset_name": "synthetic",
                "num_clients": 3,
                "batch_size": 128,
                "val_split": 0.25,
            },
        },
    ]

    for config in configs:
        print(f"\n--- {config['name']} Configuration ---")
        try:
            loader = FederatedDataLoader(**config["params"])
            stats = loader.get_client_stats()
            info = loader.get_dataset_info()

            print(f"Dataset: {info.get('name', info.get('description', 'Unknown'))}")
            print(f"Clients: {stats['num_clients']}")
            print(f"Partitioning: {stats['partitioning']}")
            print(
                f"Total samples per client: {[c['total_samples'] for c in stats['client_sizes']]}"
            )

        except Exception as e:
            print(f"Error with {config['name']}: {e}")


def demo_advanced_features():
    """Demonstrate advanced features."""
    print("\n" + "=" * 60)
    print("ADVANCED FEATURES")
    print("=" * 60)

    # Custom batch sizes per client
    loader = FederatedDataLoader(
        dataset_name="synthetic",
        num_clients=2,
        batch_size=32,  # Default batch size
    )

    print("--- Custom Batch Sizes ---")
    # Override batch size for specific client
    train_loader, val_loader = loader.create_client_dataloaders(
        client_id=0,
        batch_size=16,  # Custom batch size
    )
    print(f"Client 0 with batch_size=16: {len(train_loader)} batches")

    # Different shuffle settings
    print("\n--- Shuffle Settings ---")
    train_loader_shuffled, _ = loader.create_client_dataloaders(
        client_id=0, shuffle=True
    )
    train_loader_ordered, _ = loader.create_client_dataloaders(
        client_id=0, shuffle=False
    )
    print(f"Shuffled: {len(train_loader_shuffled)} batches")
    print(f"Ordered: {len(train_loader_ordered)} batches")


def main():
    """Run all demonstrations."""
    print("FederatedDataLoader Demonstration")
    print("This script shows various features of the FederatedDataLoader class")

    try:
        demo_basic_usage()
        demo_different_datasets()
        demo_iid_vs_non_iid()
        demo_data_loaders()
        demo_configuration_options()
        demo_advanced_features()

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("1. FederatedDataLoader provides unified interface for multiple datasets")
        print("2. Supports both IID and Non-IID data partitioning")
        print("3. Automatic data downloading and preprocessing")
        print("4. Flexible configuration options for different experiment needs")
        print("5. Easy integration with PyTorch DataLoader")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
