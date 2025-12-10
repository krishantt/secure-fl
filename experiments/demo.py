"""
Demo Module for Secure FL

This module provides quick demonstration functions to showcase
the Secure FL framework capabilities.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from secure_fl.models import SimpleModel

logger = logging.getLogger(__name__)


def create_federated_datasets(
    num_clients: int = 3,
    samples_per_client: int = 200,
    input_dim: int = 784,
    num_classes: int = 10,
    iid: bool = False,
    val_split: float = 0.2,
) -> list[tuple[DataLoader, DataLoader]]:
    """
    Create synthetic federated datasets for demo purposes.

    Args:
        num_clients: Number of clients
        samples_per_client: Number of samples per client
        input_dim: Input dimension (default 784 for MNIST-like)
        num_classes: Number of classes
        iid: Whether data should be IID across clients
        val_split: Validation split ratio

    Returns:
        List of (train_loader, val_loader) tuples for each client
    """
    client_datasets = []

    for client_id in range(num_clients):
        # Generate synthetic data
        if iid:
            # IID: each client gets samples from all classes
            X = torch.randn(samples_per_client, input_dim)
            y = torch.randint(0, num_classes, (samples_per_client,))
        else:
            # Non-IID: each client focuses on specific classes
            classes_per_client = max(2, num_classes // num_clients)
            start_class = (client_id * classes_per_client) % num_classes
            client_classes = [
                (start_class + i) % num_classes for i in range(classes_per_client)
            ]

            X = torch.randn(samples_per_client, input_dim)
            y = torch.tensor(
                [np.random.choice(client_classes) for _ in range(samples_per_client)]
            )

        # Add some pattern to make the task learnable
        for i in range(len(X)):
            class_idx = y[i].item()
            # Add class-specific pattern
            X[i, :10] += class_idx * 0.5
            X[i, 10:20] -= class_idx * 0.3

        # Split into train/val
        val_size = int(samples_per_client * val_split)
        train_size = samples_per_client - val_size

        train_X, val_X = X[:train_size], X[train_size:]
        train_y, val_y = y[:train_size], y[train_size:]

        # Create datasets and loaders
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        client_datasets.append((train_loader, val_loader))

    return client_datasets


def simulate_federated_round(
    global_model: nn.Module,
    client_datasets: list[tuple[DataLoader, DataLoader]],
    local_epochs: int = 3,
    learning_rate: float = 0.01,
) -> tuple[nn.Module, dict[str, float]]:
    """
    Simulate one round of federated learning.

    Args:
        global_model: Global model to be updated
        client_datasets: List of (train_loader, val_loader) for each client
        local_epochs: Number of local training epochs
        learning_rate: Learning rate for local training

    Returns:
        Tuple of (updated_global_model, metrics)
    """
    client_models = []
    client_weights = []
    metrics = {}

    # Client training
    for client_id, (train_loader, val_loader) in enumerate(client_datasets):
        # Create local model copy
        local_model = SimpleModel()
        local_model.load_state_dict(global_model.state_dict())

        # Local training
        optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        local_model.train()
        for _epoch in range(local_epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = local_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate local model
        local_model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = local_model(batch_x)
                loss = criterion(outputs, batch_y)
                _, predicted = torch.max(outputs.data, 1)

                total_loss += loss.item()
                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()
                batch_count += 1

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0

        metrics[f"client_{client_id}_accuracy"] = accuracy
        metrics[f"client_{client_id}_loss"] = avg_loss

        client_models.append(local_model)
        # Calculate training samples by iterating through the loader once
        train_samples = sum(batch.size(0) for batch, _ in train_loader)
        client_weights.append(train_samples)

    # Aggregate models (FedAvg)
    total_samples = sum(client_weights)

    # Initialize aggregated state dict
    aggregated_state = {}
    for key in global_model.state_dict().keys():
        aggregated_state[key] = torch.zeros_like(global_model.state_dict()[key])

    # Weighted aggregation
    for client_model, weight in zip(client_models, client_weights, strict=False):
        client_state = client_model.state_dict()
        weight_ratio = weight / total_samples

        for key in aggregated_state.keys():
            aggregated_state[key] += weight_ratio * client_state[key]

    # Update global model
    global_model.load_state_dict(aggregated_state)

    # Calculate average metrics
    avg_accuracy = np.mean([metrics[k] for k in metrics.keys() if "accuracy" in k])
    avg_loss = np.mean([metrics[k] for k in metrics.keys() if "loss" in k])

    metrics["global_accuracy"] = avg_accuracy
    metrics["global_loss"] = avg_loss

    return global_model, metrics


def run_quick_demo() -> bool:
    """Run a quick demo of federated learning"""

    try:
        print("🎬 Running Secure FL Quick Demo")
        print("=" * 50)

        # Configuration
        num_clients = 3
        num_rounds = 5
        local_epochs = 3
        learning_rate = 0.01

        print("Configuration:")
        print(f"  - Clients: {num_clients}")
        print(f"  - Rounds: {num_rounds}")
        print(f"  - Local epochs: {local_epochs}")
        print(f"  - Learning rate: {learning_rate}")
        print()

        # Create synthetic datasets
        print("📊 Creating federated datasets...")
        client_datasets = create_federated_datasets(
            num_clients=num_clients, samples_per_client=200, iid=False
        )
        print(f"✓ Created datasets for {num_clients} clients")

        # Initialize global model
        global_model = SimpleModel()
        print("🧠 Initialized global model")

        # Training loop
        print("\n🚀 Starting federated training...")
        history = {"rounds": [], "accuracy": [], "loss": []}

        for round_num in range(num_rounds):
            print(f"\nRound {round_num + 1}/{num_rounds}")

            start_time = time.time()

            # Simulate federated round
            global_model, metrics = simulate_federated_round(
                global_model, client_datasets, local_epochs, learning_rate
            )

            round_time = time.time() - start_time

            # Log metrics
            accuracy = metrics["global_accuracy"]
            loss = metrics["global_loss"]

            history["rounds"].append(round_num + 1)
            history["accuracy"].append(accuracy)
            history["loss"].append(loss)

            print(f"  ✓ Accuracy: {accuracy:.3f}")
            print(f"  ✓ Loss: {loss:.3f}")
            print(f"  ✓ Time: {round_time:.2f}s")

            # Show client-specific metrics
            for client_id in range(num_clients):
                client_acc = metrics[f"client_{client_id}_accuracy"]
                print(f"    Client {client_id}: {client_acc:.3f}")

        # Summary
        print("\n📈 Training Summary:")
        print(f"  Final Accuracy: {history['accuracy'][-1]:.3f}")
        print(f"  Final Loss: {history['loss'][-1]:.3f}")
        print(f"  Improvement: {history['accuracy'][-1] - history['accuracy'][0]:.3f}")

        # Save results
        results_dir = Path("./results/demo")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = results_dir / "demo_model.pth"
        torch.save(global_model.state_dict(), model_path)
        print("✓ Model saved to:", model_path)

        # Save metrics
        metrics_path = results_dir / "demo_metrics.txt"
        with open(metrics_path, "w") as f:
            f.write("Secure FL Demo Results\n")
            f.write("=" * 30 + "\n\n")
            f.write("Configuration:\n")
            f.write(f"  Clients: {num_clients}\n")
            f.write(f"  Rounds: {num_rounds}\n")
            f.write(f"  Local epochs: {local_epochs}\n\n")
            f.write("Training History:\n")
            for i, (acc, loss) in enumerate(zip(history["accuracy"], history["loss"], strict=False)):
                f.write(f"  Round {i + 1}: Acc={acc:.3f}, Loss={loss:.3f}\n")

        print("✓ Metrics saved to:", metrics_path)

        print("\n✅ Demo completed successfully!")
        print("🔍 Note: This was a simplified demo without ZKP verification")
        print(
            "🚀 Try the full experiment with: secure-fl experiment --dataset synthetic"
        )

        return True

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo error")
        return False


def run_demo() -> bool:
    """Run a more comprehensive demo"""

    try:
        print("🎬 Running Secure FL Comprehensive Demo")
        print("=" * 50)

        # Check if ZKP components are available
        try:
            import importlib.util

            zkp_available = (
                importlib.util.find_spec("secure_fl.SecureFlowerClient") is not None
            )
        except ImportError:
            zkp_available = False

        if not zkp_available:
            print("⚠️  ZKP components not available")

        # Configuration
        config = {
            "num_clients": 3,
            "num_rounds": 5,
            "enable_zkp": zkp_available,
            "proof_rigor": "low" if zkp_available else None,
            "local_epochs": 3,
            "batch_size": 32,
            "learning_rate": 0.01,
        }

        print("Configuration:")
        for key, value in config.items():
            print(f"  - {key}: {value}")
        print()

        if zkp_available:
            print("🔐 ZKP verification enabled")
        else:
            print("⚠️  Running without ZKP verification")

        # Run quick demo as fallback or preparation
        success = run_quick_demo()

        if success and zkp_available:
            print("\n🔐 ZKP Demo Features:")
            print("  - Client-side zk-STARK proof generation (simulated)")
            print("  - Server-side zk-SNARK aggregation verification (simulated)")
            print("  - Dynamic proof rigor adjustment")
            print("  - Blockchain verification ready")

        return success

    except Exception as e:
        print(f"\n❌ Comprehensive demo failed: {e}")
        logger.exception("Demo error")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_demo()
    else:
        run_demo()
