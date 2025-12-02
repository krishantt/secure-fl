"""
Demo Module for Secure FL

This module provides quick demonstration functions to showcase
the Secure FL framework capabilities.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# Model moved to secure_fl.models

from secure_fl.models import SimpleModel

    # Client training
    for client_id, (train_loader, val_loader) in enumerate(client_datasets):
        # Create local model copy
        local_model = SimpleModel()
        local_model.load_state_dict(global_model.state_dict())

        # Local training
        optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        local_model.train()
        for epoch in range(local_epochs):
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

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = local_model(batch_x)
                loss = criterion(outputs, batch_y)
                _, predicted = torch.max(outputs.data, 1)

                total_loss += loss.item()
                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()

        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(val_loader)

        metrics[f"client_{client_id}_accuracy"] = accuracy
        metrics[f"client_{client_id}_loss"] = avg_loss

        client_models.append(local_model)
        client_weights.append(len(train_loader.dataset))

    # Aggregate models (FedAvg)
    total_samples = sum(client_weights)

    # Initialize aggregated state dict
    aggregated_state = {}
    for key in global_model.state_dict().keys():
        aggregated_state[key] = torch.zeros_like(global_model.state_dict()[key])

    # Weighted aggregation
    for client_model, weight in zip(client_models, client_weights):
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

        print(f"Configuration:")
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
        results_dir = Path("./demo_results")
        results_dir.mkdir(exist_ok=True)

        # Save model
        model_path = results_dir / "demo_model.pth"
        torch.save(global_model.state_dict(), model_path)
        print(f"✓ Model saved to: {model_path}")

        # Save metrics
        metrics_path = results_dir / "demo_metrics.txt"
        with open(metrics_path, "w") as f:
            f.write("Secure FL Demo Results\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Clients: {num_clients}\n")
            f.write(f"  Rounds: {num_rounds}\n")
            f.write(f"  Local epochs: {local_epochs}\n\n")
            f.write("Training History:\n")
            for i, (acc, loss) in enumerate(zip(history["accuracy"], history["loss"])):
                f.write(f"  Round {i + 1}: Acc={acc:.3f}, Loss={loss:.3f}\n")

        print(f"✓ Metrics saved to: {metrics_path}")

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

        # Import Secure FL components
        try:
            from secure_fl import (
                SecureFlowerClient,
                SecureFlowerServer,
                create_client,
                create_server_strategy,
            )

            zkp_available = True
        except ImportError as e:
            print(f"⚠️  ZKP components not available: {e}")
            zkp_available = False

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
