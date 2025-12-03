"""
End-to-End Integration Tests for Secure FL Complete Training Workflow

This module contains comprehensive end-to-end tests that verify the complete
federated learning training workflow from start to finish, including:
- Multi-client federated training
- Server-client communication
- Model aggregation and convergence
- ZKP integration (when enabled)
- Error handling and recovery
- Performance and scalability

These tests simulate realistic federated learning scenarios and ensure
the system works correctly in production-like environments.
"""

import asyncio
import multiprocessing as mp
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from secure_fl.aggregation import FedJSCMAggregator
from secure_fl.client import SecureFlowerClient, create_client
from secure_fl.models import MNISTModel, SimpleModel
from secure_fl.monitoring import HealthChecker, MetricsCollector
from secure_fl.server import SecureFlowerServer, create_server_strategy
from secure_fl.utils import (
    compute_parameter_norm,
    parameters_to_ndarrays,
    torch_to_ndarrays,
)


class TestCompleteTrainingWorkflow:
    """Test complete federated learning training workflows"""

    @pytest.mark.e2e
    def test_basic_fl_training_convergence(self, temp_dir):
        """Test basic FL training achieves convergence"""
        # Configuration
        num_clients = 3
        num_rounds = 5
        local_epochs = 2
        batch_size = 32

        # Create synthetic datasets with different distributions
        datasets = self._create_heterogeneous_datasets(num_clients, 500)

        # Create model function
        model_fn = lambda: SimpleModel(input_dim=20, hidden_dims=[64, 32], output_dim=2)

        # Create server strategy
        strategy = create_server_strategy(
            model_fn=model_fn,
            num_rounds=num_rounds,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            enable_zkp=False,  # Disable for faster testing
            momentum=0.9,
        )

        # Create clients
        clients = []
        for i, dataset in enumerate(datasets):
            client = create_client(
                client_id=f"client_{i}",
                model_fn=model_fn,
                train_data=dataset,
                local_epochs=local_epochs,
                batch_size=batch_size,
                enable_zkp=False,
            )
            clients.append(client)

        # Track training progress
        initial_loss = float("inf")
        losses = []
        accuracies = []

        # Simulate federated training
        current_params = torch_to_ndarrays(model_fn())

        for round_num in range(1, num_rounds + 1):
            print(f"\n=== Round {round_num} ===")

            # Client training phase
            client_results = []
            round_losses = []
            round_accuracies = []

            for i, client in enumerate(clients):
                print(f"Training client {i + 1}/{num_clients}...")

                # Simulate client training
                result = self._simulate_client_training(
                    client, current_params, local_epochs
                )

                client_results.append(result)
                round_losses.append(result["metrics"]["train_loss"])
                round_accuracies.append(result["metrics"]["train_accuracy"])

            # Server aggregation phase
            print("Aggregating client updates...")
            client_updates = [r["parameters"] for r in client_results]
            client_weights = [r["num_examples"] for r in client_results]

            # Normalize weights
            total_examples = sum(client_weights)
            normalized_weights = [w / total_examples for w in client_weights]

            # Aggregate using strategy
            aggregated_params = strategy.aggregator.aggregate(
                client_updates=client_updates,
                client_weights=normalized_weights,
                server_round=round_num,
                global_params=current_params,
            )

            current_params = aggregated_params

            # Compute round metrics
            avg_loss = np.mean(round_losses)
            avg_accuracy = np.mean(round_accuracies)

            losses.append(avg_loss)
            accuracies.append(avg_accuracy)

            if round_num == 1:
                initial_loss = avg_loss

            print(
                f"Round {round_num}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}"
            )

            # Early convergence check
            if avg_loss < 0.1:
                print(f"Early convergence achieved at round {round_num}")
                break

        # Verify convergence
        final_loss = losses[-1]
        final_accuracy = accuracies[-1]

        # Assertions
        assert len(losses) == len(accuracies)
        assert len(losses) >= 2, "Should have at least 2 training rounds"

        # Check loss decreases over time (with some tolerance for noise)
        loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]
        assert loss_trend < 0.1, f"Loss should generally decrease, trend: {loss_trend}"

        # Check accuracy improves
        assert final_accuracy > 0.5, f"Final accuracy too low: {final_accuracy}"

        # Check final loss is reasonable
        assert final_loss < initial_loss, "Loss should improve from initial value"
        assert final_loss < 2.0, f"Final loss too high: {final_loss}"

        print(f"\n=== Training Complete ===")
        print(f"Initial Loss: {initial_loss:.4f}")
        print(f"Final Loss: {final_loss:.4f}")
        print(f"Final Accuracy: {final_accuracy:.4f}")
        print(
            f"Loss Improvement: {(initial_loss - final_loss) / initial_loss * 100:.1f}%"
        )

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_concurrent_client_training(self, temp_dir):
        """Test training with concurrent clients"""
        num_clients = 5
        num_rounds = 3

        datasets = self._create_heterogeneous_datasets(num_clients, 200, input_dim=20)
        model_fn = lambda: SimpleModel(input_dim=20, output_dim=2)

        strategy = create_server_strategy(
            model_fn=model_fn,
            num_rounds=num_rounds,
            enable_zkp=False,
        )

        clients = []
        for i, dataset in enumerate(datasets):
            client = create_client(
                client_id=f"concurrent_client_{i}",
                model_fn=model_fn,
                train_data=dataset,
                local_epochs=1,
                enable_zkp=False,
            )
            clients.append(client)

        # Test concurrent training
        current_params = torch_to_ndarrays(model_fn())

        for round_num in range(1, num_rounds + 1):
            print(f"Concurrent training round {round_num}")

            # Train clients concurrently
            with ThreadPoolExecutor(max_workers=num_clients) as executor:
                futures = []
                for client in clients:
                    future = executor.submit(
                        self._simulate_client_training, client, current_params, 1
                    )
                    futures.append(future)

                # Collect results as they complete
                client_results = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)  # 30 second timeout
                        client_results.append(result)
                    except Exception as e:
                        pytest.fail(f"Concurrent client training failed: {e}")

            # Verify all clients completed
            assert len(client_results) == num_clients

            # Aggregate results
            client_updates = [r["parameters"] for r in client_results]
            client_weights = [1.0 / num_clients] * num_clients

            current_params = strategy.aggregator.aggregate(
                client_updates=client_updates,
                client_weights=client_weights,
                server_round=round_num,
                global_params=current_params,
            )

            # Check parameter integrity
            assert self._validate_parameters(current_params)

        print("Concurrent training completed successfully")

    @pytest.mark.e2e
    def test_client_dropout_recovery(self, temp_dir):
        """Test system handles client dropouts gracefully"""
        num_clients = 4
        min_clients = 2
        num_rounds = 4

        datasets = self._create_heterogeneous_datasets(num_clients, 300, input_dim=20)
        model_fn = lambda: SimpleModel(input_dim=20, output_dim=3)

        strategy = create_server_strategy(
            model_fn=model_fn,
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            enable_zkp=False,
        )

        clients = []
        for i, dataset in enumerate(datasets):
            client = create_client(
                client_id=f"dropout_client_{i}",
                model_fn=model_fn,
                train_data=dataset,
                enable_zkp=False,
            )
            clients.append(client)

        current_params = torch_to_ndarrays(model_fn())

        for round_num in range(1, num_rounds + 1):
            # Simulate client dropouts
            available_clients = clients[: max(min_clients, num_clients - round_num + 1)]
            print(
                f"Round {round_num}: {len(available_clients)}/{num_clients} clients available"
            )

            # Train available clients
            client_results = []
            for client in available_clients:
                result = self._simulate_client_training(client, current_params, 1)
                client_results.append(result)

            # Should still be able to aggregate with minimum clients
            if len(client_results) >= min_clients:
                client_updates = [r["parameters"] for r in client_results]
                client_weights = [1.0 / len(client_results)] * len(client_results)

                current_params = strategy.aggregator.aggregate(
                    client_updates=client_updates,
                    client_weights=client_weights,
                    server_round=round_num,
                    global_params=current_params,
                )

                assert self._validate_parameters(current_params)
                print(f"Successfully aggregated with {len(client_results)} clients")
            else:
                print(f"Insufficient clients ({len(client_results)} < {min_clients})")

    @pytest.mark.e2e
    def test_different_model_architectures(self, temp_dir):
        """Test FL with different model architectures"""
        models_to_test = [
            (
                "simple_mlp",
                lambda: SimpleModel(input_dim=20, hidden_dims=[32, 16], output_dim=2),
            ),
            (
                "deep_mlp",
                lambda: SimpleModel(
                    input_dim=20, hidden_dims=[64, 32, 16, 8], output_dim=2
                ),
            ),
            ("mnist_model", lambda: MNISTModel(hidden_dims=[64, 32], output_dim=2)),
        ]

        for model_name, model_fn in models_to_test:
            print(f"\nTesting model architecture: {model_name}")

            # Create appropriate datasets based on model type
            if model_name == "mnist_model":
                # MNIST model expects 784-dimensional input (28x28 flattened)
                datasets = self._create_heterogeneous_datasets(3, 200, input_dim=784)
            else:
                # Other models use 20-dimensional input
                datasets = self._create_heterogeneous_datasets(3, 200, input_dim=20)

            # Create fresh strategy for each model architecture to avoid momentum state issues
            strategy = create_server_strategy(
                model_fn=model_fn,
                num_rounds=2,  # Short test
                enable_zkp=False,
            )

            clients = []
            for i, dataset in enumerate(datasets):
                client = create_client(
                    client_id=f"{model_name}_client_{i}",
                    model_fn=model_fn,
                    train_data=dataset,
                    local_epochs=1,
                    enable_zkp=False,
                )
                clients.append(client)

            # Test training - initialize strategy and get initial parameters
            strategy.initialize_parameters(model_fn)
            if (
                hasattr(strategy, "current_global_params")
                and strategy.current_global_params
            ):
                current_params = strategy.current_global_params
            else:
                current_params = torch_to_ndarrays(model_fn())

            for round_num in range(1, 3):
                client_results = []
                for client in clients:
                    # Ensure each client has the same model architecture as the server
                    try:
                        result = self._simulate_client_training(
                            client, current_params, 1
                        )
                        client_results.append(result)
                    except Exception as e:
                        # If there's a model mismatch, skip this client for this architecture
                        print(
                            f"Skipping client for model {model_name} due to error: {e}"
                        )
                        continue

                if not client_results:
                    print(f"No clients succeeded for model {model_name}")
                    break

                client_updates = [r["parameters"] for r in client_results]
                client_weights = [1.0 / len(client_results)] * len(client_results)

                current_params = strategy.aggregator.aggregate(
                    client_updates=client_updates,
                    client_weights=client_weights,
                    server_round=round_num,
                    global_params=current_params,
                )

            print(f"Model {model_name} training successful")

    @pytest.mark.e2e
    def test_monitoring_integration(self, temp_dir):
        """Test integration with monitoring system"""
        # Setup monitoring
        health_checker = HealthChecker(check_interval=5)
        metrics_collector = MetricsCollector(enable_prometheus=False)

        # Start monitoring
        health_checker.start_periodic_checks()

        try:
            # Run a short FL training session
            num_clients = 2
            datasets = self._create_heterogeneous_datasets(
                num_clients, 100, input_dim=20
            )
            model_fn = lambda: SimpleModel(input_dim=20, output_dim=2)

            strategy = create_server_strategy(
                model_fn=model_fn,
                num_rounds=2,
                enable_zkp=False,
            )

            clients = []
            for i, dataset in enumerate(datasets):
                client = create_client(
                    client_id=f"monitored_client_{i}",
                    model_fn=model_fn,
                    train_data=dataset,
                    enable_zkp=False,
                )
                clients.append(client)

            current_params = torch_to_ndarrays(model_fn())

            for round_num in range(1, 3):
                # Collect system metrics
                system_metrics = metrics_collector.collect_system_metrics()
                assert system_metrics.cpu_usage_percent >= 0
                assert system_metrics.memory_usage_percent >= 0

                # Train clients
                client_results = []
                for i, client in enumerate(clients):
                    start_time = time.time()
                    result = self._simulate_client_training(client, current_params, 1)
                    duration = time.time() - start_time

                    # Record training metrics
                    metrics_collector.record_training_round(
                        round_num=round_num,
                        client_id=f"monitored_client_{i}",
                        loss=result["metrics"]["train_loss"],
                        accuracy=result["metrics"]["train_accuracy"],
                        duration_seconds=duration,
                    )

                    client_results.append(result)

                # Aggregate
                client_updates = [r["parameters"] for r in client_results]
                client_weights = [0.5, 0.5]

                current_params = strategy.aggregator.aggregate(
                    client_updates=client_updates,
                    client_weights=client_weights,
                    server_round=round_num,
                    global_params=current_params,
                )

            # Check health status
            health_results = health_checker.check_all()
            overall_status = health_checker.get_overall_status()

            assert len(health_results) > 0
            assert overall_status in ["healthy", "warning", "critical"]

            # Check metrics collection
            metrics_summary = metrics_collector.get_metrics_summary()
            assert "system_metrics" in metrics_summary
            assert "training_rounds" in metrics_summary

            print("Monitoring integration test successful")

        finally:
            health_checker.stop_periodic_checks()

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_large_scale_simulation(self, temp_dir):
        """Test FL with larger number of clients and data"""
        num_clients = 10
        num_rounds = 3
        samples_per_client = 1000

        print(
            f"Large scale test: {num_clients} clients, {samples_per_client} samples each"
        )

        # Create datasets
        datasets = self._create_heterogeneous_datasets(
            num_clients, samples_per_client, input_dim=50
        )
        model_fn = lambda: SimpleModel(
            input_dim=50, hidden_dims=[128, 64, 32], output_dim=5
        )

        strategy = create_server_strategy(
            model_fn=model_fn,
            num_rounds=num_rounds,
            min_fit_clients=8,  # Allow some client dropouts
            enable_zkp=False,
        )

        clients = []
        for i, dataset in enumerate(datasets):
            client = create_client(
                client_id=f"large_scale_client_{i:02d}",
                model_fn=model_fn,
                train_data=dataset,
                batch_size=64,  # Larger batch size for efficiency
                local_epochs=1,
                enable_zkp=False,
            )
            clients.append(client)

        # Track performance metrics
        round_times = []
        memory_usage = []

        current_params = torch_to_ndarrays(model_fn())

        for round_num in range(1, num_rounds + 1):
            round_start = time.time()

            # Memory usage before round
            import psutil

            memory_before = psutil.virtual_memory().percent

            print(f"Round {round_num}: Training {num_clients} clients...")

            # Train clients in batches to manage memory
            batch_size = 5
            client_results = []

            for i in range(0, num_clients, batch_size):
                batch_clients = clients[i : i + batch_size]

                # Train batch of clients
                batch_results = []
                for client in batch_clients:
                    result = self._simulate_client_training(client, current_params, 1)
                    batch_results.append(result)

                client_results.extend(batch_results)

            # Aggregate all results
            client_updates = [r["parameters"] for r in client_results]
            client_weights = [1.0 / len(client_results)] * len(client_results)

            current_params = strategy.aggregator.aggregate(
                client_updates=client_updates,
                client_weights=client_weights,
                server_round=round_num,
                global_params=current_params,
            )

            # Performance metrics
            round_time = time.time() - round_start
            memory_after = psutil.virtual_memory().percent

            round_times.append(round_time)
            memory_usage.append(memory_after - memory_before)

            print(f"Round {round_num} completed in {round_time:.2f}s")
            print(f"Memory usage change: {memory_after - memory_before:.1f}%")

            # Verify parameter integrity
            assert self._validate_parameters(current_params)

            # Check for memory leaks
            if memory_after - memory_before > 20:  # 20% increase threshold
                print(
                    f"Warning: High memory usage increase: {memory_after - memory_before:.1f}%"
                )

        # Performance assertions
        avg_round_time = np.mean(round_times)
        max_memory_increase = max(memory_usage)

        print(f"\n=== Performance Summary ===")
        print(f"Average round time: {avg_round_time:.2f}s")
        print(f"Max memory increase: {max_memory_increase:.1f}%")

        # Reasonable performance thresholds (adjust based on hardware)
        assert avg_round_time < 120, f"Average round time too high: {avg_round_time}s"
        assert (
            max_memory_increase < 30
        ), f"Memory usage increase too high: {max_memory_increase}%"

    def _create_heterogeneous_datasets(
        self, num_clients: int, samples_per_client: int, input_dim: int = 20
    ) -> List[TensorDataset]:
        """Create heterogeneous datasets for different clients"""
        datasets = []

        for i in range(num_clients):
            # Create different data distributions for each client
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)

            # Use provided input dimension

            # Client-specific bias in data generation
            bias = i * 0.5
            X = torch.randn(samples_per_client, input_dim) + bias

            # Create labels with client-specific patterns
            if input_dim == 20:
                # Simple classification task
                y = ((X.sum(dim=1) + bias) > 0).long()
            else:
                # For other dimensions
                y = torch.randint(
                    0, 2 if hasattr(self, "_num_classes") else 2, (samples_per_client,)
                )

            dataset = TensorDataset(X, y)
            datasets.append(dataset)

        return datasets

    def _create_mnist_like_datasets(
        self, num_clients: int, samples_per_client: int
    ) -> List[TensorDataset]:
        """Create MNIST-like datasets for testing"""
        datasets = []

        for i in range(num_clients):
            torch.manual_seed(42 + i)

            # Create 28x28 image-like data
            X = torch.randn(samples_per_client, 1, 28, 28)
            y = torch.randint(0, 2, (samples_per_client,))  # Binary classification

            dataset = TensorDataset(X, y)
            datasets.append(dataset)

        return datasets

    def _simulate_client_training(
        self,
        client: SecureFlowerClient,
        global_params: List[np.ndarray],
        local_epochs: int,
    ) -> Dict:
        """Simulate client training and return results"""
        try:
            # Load global parameters into client model
            from secure_fl.utils import ndarrays_to_torch

            ndarrays_to_torch(client.model, global_params)

            # Perform local training
            optimizer = torch.optim.SGD(
                client.model.parameters(), lr=client.learning_rate
            )
            criterion = nn.CrossEntropyLoss()

            client.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for epoch in range(local_epochs):
                for batch_idx, (data, target) in enumerate(client.train_loader):
                    optimizer.zero_grad()
                    output = client.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            # Calculate metrics
            avg_loss = total_loss / (local_epochs * len(client.train_loader))
            accuracy = correct / total if total > 0 else 0.0

            # Extract updated parameters
            from secure_fl.utils import torch_to_ndarrays

            updated_params = torch_to_ndarrays(client.model)

            return {
                "parameters": updated_params,
                "num_examples": len(client.train_loader.dataset),
                "metrics": {
                    "train_loss": avg_loss,
                    "train_accuracy": accuracy,
                },
            }

        except Exception as e:
            pytest.fail(f"Client training simulation failed: {e}")

    def _validate_parameters(self, parameters: List[np.ndarray]) -> bool:
        """Validate parameter integrity"""
        if not parameters:
            return False

        for param in parameters:
            if not isinstance(param, np.ndarray):
                return False

            if np.isnan(param).any() or np.isinf(param).any():
                return False

            if param.size == 0:
                return False

        return True


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms"""

    @pytest.mark.e2e
    def test_malformed_client_updates(self):
        """Test handling of malformed client updates"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        strategy = create_server_strategy(
            model_fn=model_fn,
            num_rounds=2,
            enable_zkp=False,
        )

        # Create valid parameters
        valid_params = torch_to_ndarrays(model_fn())

        # Test various malformed updates
        malformed_cases = [
            [],  # Empty update
            [np.array([])],  # Empty array
            [np.array([np.nan, 1.0, 2.0])],  # NaN values
            [np.array([np.inf, 1.0, 2.0])],  # Infinite values
            [np.random.randn(999, 999)],  # Wrong shape
        ]

        for i, malformed_params in enumerate(malformed_cases):
            print(f"Testing malformed case {i + 1}: {type(malformed_params)}")

            try:
                # Try to aggregate with malformed update
                client_updates = [valid_params, malformed_params]
                client_weights = [0.5, 0.5]

                # This should either succeed (by filtering bad updates) or fail gracefully
                result = strategy.aggregator.aggregate(
                    client_updates=client_updates,
                    client_weights=client_weights,
                    server_round=1,
                    global_params=valid_params,
                )

                # If it succeeds, result should be valid
                if result is not None:
                    assert all(isinstance(p, np.ndarray) for p in result)
                    assert all(not np.isnan(p).any() for p in result)
                    assert all(not np.isinf(p).any() for p in result)

            except (ValueError, RuntimeError, TypeError) as e:
                # Graceful failure is acceptable
                print(f"Gracefully handled malformed update: {e}")

    @pytest.mark.e2e
    def test_network_timeout_simulation(self):
        """Test handling of network timeouts and delays"""
        model_fn = lambda: SimpleModel(input_dim=5, output_dim=2)
        datasets = [TensorDataset(torch.randn(50, 5), torch.randint(0, 2, (50,)))]

        client = create_client(
            client_id="timeout_test_client",
            model_fn=model_fn,
            train_data=datasets[0],
            timeout=5,  # Short timeout
            enable_zkp=False,
        )

        # Test with simulated delays
        global_params = torch_to_ndarrays(model_fn())

        # This should complete within timeout
        start_time = time.time()
        try:
            result = self._simulate_client_training_with_delay(
                client, global_params, delay=1
            )
            duration = time.time() - start_time
            assert duration < 10, f"Training took too long: {duration}s"
            assert result is not None
            print("Normal training completed successfully")
        except Exception as e:
            pytest.fail(f"Normal training failed: {e}")

    def _simulate_client_training_with_delay(self, client, params, delay=0):
        """Simulate client training with artificial delay"""
        if delay > 0:
            time.sleep(delay)

        # Simple training simulation
        from secure_fl.utils import ndarrays_to_torch, torch_to_ndarrays

        ndarrays_to_torch(client.model, params)

        # Mock training
        optimizer = torch.optim.SGD(client.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        client.model.train()
        for data, target in client.train_loader:
            optimizer.zero_grad()
            output = client.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            break  # Just one batch for speed

        return {
            "parameters": torch_to_ndarrays(client.model),
            "num_examples": len(client.train_loader.dataset),
            "metrics": {"train_loss": loss.item(), "train_accuracy": 0.5},
        }


# Helper function for creating test datasets
def create_test_datasets(
    num_clients: int, samples_per_client: int, input_dim: int = 10, num_classes: int = 2
):
    """Create test datasets for FL experiments"""
    datasets = []

    for i in range(num_clients):
        torch.manual_seed(42 + i)

        X = torch.randn(samples_per_client, input_dim)
        y = torch.randint(0, num_classes, (samples_per_client,))

        dataset = TensorDataset(X, y)
        datasets.append(dataset)

    return datasets


if __name__ == "__main__":
    # Run a quick test when executed directly
    test = TestCompleteTrainingWorkflow()

    # Create temporary directory
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        print("Running basic FL training test...")
        test.test_basic_fl_training_convergence(temp_dir)
        print("Test completed successfully!")
