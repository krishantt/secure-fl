"""
Performance Benchmarks for Secure FL using pytest-benchmark

This module provides comprehensive performance benchmarks for the Secure FL framework
using pytest-benchmark for accurate timing measurements and statistical analysis.

Run with: pytest experiments/test_performance_benchmarks.py --benchmark-only
"""

import logging

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from secure_fl.aggregation import FedJSCMAggregator
from secure_fl.client import SecureFlowerClient
from secure_fl.data.dataloader import FederatedDataLoader
from secure_fl.models import MNISTModel
from secure_fl.proof_manager import ClientProofManager, ServerProofManager

logging.basicConfig(level=logging.WARNING)  # Reduce noise during benchmarks


class BenchmarkFixtures:
    """Shared fixtures for benchmarks"""

    @staticmethod
    def create_test_data(num_clients=3, samples_per_client=500):
        """Create test data for benchmarks"""
        fed_loader = FederatedDataLoader(
            dataset_name="mnist",
            num_clients=num_clients,
            iid=True,
            val_split=0.0,  # No validation for speed
            batch_size=32,
        )

        client_datasets = fed_loader.create_client_datasets()

        # Limit dataset size for benchmarking
        limited_datasets = []
        for train_data, _ in client_datasets:
            if len(train_data) > samples_per_client:
                indices = list(range(samples_per_client))
                limited_data = torch.utils.data.Subset(train_data, indices)
            else:
                limited_data = train_data

            limited_datasets.append((limited_data, None))

        return limited_datasets

    @staticmethod
    def create_test_clients(client_datasets, enable_zkp=False, model_size="small"):
        """Create test clients for benchmarks"""
        clients = []

        for i, (train_data, _) in enumerate(client_datasets):
            if model_size == "small":
                model = MNISTModel(hidden_dims=[32], output_dim=10)
            elif model_size == "medium":
                model = MNISTModel(hidden_dims=[128, 64], output_dim=10)
            else:  # large
                model = MNISTModel(hidden_dims=[256, 128, 64], output_dim=10)

            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

            client = SecureFlowerClient(
                client_id=f"client_{i}",
                model=model,
                train_loader=train_loader,
                val_loader=None,
                enable_zkp=enable_zkp,
                proof_rigor="medium" if enable_zkp else "none",
                local_epochs=1,  # Single epoch for speed
                learning_rate=0.01,
                quantize_weights=False,
            )

            clients.append(client)

        return clients


# ==============================================================================
# Proof Manager Benchmarks
# ==============================================================================


class TestProofManagerBenchmarks:
    """Benchmarks for proof manager components"""

    @pytest.mark.parametrize("param_size", [10, 50, 100, 200, 500])
    def test_client_proof_generation_scaling(self, benchmark, param_size):
        """Benchmark client proof generation with different parameter sizes"""
        manager = ClientProofManager(use_pysnark=True)

        def generate_proof():
            initial_params = [np.random.randn(param_size).astype(np.float32) * 0.1]
            updated_params = [
                initial_params[0]
                + 0.01 * np.random.randn(param_size).astype(np.float32)
            ]
            delta_norm = np.sqrt(np.sum((updated_params[0] - initial_params[0]) ** 2))

            return manager._generate_pysnark_delta_bound_proof(
                initial_params, updated_params, delta_norm + 0.1
            )

        result = benchmark(generate_proof)
        assert result is not None
        assert result.get("enabled", False)

    @pytest.mark.parametrize("num_clients", [3, 5, 10])
    def test_server_proof_generation_scaling(self, benchmark, num_clients):
        """Benchmark server proof generation with different client counts"""
        manager = ServerProofManager()
        param_size = 100

        def generate_server_proof():
            # Generate random client updates
            client_updates = []
            client_weights = []

            for _ in range(num_clients):
                client_update = [np.random.randn(param_size).astype(np.float32) * 0.01]
                client_updates.append(client_update)
                client_weights.append(np.random.uniform(0.5, 2.0))

            # Normalize weights
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]

            # Compute aggregated parameters
            aggregated_params = [np.zeros(param_size, dtype=np.float32)]
            for client_update, weight in zip(
                client_updates, client_weights, strict=False
            ):
                aggregated_params[0] += weight * client_update[0]

            momentum = [np.random.randn(param_size).astype(np.float32) * 0.001]

            return manager.generate_server_proof(
                client_updates=client_updates,
                client_weights=client_weights,
                aggregated_params=aggregated_params,
                momentum=momentum,
                momentum_coeff=0.9,
            )

        result = benchmark(generate_server_proof)
        assert result is not None

    def test_proof_verification_performance(self, benchmark):
        """Benchmark proof verification speed"""
        client_manager = ClientProofManager(use_pysnark=True)
        server_manager = ServerProofManager()

        # Generate a test proof
        initial_params = [np.random.randn(100).astype(np.float32) * 0.1]
        updated_params = [
            initial_params[0] + 0.01 * np.random.randn(100).astype(np.float32)
        ]

        proof_inputs = {
            "client_id": "test_client",
            "round": 1,
            "initial_params": initial_params,
            "updated_params": updated_params,
            "param_delta": [updated_params[0] - initial_params[0]],
            "learning_rate": 0.01,
            "local_epochs": 1,
            "total_samples": 100,
            "rigor_level": "medium",
            "data_commitment": "test_commitment",
            "batch_losses": [0.5, 0.4, 0.3],
            "gradient_norms": [1.0, 0.8, 0.6],
        }

        client_proof = client_manager.generate_training_proof(proof_inputs)

        def verify_proof():
            return server_manager.verify_client_proof(
                client_proof, updated_params, initial_params
            )

        result = benchmark(verify_proof)
        assert isinstance(result, bool)


# ==============================================================================
# Client Training Benchmarks
# ==============================================================================


class TestClientTrainingBenchmarks:
    """Benchmarks for client training performance"""

    @pytest.fixture
    def small_datasets(self):
        """Small datasets for fast benchmarks"""
        return BenchmarkFixtures.create_test_data(num_clients=3, samples_per_client=200)

    @pytest.fixture
    def medium_datasets(self):
        """Medium datasets for realistic benchmarks"""
        return BenchmarkFixtures.create_test_data(
            num_clients=5, samples_per_client=1000
        )

    def test_training_without_zkp(self, benchmark, small_datasets):
        """Benchmark training without ZKP (baseline)"""
        clients = BenchmarkFixtures.create_test_clients(
            small_datasets, enable_zkp=False, model_size="small"
        )
        client = clients[0]

        initial_params = client.get_parameters({})
        config = {"server_round": 1}

        def train_client():
            return client.fit(initial_params, config)

        updated_params, num_examples, metrics = benchmark(train_client)
        assert len(updated_params) > 0
        assert num_examples > 0
        assert metrics["proof_time"] == 0.0  # No ZKP

    def test_training_with_zkp(self, benchmark, small_datasets):
        """Benchmark training with ZKP enabled"""
        clients = BenchmarkFixtures.create_test_clients(
            small_datasets, enable_zkp=True, model_size="small"
        )
        client = clients[0]

        initial_params = client.get_parameters({})
        config = {"server_round": 1}

        def train_client():
            return client.fit(initial_params, config)

        updated_params, num_examples, metrics = benchmark(train_client)
        assert len(updated_params) > 0
        assert num_examples > 0
        assert metrics["proof_time"] > 0.0  # ZKP enabled

    @pytest.mark.parametrize("model_size", ["small", "medium", "large"])
    def test_training_model_scaling(self, benchmark, small_datasets, model_size):
        """Benchmark training with different model sizes"""
        clients = BenchmarkFixtures.create_test_clients(
            small_datasets, enable_zkp=False, model_size=model_size
        )
        client = clients[0]

        initial_params = client.get_parameters({})
        config = {"server_round": 1}

        def train_client():
            return client.fit(initial_params, config)

        updated_params, num_examples, metrics = benchmark(train_client)
        assert len(updated_params) > 0

    @pytest.mark.parametrize("local_epochs", [1, 2, 3])
    def test_training_epoch_scaling(self, benchmark, small_datasets, local_epochs):
        """Benchmark training with different numbers of local epochs"""
        clients = BenchmarkFixtures.create_test_clients(
            small_datasets, enable_zkp=False, model_size="small"
        )
        client = clients[0]
        client.local_epochs = local_epochs

        initial_params = client.get_parameters({})
        config = {"server_round": 1, "local_epochs": local_epochs}

        def train_client():
            return client.fit(initial_params, config)

        updated_params, num_examples, metrics = benchmark(train_client)
        assert len(updated_params) > 0


# ==============================================================================
# Server Aggregation Benchmarks
# ==============================================================================


class TestServerAggregationBenchmarks:
    """Benchmarks for server aggregation performance"""

    @pytest.mark.parametrize("num_clients", [3, 5, 10, 20])
    def test_aggregation_scaling(self, benchmark, num_clients):
        """Benchmark aggregation with different numbers of clients"""
        aggregator = FedJSCMAggregator(momentum=0.9, learning_rate=0.01)
        param_size = 1000

        # Generate client updates
        client_updates = []
        client_weights = []

        for _ in range(num_clients):
            # Single layer for simplicity
            client_update = [np.random.randn(param_size).astype(np.float32) * 0.01]
            client_updates.append(client_update)
            client_weights.append(np.random.uniform(0.5, 2.0))

        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]

        # Initial global parameters
        global_params = [np.random.randn(param_size).astype(np.float32)]

        def aggregate_updates():
            return aggregator.aggregate(
                client_updates=client_updates,
                client_weights=client_weights,
                server_round=1,
                global_params=global_params,
            )

        result = benchmark(aggregate_updates)
        assert len(result) == 1
        assert result[0].shape == (param_size,)

    @pytest.mark.parametrize("param_size", [100, 500, 1000, 5000])
    def test_aggregation_parameter_scaling(self, benchmark, param_size):
        """Benchmark aggregation with different parameter sizes"""
        aggregator = FedJSCMAggregator(momentum=0.9, learning_rate=0.01)
        num_clients = 5

        # Generate client updates
        client_updates = []
        client_weights = []

        for _ in range(num_clients):
            client_update = [np.random.randn(param_size).astype(np.float32) * 0.01]
            client_updates.append(client_update)
            client_weights.append(1.0 / num_clients)  # Equal weights

        global_params = [np.random.randn(param_size).astype(np.float32)]

        def aggregate_updates():
            return aggregator.aggregate(
                client_updates=client_updates,
                client_weights=client_weights,
                server_round=1,
                global_params=global_params,
            )

        result = benchmark(aggregate_updates)
        assert len(result) == 1
        assert result[0].shape == (param_size,)


# ==============================================================================
# End-to-End FL Round Benchmarks
# ==============================================================================


class TestEndToEndBenchmarks:
    """End-to-end federated learning round benchmarks"""

    @pytest.fixture
    def benchmark_datasets(self):
        """Datasets for end-to-end benchmarks"""
        return BenchmarkFixtures.create_test_data(num_clients=3, samples_per_client=300)

    def test_complete_fl_round_without_zkp(self, benchmark, benchmark_datasets):
        """Benchmark complete FL round without ZKP"""
        clients = BenchmarkFixtures.create_test_clients(
            benchmark_datasets, enable_zkp=False, model_size="small"
        )
        aggregator = FedJSCMAggregator(momentum=0.9, learning_rate=0.01)

        # Initial global parameters
        global_model = MNISTModel(hidden_dims=[32], output_dim=10)
        global_params = [
            param.detach().cpu().numpy() for param in global_model.parameters()
        ]

        def complete_fl_round():
            # Client training phase
            client_updates = []
            client_weights = []

            for client in clients:
                client.set_parameters(global_params)
                updated_params, num_examples, _ = client.fit(
                    global_params, {"server_round": 1}
                )
                client_updates.append(updated_params)
                client_weights.append(num_examples)

            # Server aggregation phase
            total_examples = sum(client_weights)
            normalized_weights = [w / total_examples for w in client_weights]

            aggregated_params = aggregator.aggregate(
                client_updates=client_updates,
                client_weights=normalized_weights,
                server_round=1,
                global_params=global_params,
            )

            return aggregated_params

        result = benchmark(complete_fl_round)
        assert len(result) > 0

    def test_complete_fl_round_with_zkp(self, benchmark, benchmark_datasets):
        """Benchmark complete FL round with ZKP"""
        clients = BenchmarkFixtures.create_test_clients(
            benchmark_datasets, enable_zkp=True, model_size="small"
        )
        aggregator = FedJSCMAggregator(momentum=0.9, learning_rate=0.01)

        # Initial global parameters
        global_model = MNISTModel(hidden_dims=[32], output_dim=10)
        global_params = [
            param.detach().cpu().numpy() for param in global_model.parameters()
        ]

        def complete_fl_round():
            # Client training phase
            client_updates = []
            client_weights = []

            for client in clients:
                client.set_parameters(global_params)
                updated_params, num_examples, _ = client.fit(
                    global_params, {"server_round": 1}
                )
                client_updates.append(updated_params)
                client_weights.append(num_examples)

            # Server aggregation phase
            total_examples = sum(client_weights)
            normalized_weights = [w / total_examples for w in client_weights]

            aggregated_params = aggregator.aggregate(
                client_updates=client_updates,
                client_weights=normalized_weights,
                server_round=1,
                global_params=global_params,
            )

            return aggregated_params

        result = benchmark(complete_fl_round)
        assert len(result) > 0

    @pytest.mark.parametrize("num_clients", [3, 5, 10])
    def test_fl_round_client_scaling(self, benchmark, num_clients):
        """Benchmark FL round scaling with number of clients"""
        datasets = BenchmarkFixtures.create_test_data(
            num_clients=num_clients, samples_per_client=200
        )
        clients = BenchmarkFixtures.create_test_clients(
            datasets, enable_zkp=False, model_size="small"
        )
        aggregator = FedJSCMAggregator(momentum=0.9, learning_rate=0.01)

        global_model = MNISTModel(hidden_dims=[32], output_dim=10)
        global_params = [
            param.detach().cpu().numpy() for param in global_model.parameters()
        ]

        def complete_fl_round():
            client_updates = []
            client_weights = []

            for client in clients:
                client.set_parameters(global_params)
                updated_params, num_examples, _ = client.fit(
                    global_params, {"server_round": 1}
                )
                client_updates.append(updated_params)
                client_weights.append(num_examples)

            total_examples = sum(client_weights)
            normalized_weights = [w / total_examples for w in client_weights]

            return aggregator.aggregate(
                client_updates=client_updates,
                client_weights=normalized_weights,
                server_round=1,
                global_params=global_params,
            )

        result = benchmark(complete_fl_round)
        assert len(result) > 0


# ==============================================================================
# Memory Usage Benchmarks
# ==============================================================================


class TestMemoryBenchmarks:
    """Memory usage benchmarks"""

    def test_memory_usage_without_zkp(self, benchmark):
        """Benchmark memory usage without ZKP"""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        datasets = BenchmarkFixtures.create_test_data(
            num_clients=3, samples_per_client=500
        )
        clients = BenchmarkFixtures.create_test_clients(
            datasets, enable_zkp=False, model_size="medium"
        )

        def measure_memory():
            initial_memory = process.memory_info().rss

            # Run training
            client = clients[0]
            global_params = client.get_parameters({})
            client.fit(global_params, {"server_round": 1})

            peak_memory = process.memory_info().rss
            return peak_memory - initial_memory

        memory_increase = benchmark(measure_memory)
        assert memory_increase >= 0

    def test_memory_usage_with_zkp(self, benchmark):
        """Benchmark memory usage with ZKP"""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        datasets = BenchmarkFixtures.create_test_data(
            num_clients=3, samples_per_client=500
        )
        clients = BenchmarkFixtures.create_test_clients(
            datasets, enable_zkp=True, model_size="medium"
        )

        def measure_memory():
            initial_memory = process.memory_info().rss

            # Run training
            client = clients[0]
            global_params = client.get_parameters({})
            client.fit(global_params, {"server_round": 1})

            peak_memory = process.memory_info().rss
            return peak_memory - initial_memory

        memory_increase = benchmark(measure_memory)
        assert memory_increase >= 0


# ==============================================================================
# Benchmark Result Analysis
# ==============================================================================


def generate_benchmark_report():
    """Generate a performance benchmark report"""
    import datetime
    import subprocess

    # Run benchmarks and save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.json"

    cmd = [
        "pytest",
        "experiments/test_performance_benchmarks.py",
        "--benchmark-only",
        "--benchmark-json",
        output_file,
        "--benchmark-sort=mean",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            print("✅ Benchmark completed successfully!")
            print(f"📊 Results saved to: {output_file}")
            print(f"📈 View with: pytest-benchmark compare {output_file}")
        else:
            print(f"❌ Benchmark failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ Benchmark timed out after 30 minutes")


if __name__ == "__main__":
    generate_benchmark_report()
