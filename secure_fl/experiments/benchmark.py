"""
Benchmark Module for Secure FL

This module provides benchmarking utilities to evaluate the performance
of the Secure FL framework across different configurations and datasets.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from secure_fl.experiments.demo import create_federated_datasets
from secure_fl.models import SimpleModel

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Benchmark runner for Secure FL framework performance evaluation.

    Evaluates different configurations and measures:
    - Training convergence speed
    - Communication overhead
    - Proof generation time (simulated)
    - Memory usage
    - Accuracy vs rounds
    """

    def __init__(self, output_dir: str = "results/benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def run_benchmark_suite(self, configs: list[dict[str, Any]]) -> dict[str, Any]:
        """Run complete benchmark suite across multiple configurations"""
        logger.info(f"Running benchmark suite with {len(configs)} configurations")

        all_results = {}

        for i, config in enumerate(configs):
            config_name = config.get("name", f"config_{i}")
            logger.info(f"Running benchmark for {config_name}")

            try:
                result = self.run_single_benchmark(config)
                all_results[config_name] = result
                logger.info(f"✓ Completed benchmark for {config_name}")

            except Exception as e:
                logger.error(f"✗ Benchmark failed for {config_name}: {e}")
                all_results[config_name] = {"error": str(e)}

        # Save results
        self.save_results(all_results)

        # Generate plots
        self.generate_plots(all_results)

        return all_results

    def run_single_benchmark(self, config: dict[str, Any]) -> dict[str, Any]:
        """Run benchmark for a single configuration"""

        # Extract config parameters
        num_clients = config.get("num_clients", 3)
        num_rounds = config.get("num_rounds", 5)
        samples_per_client = config.get("samples_per_client", 200)
        local_epochs = config.get("local_epochs", 3)
        learning_rate = config.get("learning_rate", 0.01)
        enable_zkp = config.get("enable_zkp", False)
        iid = config.get("iid", False)

        results = {
            "config": config,
            "metrics": {
                "accuracy_history": [],
                "loss_history": [],
                "round_times": [],
                "communication_overhead": [],
                "proof_times": [],
                "memory_usage": [],
            },
            "summary": {},
        }

        # Create federated datasets
        start_time = time.time()
        client_datasets = create_federated_datasets(
            num_clients=num_clients, samples_per_client=samples_per_client, iid=iid
        )
        data_creation_time = time.time() - start_time

        # Initialize global model
        global_model = SimpleModel()

        # Benchmark federated training
        total_start_time = time.time()

        for round_num in range(num_rounds):
            round_start = time.time()

            # Simulate federated round with metrics collection
            round_metrics = self._benchmark_round(
                global_model, client_datasets, local_epochs, learning_rate, enable_zkp
            )

            round_time = time.time() - round_start

            # Record metrics
            results["metrics"]["accuracy_history"].append(round_metrics["accuracy"])
            results["metrics"]["loss_history"].append(round_metrics["loss"])
            results["metrics"]["round_times"].append(round_time)
            results["metrics"]["communication_overhead"].append(
                round_metrics["comm_overhead"]
            )
            results["metrics"]["proof_times"].append(round_metrics["proof_time"])
            results["metrics"]["memory_usage"].append(round_metrics["memory_mb"])

            logger.debug(
                f"Round {round_num + 1}: Acc={round_metrics['accuracy']:.3f}, "
                f"Time={round_time:.2f}s"
            )

        total_time = time.time() - total_start_time

        # Calculate summary statistics
        results["summary"] = {
            "total_time": total_time,
            "data_creation_time": data_creation_time,
            "final_accuracy": results["metrics"]["accuracy_history"][-1],
            "final_loss": results["metrics"]["loss_history"][-1],
            "avg_round_time": np.mean(results["metrics"]["round_times"]),
            "total_proof_time": sum(results["metrics"]["proof_times"]),
            "avg_communication_overhead": np.mean(
                results["metrics"]["communication_overhead"]
            ),
            "peak_memory_usage": max(results["metrics"]["memory_usage"]),
            "convergence_rounds": self._calculate_convergence_rounds(
                results["metrics"]["accuracy_history"]
            ),
        }

        return results

    def _benchmark_round(
        self,
        global_model: nn.Module,
        client_datasets: list[tuple[DataLoader, DataLoader]],
        local_epochs: int,
        learning_rate: float,
        enable_zkp: bool,
    ) -> dict[str, float]:
        """Benchmark a single federated learning round"""

        round_start = time.time()
        client_models = []
        client_weights = []

        # Simulate proof generation time
        proof_time = 0.0
        if enable_zkp:
            # Simulated ZKP times based on complexity
            proof_time = np.random.normal(1.2, 0.3)  # ~1.2s ± 0.3s

        # Client training simulation
        total_samples = 0
        total_correct = 0
        total_loss = 0.0

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

            # Evaluate
            local_model.eval()
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = local_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    _, predicted = torch.max(outputs.data, 1)

                    total_loss += loss.item()
                    total_samples += batch_y.size(0)
                    total_correct += (predicted == batch_y).sum().item()

            client_models.append(local_model)
            # Calculate training samples by counting
            train_samples = sum(batch.size(0) for batch, _ in train_loader)
            client_weights.append(train_samples)

        # Federated aggregation (FedAvg)
        total_weight = sum(client_weights)
        aggregated_state = {}

        for key in global_model.state_dict().keys():
            aggregated_state[key] = torch.zeros_like(global_model.state_dict()[key])

        for client_model, weight in zip(client_models, client_weights):
            client_state = client_model.state_dict()
            weight_ratio = weight / total_weight

            for key in aggregated_state.keys():
                aggregated_state[key] += weight_ratio * client_state[key]

        global_model.load_state_dict(aggregated_state)

        # Calculate metrics
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = (
            total_loss / len(client_datasets) if len(client_datasets) > 0 else 0.0
        )

        # Simulate communication overhead (percentage)
        base_comm = 100.0  # Base communication cost
        zkp_overhead = 15.0 if enable_zkp else 0.0  # ZKP adds ~15% overhead
        comm_overhead = base_comm + zkp_overhead

        # Simulate memory usage (MB)
        base_memory = 50.0
        model_memory = sum(p.numel() * 4 for p in global_model.parameters()) / (
            1024 * 1024
        )
        memory_usage = base_memory + model_memory + (proof_time * 10)  # Rough estimate

        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "proof_time": proof_time,
            "comm_overhead": comm_overhead,
            "memory_mb": memory_usage,
        }

    def _calculate_convergence_rounds(self, accuracy_history: list[float]) -> int:
        """Calculate number of rounds to reach convergence (stability)"""
        if len(accuracy_history) < 3:
            return len(accuracy_history)

        # Look for when accuracy stops improving significantly
        for i in range(2, len(accuracy_history)):
            if i >= 3:  # Need at least 3 rounds to check stability
                recent_improvement = (accuracy_history[i] - accuracy_history[i - 2]) / 2
                if abs(recent_improvement) < 0.01:  # Less than 1% improvement
                    return i + 1

        return len(accuracy_history)

    def save_results(self, results: dict[str, Any]):
        """Save benchmark results to JSON file"""
        output_file = self.output_dir / "benchmark_results.json"

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        results_serializable = convert_numpy(results)

        with open(output_file, "w") as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Benchmark results saved to {output_file}")

    def generate_plots(self, results: dict[str, Any]):
        """Generate visualization plots for benchmark results"""

        # Set up plotting style
        plt.style.use(
            "seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default"
        )

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Secure FL Benchmark Results", fontsize=16)

        # Plot 1: Accuracy over rounds
        ax1 = axes[0, 0]
        for config_name, result in results.items():
            if "error" not in result:
                rounds = range(1, len(result["metrics"]["accuracy_history"]) + 1)
                ax1.plot(
                    rounds,
                    result["metrics"]["accuracy_history"],
                    label=config_name,
                    marker="o",
                )
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Training Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Round times
        ax2 = axes[0, 1]
        config_names = []
        round_times = []
        for config_name, result in results.items():
            if "error" not in result:
                config_names.append(config_name)
                round_times.append(result["summary"]["avg_round_time"])

        ax2.bar(config_names, round_times)
        ax2.set_xlabel("Configuration")
        ax2.set_ylabel("Average Round Time (s)")
        ax2.set_title("Performance Comparison")
        ax2.tick_params(axis="x", rotation=45)

        # Plot 3: Communication overhead
        ax3 = axes[1, 0]
        comm_overheads = []
        for config_name, result in results.items():
            if "error" not in result:
                comm_overheads.append(result["summary"]["avg_communication_overhead"])

        ax3.bar(config_names, comm_overheads)
        ax3.set_xlabel("Configuration")
        ax3.set_ylabel("Communication Overhead (%)")
        ax3.set_title("Communication Cost")
        ax3.tick_params(axis="x", rotation=45)

        # Plot 4: Memory usage
        ax4 = axes[1, 1]
        memory_usage = []
        for config_name, result in results.items():
            if "error" not in result:
                memory_usage.append(result["summary"]["peak_memory_usage"])

        ax4.bar(config_names, memory_usage)
        ax4.set_xlabel("Configuration")
        ax4.set_ylabel("Peak Memory (MB)")
        ax4.set_title("Memory Usage")
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plot_path = self.output_dir / "benchmark_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Benchmark plots saved to {plot_path}")


def get_default_benchmark_configs() -> list[dict[str, Any]]:
    """Get default benchmark configurations"""
    return [
        {
            "name": "baseline",
            "num_clients": 3,
            "num_rounds": 5,
            "samples_per_client": 200,
            "enable_zkp": False,
            "iid": True,
        },
        {
            "name": "non_iid",
            "num_clients": 3,
            "num_rounds": 5,
            "samples_per_client": 200,
            "enable_zkp": False,
            "iid": False,
        },
        {
            "name": "with_zkp",
            "num_clients": 3,
            "num_rounds": 5,
            "samples_per_client": 200,
            "enable_zkp": True,
            "iid": False,
        },
        {
            "name": "scaled_up",
            "num_clients": 5,
            "num_rounds": 8,
            "samples_per_client": 300,
            "enable_zkp": False,
            "iid": False,
        },
    ]


def main():
    """Main entry point for benchmark script"""
    parser = argparse.ArgumentParser(description="Secure FL Benchmark Suite")
    parser.add_argument(
        "--config", type=str, help="Path to benchmark config file (JSON)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with reduced parameters",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configurations
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            configs = json.load(f)
    else:
        configs = get_default_benchmark_configs()

        # Adjust for quick benchmark
        if args.quick:
            for config in configs:
                config["num_rounds"] = 3
                config["samples_per_client"] = 100

    logger.info("Starting Secure FL benchmark suite...")
    logger.info(f"Running {len(configs)} benchmark configurations")

    # Run benchmarks
    runner = BenchmarkRunner(output_dir=args.output_dir)
    results = runner.run_benchmark_suite(configs)

    # Print summary
    print("\n" + "=" * 60)
    print("SECURE FL BENCHMARK SUMMARY")
    print("=" * 60)

    for config_name, result in results.items():
        if "error" in result:
            print(f"{config_name:15s}: ERROR - {result['error']}")
        else:
            summary = result["summary"]
            print(
                f"{config_name:15s}: "
                f"Acc={summary['final_accuracy']:.3f}, "
                f"Time={summary['total_time']:.1f}s, "
                f"Rounds={summary['convergence_rounds']}"
            )

    print(f"\nDetailed results saved to: {args.output_dir}")
    logger.info("Benchmark suite completed!")


if __name__ == "__main__":
    main()
