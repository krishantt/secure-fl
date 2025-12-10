"""
Enhanced Performance Benchmark for Secure FL

This module provides comprehensive performance benchmarking with detailed timing metrics
for client proof generation, server aggregation, and overall system overhead.

Key Metrics Measured:
- Client proof generation time
- Server aggregation time
- Server proof verification time
- Training time without proofs
- Memory usage and throughput
- Convergence analysis
"""

import gc
import json
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from secure_fl.aggregation import FedJSCMAggregator
from secure_fl.client import SecureFlowerClient, create_client
from secure_fl.data.dataloader import FederatedDataLoader
from secure_fl.models import CIFAR10Model, MNISTModel, SimpleModel
from secure_fl.proof_manager import ClientProofManager, ServerProofManager
from secure_fl.server import SecureFlowerStrategy
from secure_fl.utils import ndarrays_to_torch, torch_to_ndarrays

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class PerformanceProfiler:
    """Detailed performance profiler for FL operations"""

    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.process = psutil.Process()

    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.perf_counter()

    def end_timer(self, operation: str) -> float:
        """End timing and record duration"""
        if operation in self.start_times:
            duration = time.perf_counter() - self.start_times[operation]
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            del self.start_times[operation]
            return duration
        return 0.0

    def record_memory(self, label: str):
        """Record current memory usage"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        if f"memory_{label}" not in self.metrics:
            self.metrics[f"memory_{label}"] = []
        self.metrics[f"memory_{label}"].append(memory_mb)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of all metrics"""
        stats = {}
        for metric, values in self.metrics.items():
            if values:
                stats[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values),
                }
        return stats


class DatasetLoader:
    """Simplified dataset loader for benchmarking"""

    def __init__(self, data_dir: str = None):
        # Not used anymore, kept for compatibility
        pass

    def load_mnist(self, num_clients: int = 3) -> Tuple[List, Any]:
        """Load MNIST dataset split for FL using centralized loader"""
        fed_loader = FederatedDataLoader(
            dataset_name="mnist",
            num_clients=num_clients,
            iid=True,
            val_split=0.2,
            batch_size=32,
        )

        # Get client datasets
        client_datasets = fed_loader.create_client_datasets()

        # Get test dataset for global evaluation
        _, test_dataset = fed_loader.load_dataset()

        return client_datasets, test_dataset

    def get_model_fn(self, dataset_name: str):
        """Get model function for dataset"""
        if dataset_name == "mnist":
            return lambda: MNISTModel(hidden_dims=[128, 64], output_dim=10)
        else:
            return lambda: SimpleModel(input_dim=784, hidden_dim=64, output_dim=10)


class SecureFlBenchmark:
    """Comprehensive benchmark for Secure FL performance"""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.profiler = PerformanceProfiler()
        self.dataset_loader = DatasetLoader()

    def run_comprehensive_benchmark(
        self,
        datasets: List[str] = ["mnist"],
        num_clients_list: List[int] = [3, 5, 10],
        num_rounds: int = 10,
        zkp_configs: List[bool] = [False, True],
        rigor_levels: List[str] = ["low", "medium", "high"],
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark across multiple configurations"""

        logger.info("🚀 Starting Comprehensive Secure FL Benchmark")
        logger.info(f"Datasets: {datasets}")
        logger.info(f"Client counts: {num_clients_list}")
        logger.info(f"Rounds: {num_rounds}")
        logger.info(f"ZKP configs: {zkp_configs}")
        logger.info(f"Rigor levels: {rigor_levels}")

        all_results = {}

        for dataset in datasets:
            logger.info(f"\n📊 Benchmarking dataset: {dataset}")

            for num_clients in num_clients_list:
                logger.info(f"\n👥 Testing with {num_clients} clients")

                for enable_zkp in zkp_configs:
                    if enable_zkp:
                        # Test different rigor levels when ZKP is enabled
                        test_rigor_levels = rigor_levels
                    else:
                        # Only test once when ZKP is disabled
                        test_rigor_levels = ["none"]

                    for rigor in test_rigor_levels:
                        config_name = f"{dataset}_clients{num_clients}_zkp{enable_zkp}_rigor{rigor}"

                        logger.info(f"\n🔧 Config: {config_name}")

                        try:
                            result = self._run_single_benchmark(
                                dataset=dataset,
                                num_clients=num_clients,
                                num_rounds=num_rounds,
                                enable_zkp=enable_zkp,
                                proof_rigor=rigor if enable_zkp else "none",
                            )
                            all_results[config_name] = result

                            # Save intermediate results
                            self._save_results(all_results, "intermediate_results.json")

                        except Exception as e:
                            logger.error(f"❌ Failed config {config_name}: {e}")
                            continue

        # Save final results and generate plots
        self._save_results(all_results, "final_results.json")
        self._generate_comprehensive_plots(all_results)

        logger.info(f"✅ Benchmark completed! Results saved to {self.output_dir}")
        return all_results

    def _run_single_benchmark(
        self,
        dataset: str,
        num_clients: int,
        num_rounds: int,
        enable_zkp: bool,
        proof_rigor: str,
    ) -> Dict[str, Any]:
        """Run benchmark for a single configuration"""

        # Load dataset
        client_datasets, test_dataset = self.dataset_loader.load_mnist(num_clients)
        model_fn = self.dataset_loader.get_model_fn(dataset)

        # Initialize aggregator and proof managers
        aggregator = FedJSCMAggregator(momentum=0.9, learning_rate=0.01)
        client_proof_manager = ClientProofManager() if enable_zkp else None
        server_proof_manager = ServerProofManager() if enable_zkp else None

        # Results storage
        results = {
            "config": {
                "dataset": dataset,
                "num_clients": num_clients,
                "num_rounds": num_rounds,
                "enable_zkp": enable_zkp,
                "proof_rigor": proof_rigor,
            },
            "round_metrics": [],
            "client_metrics": [],
            "server_metrics": [],
            "aggregation_metrics": [],
            "memory_usage": [],
            "convergence_metrics": [],
        }

        # Create clients
        clients = []
        for i in range(num_clients):
            train_data, val_data = client_datasets[i]
            client = SecureFlowerClient(
                client_id=f"client_{i}",
                model=model_fn(),
                train_loader=DataLoader(train_data, batch_size=32, shuffle=True),
                val_loader=DataLoader(val_data, batch_size=32) if val_data else None,
                enable_zkp=enable_zkp,
                proof_rigor=proof_rigor,
                local_epochs=2,
                learning_rate=0.01,
            )
            clients.append(client)

        # Initialize global model
        global_model = model_fn()
        global_params = [
            param.detach().cpu().numpy() for param in global_model.parameters()
        ]

        logger.info(
            f"Starting FL simulation with {num_clients} clients for {num_rounds} rounds"
        )

        # FL Training Loop
        for round_num in range(num_rounds):
            logger.info(f"Round {round_num + 1}/{num_rounds}")

            self.profiler.start_timer("round_total")
            self.profiler.record_memory("round_start")

            # Client training phase
            client_updates = []
            client_weights = []
            round_client_metrics = []

            for client_idx, client in enumerate(clients):
                self.profiler.start_timer(f"client_{client_idx}_total")

                # Training without proof (baseline)
                client_no_zkp = SecureFlowerClient(
                    client_id=f"client_{client_idx}_no_zkp",
                    model=model_fn(),
                    train_loader=client.train_loader,
                    val_loader=client.val_loader,
                    enable_zkp=False,
                    local_epochs=2,
                    learning_rate=0.01,
                )
                client_no_zkp.set_parameters(global_params)

                self.profiler.start_timer(f"client_{client_idx}_train_no_zkp")
                config = {"server_round": round_num + 1}
                _, _, metrics_no_zkp = client_no_zkp.fit(global_params, config)
                training_time_no_zkp = self.profiler.end_timer(
                    f"client_{client_idx}_train_no_zkp"
                )

                # Training with proof (if enabled)
                client.set_parameters(global_params)

                self.profiler.start_timer(f"client_{client_idx}_train_with_zkp")
                updated_params, num_examples, metrics_with_zkp = client.fit(
                    global_params, config
                )
                training_time_with_zkp = self.profiler.end_timer(
                    f"client_{client_idx}_train_with_zkp"
                )

                client_total_time = self.profiler.end_timer(
                    f"client_{client_idx}_total"
                )

                # Extract proof generation time
                proof_time = metrics_with_zkp.get("proof_time", 0.0)
                pure_training_time = training_time_with_zkp - proof_time

                client_metrics = {
                    "client_id": client_idx,
                    "round": round_num + 1,
                    "training_time_no_zkp": training_time_no_zkp,
                    "training_time_with_zkp": training_time_with_zkp,
                    "pure_training_time": pure_training_time,
                    "proof_generation_time": proof_time,
                    "total_time": client_total_time,
                    "num_examples": num_examples,
                    "train_loss": metrics_with_zkp.get("train_loss", 0.0),
                    "train_accuracy": metrics_with_zkp.get("train_accuracy", 0.0),
                    "zkp_overhead_ratio": proof_time / pure_training_time
                    if pure_training_time > 0
                    else 0,
                    "enable_zkp": enable_zkp,
                }

                round_client_metrics.append(client_metrics)
                client_updates.append(updated_params)
                client_weights.append(num_examples)

            # Server aggregation phase
            self.profiler.start_timer("server_aggregation")

            # Normalize weights
            total_examples = sum(client_weights)
            normalized_weights = [w / total_examples for w in client_weights]

            # Perform aggregation
            aggregated_params = aggregator.aggregate(
                client_updates=client_updates,
                client_weights=normalized_weights,
                server_round=round_num + 1,
                global_params=global_params,
            )

            aggregation_time = self.profiler.end_timer("server_aggregation")

            # Server proof generation (if enabled)
            server_proof_time = 0.0
            if enable_zkp and server_proof_manager:
                self.profiler.start_timer("server_proof")
                try:
                    server_proof = server_proof_manager.generate_server_proof(
                        client_updates=client_updates,
                        client_weights=normalized_weights,
                        aggregated_params=aggregated_params,
                        momentum=aggregator.server_momentum,
                        momentum_coeff=aggregator.momentum,
                    )
                    server_proof_time = self.profiler.end_timer("server_proof")
                except Exception as e:
                    logger.warning(f"Server proof generation failed: {e}")
                    server_proof_time = self.profiler.end_timer("server_proof")

            # Update global parameters
            global_params = aggregated_params

            # Calculate convergence metrics
            param_norm = np.sqrt(sum(np.sum(p**2) for p in global_params))

            self.profiler.record_memory("round_end")
            round_total_time = self.profiler.end_timer("round_total")

            # Store round metrics
            round_metrics = {
                "round": round_num + 1,
                "aggregation_time": aggregation_time,
                "server_proof_time": server_proof_time,
                "round_total_time": round_total_time,
                "param_norm": param_norm,
                "num_clients": num_clients,
                "avg_client_proof_time": np.mean(
                    [m["proof_generation_time"] for m in round_client_metrics]
                ),
                "total_proof_overhead": np.sum(
                    [m["proof_generation_time"] for m in round_client_metrics]
                )
                + server_proof_time,
                "avg_training_time": np.mean(
                    [m["pure_training_time"] for m in round_client_metrics]
                ),
                "enable_zkp": enable_zkp,
            }

            results["round_metrics"].append(round_metrics)
            results["client_metrics"].extend(round_client_metrics)

            # Log progress
            total_proof_time = round_metrics["total_proof_overhead"]
            avg_train_time = round_metrics["avg_training_time"]
            logger.info(
                f"  Round {round_num + 1}: agg_time={aggregation_time:.2f}s, "
                f"proof_time={total_proof_time:.2f}s, train_time={avg_train_time:.2f}s"
            )

        # Calculate summary statistics
        results["summary"] = self._calculate_summary_stats(results)

        # Cleanup
        del clients, global_model, aggregator
        gc.collect()

        return results

    def _calculate_summary_stats(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate summary statistics from benchmark results"""
        round_metrics = results["round_metrics"]
        client_metrics = results["client_metrics"]

        if not round_metrics or not client_metrics:
            return {}

        # Aggregation timing stats
        agg_times = [r["aggregation_time"] for r in round_metrics]
        server_proof_times = [r["server_proof_time"] for r in round_metrics]

        # Client timing stats
        client_proof_times = [c["proof_generation_time"] for c in client_metrics]
        training_times = [c["pure_training_time"] for c in client_metrics]

        # Overhead calculations
        total_proof_overhead = [r["total_proof_overhead"] for r in round_metrics]
        avg_training_time = [r["avg_training_time"] for r in round_metrics]

        overhead_ratios = []
        for i in range(len(total_proof_overhead)):
            if avg_training_time[i] > 0:
                overhead_ratios.append(total_proof_overhead[i] / avg_training_time[i])

        return {
            "avg_aggregation_time": np.mean(agg_times),
            "std_aggregation_time": np.std(agg_times),
            "avg_server_proof_time": np.mean(server_proof_times),
            "std_server_proof_time": np.std(server_proof_times),
            "avg_client_proof_time": np.mean(client_proof_times),
            "std_client_proof_time": np.std(client_proof_times),
            "avg_training_time": np.mean(training_times),
            "std_training_time": np.std(training_times),
            "avg_overhead_ratio": np.mean(overhead_ratios) if overhead_ratios else 0.0,
            "std_overhead_ratio": np.std(overhead_ratios) if overhead_ratios else 0.0,
            "max_overhead_ratio": np.max(overhead_ratios) if overhead_ratios else 0.0,
            "total_rounds": len(round_metrics),
            "total_clients": len(set(c["client_id"] for c in client_metrics)),
            "convergence_rate": self._estimate_convergence_rate(round_metrics),
        }

    def _estimate_convergence_rate(self, round_metrics: List[Dict]) -> float:
        """Estimate convergence rate from parameter norms"""
        if len(round_metrics) < 2:
            return 0.0

        param_norms = [r["param_norm"] for r in round_metrics]

        # Calculate rate of change in parameter norms
        changes = []
        for i in range(1, len(param_norms)):
            if param_norms[i - 1] > 0:
                change_rate = (
                    abs(param_norms[i] - param_norms[i - 1]) / param_norms[i - 1]
                )
                changes.append(change_rate)

        return np.mean(changes) if changes else 0.0

    def _save_results(self, results: Dict[str, Any], filename: str):
        """Save benchmark results to JSON file"""
        output_file = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for config, data in results.items():
            serializable_results[config] = self._make_json_serializable(data)

        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {output_file}")

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    def _generate_comprehensive_plots(self, results: Dict[str, Any]):
        """Generate comprehensive performance plots"""
        logger.info("📈 Generating performance plots...")

        # Convert results to DataFrame for easier plotting
        df_rounds, df_clients = self._results_to_dataframes(results)

        if df_rounds.empty or df_clients.empty:
            logger.warning("No data available for plotting")
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # 1. ZKP Overhead Comparison
        ax1 = plt.subplot(3, 3, 1)
        self._plot_zkp_overhead(df_rounds, ax1)

        # 2. Client Proof Generation Time
        ax2 = plt.subplot(3, 3, 2)
        self._plot_client_proof_times(df_clients, ax2)

        # 3. Server Performance Breakdown
        ax3 = plt.subplot(3, 3, 3)
        self._plot_server_breakdown(df_rounds, ax3)

        # 4. Scalability Analysis
        ax4 = plt.subplot(3, 3, 4)
        self._plot_scalability(df_rounds, ax4)

        # 5. Training vs Proof Time
        ax5 = plt.subplot(3, 3, 5)
        self._plot_training_vs_proof(df_clients, ax5)

        # 6. Overhead Ratio by Configuration
        ax6 = plt.subplot(3, 3, 6)
        self._plot_overhead_ratios(df_rounds, ax6)

        # 7. Round-by-Round Performance
        ax7 = plt.subplot(3, 3, 7)
        self._plot_round_performance(df_rounds, ax7)

        # 8. Memory Usage
        ax8 = plt.subplot(3, 3, 8)
        self._plot_memory_usage(ax8)

        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        self._plot_summary_stats(results, ax9)

        plt.tight_layout()
        plot_file = self.output_dir / "comprehensive_performance_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        # Generate individual detailed plots
        self._generate_detailed_plots(df_rounds, df_clients, results)

        logger.info(f"📊 Plots saved to {self.output_dir}")

    def _results_to_dataframes(
        self, results: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert results dictionary to pandas DataFrames"""
        round_data = []
        client_data = []

        for config_name, config_results in results.items():
            if (
                "round_metrics" not in config_results
                or "client_metrics" not in config_results
            ):
                continue

            config_info = config_results.get("config", {})

            # Process round metrics
            for round_metric in config_results["round_metrics"]:
                row = round_metric.copy()
                row.update(config_info)
                row["config_name"] = config_name
                round_data.append(row)

            # Process client metrics
            for client_metric in config_results["client_metrics"]:
                row = client_metric.copy()
                row.update(config_info)
                row["config_name"] = config_name
                client_data.append(row)

        df_rounds = pd.DataFrame(round_data) if round_data else pd.DataFrame()
        df_clients = pd.DataFrame(client_data) if client_data else pd.DataFrame()

        return df_rounds, df_clients

    def _plot_zkp_overhead(self, df: pd.DataFrame, ax):
        """Plot ZKP overhead comparison"""
        if df.empty or "enable_zkp" not in df.columns:
            ax.text(0.5, 0.5, "No ZKP data available", ha="center", va="center")
            ax.set_title("ZKP Overhead Analysis")
            return

        # Group by ZKP enabled/disabled
        zkp_comparison = (
            df.groupby(["enable_zkp", "num_clients"])
            .agg({"total_proof_overhead": "mean", "avg_training_time": "mean"})
            .reset_index()
        )

        zkp_enabled = zkp_comparison[zkp_comparison["enable_zkp"] == True]
        zkp_disabled = zkp_comparison[zkp_comparison["enable_zkp"] == False]

        if not zkp_enabled.empty:
            ax.bar(
                zkp_enabled["num_clients"] - 0.2,
                zkp_enabled["total_proof_overhead"],
                width=0.4,
                label="Proof Overhead",
                alpha=0.8,
            )
        if not zkp_disabled.empty:
            ax.bar(
                zkp_disabled["num_clients"] + 0.2,
                zkp_disabled["avg_training_time"],
                width=0.4,
                label="Training Time (No ZKP)",
                alpha=0.8,
            )

        ax.set_xlabel("Number of Clients")
        ax.set_ylabel("Time (seconds)")
        ax.set_title("ZKP Overhead vs Training Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_client_proof_times(self, df: pd.DataFrame, ax):
        """Plot client proof generation times"""
        if df.empty or "proof_generation_time" not in df.columns:
            ax.text(0.5, 0.5, "No client proof data", ha="center", va="center")
            ax.set_title("Client Proof Generation Time")
            return

        zkp_data = df[df["enable_zkp"] == True]
        if zkp_data.empty:
            ax.text(0.5, 0.5, "No ZKP data available", ha="center", va="center")
            ax.set_title("Client Proof Generation Time")
            return

        sns.boxplot(data=zkp_data, x="num_clients", y="proof_generation_time", ax=ax)
        ax.set_xlabel("Number of Clients")
        ax.set_ylabel("Proof Generation Time (seconds)")
        ax.set_title("Client Proof Generation Time Distribution")
        ax.grid(True, alpha=0.3)

    def _plot_server_breakdown(self, df: pd.DataFrame, ax):
        """Plot server performance breakdown"""
        if df.empty:
            ax.text(0.5, 0.5, "No server data available", ha="center", va="center")
            ax.set_title("Server Performance Breakdown")
            return

        # Average server times by configuration
        server_metrics = (
            df.groupby(["enable_zkp", "num_clients"])
            .agg({"aggregation_time": "mean", "server_proof_time": "mean"})
            .reset_index()
        )

        x_pos = range(len(server_metrics))
        labels = [
            f"ZKP={row['enable_zkp']}\nClients={row['num_clients']}"
            for _, row in server_metrics.iterrows()
        ]

        ax.bar(
            x_pos, server_metrics["aggregation_time"], label="Aggregation", alpha=0.8
        )
        ax.bar(
            x_pos,
            server_metrics["server_proof_time"],
            bottom=server_metrics["aggregation_time"],
            label="Server Proof",
            alpha=0.8,
        )

        ax.set_xlabel("Configuration")
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Server Performance Breakdown")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_scalability(self, df: pd.DataFrame, ax):
        """Plot scalability analysis"""
        if df.empty:
            ax.text(0.5, 0.5, "No scalability data", ha="center", va="center")
            ax.set_title("Scalability Analysis")
            return

        scalability = (
            df.groupby(["enable_zkp", "num_clients"])
            .agg({"round_total_time": "mean"})
            .reset_index()
        )

        zkp_enabled = scalability[scalability["enable_zkp"] == True]
        zkp_disabled = scalability[scalability["enable_zkp"] == False]

        if not zkp_enabled.empty:
            ax.plot(
                zkp_enabled["num_clients"],
                zkp_enabled["round_total_time"],
                "o-",
                label="With ZKP",
                linewidth=2,
                markersize=6,
            )
        if not zkp_disabled.empty:
            ax.plot(
                zkp_disabled["num_clients"],
                zkp_disabled["round_total_time"],
                "s--",
                label="Without ZKP",
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Number of Clients")
        ax.set_ylabel("Round Time (seconds)")
        ax.set_title("Scalability: Round Time vs Client Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_training_vs_proof(self, df: pd.DataFrame, ax):
        """Plot training time vs proof time correlation"""
        if df.empty:
            ax.text(0.5, 0.5, "No training/proof data", ha="center", va="center")
            ax.set_title("Training vs Proof Time")
            return

        zkp_data = df[df["enable_zkp"] == True]
        if zkp_data.empty or "pure_training_time" not in zkp_data.columns:
            ax.text(0.5, 0.5, "No ZKP correlation data", ha="center", va="center")
            ax.set_title("Training vs Proof Time")
            return

        ax.scatter(
            zkp_data["pure_training_time"],
            zkp_data["proof_generation_time"],
            alpha=0.6,
            s=50,
        )
        ax.set_xlabel("Training Time (seconds)")
        ax.set_ylabel("Proof Generation Time (seconds)")
        ax.set_title("Training vs Proof Time Correlation")

        # Add trend line
        if len(zkp_data) > 1:
            z = np.polyfit(
                zkp_data["pure_training_time"], zkp_data["proof_generation_time"], 1
            )
            p = np.poly1d(z)
            ax.plot(
                zkp_data["pure_training_time"],
                p(zkp_data["pure_training_time"]),
                "r--",
                alpha=0.8,
                linewidth=2,
            )

        ax.grid(True, alpha=0.3)

    def _plot_overhead_ratios(self, df: pd.DataFrame, ax):
        """Plot overhead ratios by configuration"""
        if df.empty:
            ax.text(0.5, 0.5, "No overhead data", ha="center", va="center")
            ax.set_title("Overhead Ratios")
            return

        # Calculate overhead ratios
        overhead_ratios = []
        for _, row in df.iterrows():
            if row["avg_training_time"] > 0:
                ratio = row["total_proof_overhead"] / row["avg_training_time"]
                overhead_ratios.append(
                    {
                        "config": f"Clients={row['num_clients']}, ZKP={row['enable_zkp']}",
                        "overhead_ratio": ratio,
                        "enable_zkp": row["enable_zkp"],
                        "num_clients": row["num_clients"],
                    }
                )

        if not overhead_ratios:
            ax.text(0.5, 0.5, "No valid overhead ratios", ha="center", va="center")
            ax.set_title("Overhead Ratios")
            return

        df_overhead = pd.DataFrame(overhead_ratios)
        zkp_data = df_overhead[df_overhead["enable_zkp"] == True]

        if not zkp_data.empty:
            sns.barplot(data=zkp_data, x="num_clients", y="overhead_ratio", ax=ax)

        ax.set_xlabel("Number of Clients")
        ax.set_ylabel("Overhead Ratio (Proof/Training)")
        ax.set_title("ZKP Overhead Ratio by Client Count")
        ax.grid(True, alpha=0.3)

    def _plot_round_performance(self, df: pd.DataFrame, ax):
        """Plot round-by-round performance"""
        if df.empty:
            ax.text(0.5, 0.5, "No round data available", ha="center", va="center")
            ax.set_title("Round Performance")
            return

        # Plot performance over rounds for different configurations
        for (enable_zkp, num_clients), group in df.groupby(
            ["enable_zkp", "num_clients"]
        ):
            label = f"ZKP={enable_zkp}, Clients={num_clients}"
            ax.plot(
                group["round"], group["round_total_time"], "o-", label=label, alpha=0.8
            )

        ax.set_xlabel("Round Number")
        ax.set_ylabel("Round Total Time (seconds)")
        ax.set_title("Performance Over Training Rounds")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    def _plot_memory_usage(self, ax):
        """Plot memory usage analysis"""
        memory_stats = self.profiler.get_stats()
        memory_keys = [k for k in memory_stats.keys() if k.startswith("memory_")]

        if not memory_keys:
            ax.text(0.5, 0.5, "No memory data collected", ha="center", va="center")
            ax.set_title("Memory Usage")
            return

        labels = [k.replace("memory_", "").title() for k in memory_keys]
        values = [memory_stats[k]["mean"] for k in memory_keys]

        ax.bar(labels, values, alpha=0.8)
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_title("Average Memory Usage")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    def _plot_summary_stats(self, results: Dict[str, Any], ax):
        """Plot summary statistics"""
        summary_data = []

        for config_name, config_results in results.items():
            summary = config_results.get("summary", {})
            if summary:
                summary_data.append(
                    {
                        "config": config_name.replace("mnist_", "").replace(
                            "_zkp", "\nZKP="
                        ),
                        "avg_overhead_ratio": summary.get("avg_overhead_ratio", 0),
                        "convergence_rate": summary.get("convergence_rate", 0),
                    }
                )

        if not summary_data:
            ax.text(0.5, 0.5, "No summary statistics", ha="center", va="center")
            ax.set_title("Summary Statistics")
            return

        df_summary = pd.DataFrame(summary_data)

        # Create dual axis plot
        ax2 = ax.twinx()

        x_pos = range(len(df_summary))
        ax.bar(
            [x - 0.2 for x in x_pos],
            df_summary["avg_overhead_ratio"],
            width=0.4,
            alpha=0.8,
            label="Overhead Ratio",
            color="red",
        )
        ax2.bar(
            [x + 0.2 for x in x_pos],
            df_summary["convergence_rate"],
            width=0.4,
            alpha=0.8,
            label="Convergence Rate",
            color="blue",
        )

        ax.set_xlabel("Configuration")
        ax.set_ylabel("Overhead Ratio", color="red")
        ax2.set_ylabel("Convergence Rate", color="blue")
        ax.set_title("Performance Summary")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_summary["config"], rotation=45)

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        ax.grid(True, alpha=0.3)

    def _generate_detailed_plots(
        self, df_rounds: pd.DataFrame, df_clients: pd.DataFrame, results: Dict[str, Any]
    ):
        """Generate additional detailed plots"""

        # 1. Detailed ZKP overhead analysis
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        if not df_rounds.empty:
            zkp_data = df_rounds[df_rounds["enable_zkp"] == True]
            no_zkp_data = df_rounds[df_rounds["enable_zkp"] == False]

            if not zkp_data.empty and not no_zkp_data.empty:
                x = zkp_data.groupby("num_clients")["total_proof_overhead"].mean()
                y = no_zkp_data.groupby("num_clients")["avg_training_time"].mean()

                plt.bar(
                    x.index - 0.2,
                    x.values,
                    width=0.4,
                    label="With ZKP (Total)",
                    alpha=0.8,
                )
                plt.bar(
                    y.index + 0.2, y.values, width=0.4, label="Without ZKP", alpha=0.8
                )

            plt.xlabel("Number of Clients")
            plt.ylabel("Time (seconds)")
            plt.title("ZKP Overhead Analysis")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        # Proof generation efficiency
        if not df_clients.empty:
            zkp_clients = df_clients[df_clients["enable_zkp"] == True]
            if not zkp_clients.empty and "proof_generation_time" in zkp_clients.columns:
                efficiency = (
                    zkp_clients.groupby(["num_clients", "round"])
                    .agg(
                        {"proof_generation_time": "mean", "pure_training_time": "mean"}
                    )
                    .reset_index()
                )

                efficiency["efficiency"] = (
                    efficiency["pure_training_time"]
                    / efficiency["proof_generation_time"]
                )

                for num_clients in efficiency["num_clients"].unique():
                    client_data = efficiency[efficiency["num_clients"] == num_clients]
                    plt.plot(
                        client_data["round"],
                        client_data["efficiency"],
                        "o-",
                        label=f"{num_clients} clients",
                    )

                plt.xlabel("Round")
                plt.ylabel("Training/Proof Ratio")
                plt.title("Proof Generation Efficiency")
                plt.legend()
                plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        # Server performance trends
        if not df_rounds.empty:
            server_perf = (
                df_rounds.groupby(["enable_zkp", "round"])
                .agg({"aggregation_time": "mean", "server_proof_time": "mean"})
                .reset_index()
            )

            zkp_server = server_perf[server_perf["enable_zkp"] == True]
            no_zkp_server = server_perf[server_perf["enable_zkp"] == False]

            if not zkp_server.empty:
                plt.plot(
                    zkp_server["round"],
                    zkp_server["aggregation_time"],
                    "o-",
                    label="Aggregation (ZKP)",
                    linewidth=2,
                )
                plt.plot(
                    zkp_server["round"],
                    zkp_server["server_proof_time"],
                    "s--",
                    label="Server Proof",
                    linewidth=2,
                )

            if not no_zkp_server.empty:
                plt.plot(
                    no_zkp_server["round"],
                    no_zkp_server["aggregation_time"],
                    "^:",
                    label="Aggregation (No ZKP)",
                    linewidth=2,
                )

            plt.xlabel("Round")
            plt.ylabel("Time (seconds)")
            plt.title("Server Performance Trends")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        # Cost analysis
        if not df_rounds.empty:
            cost_analysis = (
                df_rounds.groupby(["enable_zkp", "num_clients"])
                .agg({"total_proof_overhead": "sum", "avg_training_time": "sum"})
                .reset_index()
            )

            cost_analysis["total_cost_zkp"] = (
                cost_analysis["total_proof_overhead"]
                + cost_analysis["avg_training_time"]
            )

            zkp_cost = cost_analysis[cost_analysis["enable_zkp"] == True]
            no_zkp_cost = cost_analysis[cost_analysis["enable_zkp"] == False]

            if not zkp_cost.empty:
                plt.bar(
                    zkp_cost["num_clients"] - 0.2,
                    zkp_cost["total_cost_zkp"],
                    width=0.4,
                    label="With ZKP",
                    alpha=0.8,
                )

            if not no_zkp_cost.empty:
                plt.bar(
                    no_zkp_cost["num_clients"] + 0.2,
                    no_zkp_cost["avg_training_time"],
                    width=0.4,
                    label="Without ZKP",
                    alpha=0.8,
                )

            plt.xlabel("Number of Clients")
            plt.ylabel("Total Time (seconds)")
            plt.title("Total Training Cost Analysis")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "detailed_performance_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Generate summary report
        self._generate_summary_report(results)

    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate comprehensive text summary report"""
        report_file = self.output_dir / "performance_summary_report.txt"

        with open(report_file, "w") as f:
            f.write("=== SECURE FL PERFORMANCE BENCHMARK REPORT ===\n\n")

            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Configurations Tested: {len(results)}\n\n")

            for config_name, config_results in results.items():
                f.write(f"\n--- Configuration: {config_name} ---\n")

                config_info = config_results.get("config", {})
                f.write(f"Dataset: {config_info.get('dataset', 'N/A')}\n")
                f.write(f"Clients: {config_info.get('num_clients', 'N/A')}\n")
                f.write(f"Rounds: {config_info.get('num_rounds', 'N/A')}\n")
                f.write(f"ZKP Enabled: {config_info.get('enable_zkp', 'N/A')}\n")
                f.write(f"Proof Rigor: {config_info.get('proof_rigor', 'N/A')}\n\n")

                summary = config_results.get("summary", {})
                if summary:
                    f.write("Performance Summary:\n")
                    f.write(
                        f"  Average Aggregation Time: {summary.get('avg_aggregation_time', 0):.4f} ± {summary.get('std_aggregation_time', 0):.4f} seconds\n"
                    )
                    f.write(
                        f"  Average Server Proof Time: {summary.get('avg_server_proof_time', 0):.4f} ± {summary.get('std_server_proof_time', 0):.4f} seconds\n"
                    )
                    f.write(
                        f"  Average Client Proof Time: {summary.get('avg_client_proof_time', 0):.4f} ± {summary.get('std_client_proof_time', 0):.4f} seconds\n"
                    )
                    f.write(
                        f"  Average Training Time: {summary.get('avg_training_time', 0):.4f} ± {summary.get('std_training_time', 0):.4f} seconds\n"
                    )
                    f.write(
                        f"  Average Overhead Ratio: {summary.get('avg_overhead_ratio', 0):.4f} ± {summary.get('std_overhead_ratio', 0):.4f}\n"
                    )
                    f.write(
                        f"  Maximum Overhead Ratio: {summary.get('max_overhead_ratio', 0):.4f}\n"
                    )
                    f.write(
                        f"  Convergence Rate: {summary.get('convergence_rate', 0):.6f}\n\n"
                    )

        logger.info(f"📝 Summary report saved to {report_file}")


def main():
    """Main function to run comprehensive benchmark"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run comprehensive Secure FL performance benchmark"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--clients",
        type=int,
        nargs="+",
        default=[3, 5, 10],
        help="Number of clients to test",
    )
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=["mnist"], help="Datasets to test"
    )
    parser.add_argument(
        "--no-zkp", action="store_true", help="Skip ZKP testing (only baseline)"
    )
    parser.add_argument(
        "--rigor-levels",
        type=str,
        nargs="+",
        default=["low", "medium", "high"],
        help="ZKP rigor levels to test",
    )

    args = parser.parse_args()

    # Create output directory first
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{args.output_dir}/benchmark.log"),
            logging.StreamHandler(),
        ],
    )

    # Create benchmark runner
    benchmark = SecureFlBenchmark(output_dir=args.output_dir)

    # Configure ZKP settings
    zkp_configs = [False] if args.no_zkp else [False, True]

    # Run benchmark
    results = benchmark.run_comprehensive_benchmark(
        datasets=args.datasets,
        num_clients_list=args.clients,
        num_rounds=args.rounds,
        zkp_configs=zkp_configs,
        rigor_levels=args.rigor_levels,
    )

    print(f"\n✅ Benchmark completed successfully!")
    print(f"📊 Results and plots saved to: {args.output_dir}")
    print(f"📈 Generated {len([f for f in Path(args.output_dir).glob('*.png')])} plots")
    print(f"📝 Summary report: {args.output_dir}/performance_summary_report.txt")


if __name__ == "__main__":
    main()
