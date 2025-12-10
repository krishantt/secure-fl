"""
Simple Performance Benchmark for Secure FL

This is a simplified version to identify and fix issues with the comprehensive benchmark.
It measures key performance metrics without complex configurations.
"""

import json
import logging

# Add parent directory to path
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from secure_fl.aggregation import FedJSCMAggregator
from secure_fl.client import SecureFlowerClient
from secure_fl.data.dataloader import FederatedDataLoader
from secure_fl.models import MNISTModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleBenchmark:
    """Simple benchmark for Secure FL performance measurement"""

    def __init__(self, benchmark_name: str = "simple_benchmark"):
        self.output_dir = Path("results") / benchmark_name
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def prepare_data(self, num_clients: int = 3, samples_per_client: int = 1000):
        """Prepare MNIST data for federated learning"""
        logger.info(
            f"Preparing data for {num_clients} clients with {samples_per_client} samples each"
        )

        # Use centralized federated data loader
        fed_loader = FederatedDataLoader(
            dataset_name="mnist",
            num_clients=num_clients,
            iid=True,
            val_split=0.2,  # Use 20% for validation
            batch_size=32,
        )

        # Get client datasets (returns list of (train_dataset, val_dataset) tuples)
        client_datasets = fed_loader.create_client_datasets()

        logger.info(f"✅ Data prepared for {len(client_datasets)} clients")
        return client_datasets

    def create_clients(self, client_datasets, enable_zkp: bool = False):
        """Create FL clients"""
        clients = []

        for i, (train_data, test_data) in enumerate(client_datasets):
            model = MNISTModel(hidden_dims=[128, 64], output_dim=10)

            train_loader = DataLoader(
                train_data, batch_size=32, shuffle=True, drop_last=True
            )
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

            client = SecureFlowerClient(
                client_id=f"client_{i}",
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                enable_zkp=enable_zkp,
                proof_rigor="medium" if enable_zkp else "none",
                local_epochs=2,
                learning_rate=0.01,
                quantize_weights=False,  # Disable to avoid issues
            )

            clients.append(client)

        logger.info(f"✅ Created {len(clients)} clients (ZKP={enable_zkp})")
        return clients

    def run_benchmark(
        self, num_clients: int = 3, num_rounds: int = 5, test_zkp: bool = True
    ):
        """Run simple benchmark comparing with/without ZKP"""
        logger.info(
            f"🚀 Starting simple benchmark: {num_clients} clients, {num_rounds} rounds"
        )

        # Prepare data
        client_datasets = self.prepare_data(num_clients, samples_per_client=500)

        results = {
            "config": {
                "num_clients": num_clients,
                "num_rounds": num_rounds,
                "samples_per_client": 500,
            },
            "without_zkp": None,
            "with_zkp": None,
        }

        # Test without ZKP
        logger.info("📊 Running benchmark WITHOUT ZKP...")
        results["without_zkp"] = self._run_single_config(
            client_datasets, num_rounds, enable_zkp=False
        )

        # Test with ZKP (if enabled)
        if test_zkp:
            logger.info("📊 Running benchmark WITH ZKP...")
            results["with_zkp"] = self._run_single_config(
                client_datasets, num_rounds, enable_zkp=True
            )

        # Save results
        self._save_results(results)

        # Generate plots
        self._generate_plots(results)

        logger.info(f"✅ Benchmark completed! Results saved to {self.output_dir}")
        return results

    def _run_single_config(self, client_datasets, num_rounds: int, enable_zkp: bool):
        """Run benchmark for single configuration"""
        config_name = "WITH ZKP" if enable_zkp else "WITHOUT ZKP"

        # Create clients
        clients = self.create_clients(client_datasets, enable_zkp=enable_zkp)

        # Initialize aggregator
        aggregator = FedJSCMAggregator(momentum=0.9, learning_rate=0.01)

        # Initialize global model
        global_model = MNISTModel(hidden_dims=[128, 64], output_dim=10)
        global_params = [
            param.detach().cpu().numpy() for param in global_model.parameters()
        ]

        # Results storage
        round_results = []
        client_results = []

        logger.info(f"Running FL simulation for {num_rounds} rounds ({config_name})...")

        total_start_time = time.perf_counter()

        for round_num in range(num_rounds):
            logger.info(f"  Round {round_num + 1}/{num_rounds}")

            round_start_time = time.perf_counter()

            # Client training phase
            client_updates = []
            client_weights = []
            round_client_metrics = []

            total_client_time = 0
            total_proof_time = 0

            for client_idx, client in enumerate(clients):
                # Set global parameters
                client.set_parameters(global_params)

                # Training
                config = {"server_round": round_num + 1}

                client_start = time.perf_counter()
                updated_params, num_examples, metrics = client.fit(
                    global_params, config
                )
                client_end = time.perf_counter()

                client_time = client_end - client_start
                proof_time = metrics.get("proof_time", 0.0)
                training_time = metrics.get("training_time", client_time - proof_time)

                total_client_time += client_time
                total_proof_time += proof_time

                # Store client metrics
                client_metric = {
                    "round": round_num + 1,
                    "client_id": client_idx,
                    "total_time": client_time,
                    "training_time": training_time,
                    "proof_time": proof_time,
                    "num_examples": num_examples,
                    "train_loss": metrics.get("train_loss", 0.0),
                    "train_accuracy": metrics.get("train_accuracy", 0.0),
                    "zkp_overhead": proof_time / training_time
                    if training_time > 0
                    else 0,
                }

                round_client_metrics.append(client_metric)
                client_updates.append(updated_params)
                client_weights.append(num_examples)

            # Server aggregation phase
            aggregation_start = time.perf_counter()

            # Normalize weights
            total_examples = sum(client_weights)
            normalized_weights = [w / total_examples for w in client_weights]

            # Aggregate
            aggregated_params = aggregator.aggregate(
                client_updates=client_updates,
                client_weights=normalized_weights,
                server_round=round_num + 1,
                global_params=global_params,
            )

            aggregation_end = time.perf_counter()
            aggregation_time = aggregation_end - aggregation_start

            # Update global parameters
            global_params = aggregated_params

            round_end_time = time.perf_counter()
            round_total_time = round_end_time - round_start_time

            # Store round metrics
            round_metric = {
                "round": round_num + 1,
                "round_total_time": round_total_time,
                "total_client_time": total_client_time,
                "aggregation_time": aggregation_time,
                "total_proof_time": total_proof_time,
                "avg_client_time": total_client_time / len(clients),
                "avg_proof_time": total_proof_time / len(clients),
                "avg_training_time": np.mean(
                    [m["training_time"] for m in round_client_metrics]
                ),
                "proof_overhead_ratio": total_proof_time
                / np.sum([m["training_time"] for m in round_client_metrics])
                if enable_zkp
                else 0,
            }

            round_results.append(round_metric)
            client_results.extend(round_client_metrics)

            logger.info(
                f"    Round time: {round_total_time:.2f}s, Proof overhead: {total_proof_time:.2f}s"
            )

        total_end_time = time.perf_counter()
        total_experiment_time = total_end_time - total_start_time

        # Calculate summary statistics
        summary = {
            "total_experiment_time": total_experiment_time,
            "avg_round_time": np.mean([r["round_total_time"] for r in round_results]),
            "avg_aggregation_time": np.mean(
                [r["aggregation_time"] for r in round_results]
            ),
            "avg_client_time": np.mean([c["total_time"] for c in client_results]),
            "avg_training_time": np.mean([c["training_time"] for c in client_results]),
            "avg_proof_time": np.mean([c["proof_time"] for c in client_results]),
            "total_proof_overhead": sum([r["total_proof_time"] for r in round_results]),
            "avg_proof_overhead_ratio": np.mean(
                [r["proof_overhead_ratio"] for r in round_results]
            ),
            "enable_zkp": enable_zkp,
        }

        return {
            "summary": summary,
            "round_metrics": round_results,
            "client_metrics": client_results,
        }

    def _save_results(self, results: dict[str, Any]):
        """Save benchmark results"""
        output_file = self.output_dir / "benchmark_results.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"📄 Results saved to {output_file}")

    def _generate_plots(self, results: dict[str, Any]):
        """Generate performance comparison plots"""
        logger.info("📈 Generating performance plots...")

        # Extract data for plotting
        without_zkp = results.get("without_zkp")
        with_zkp = results.get("with_zkp")

        if not without_zkp:
            logger.warning("No baseline (without ZKP) results to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Secure FL Performance Analysis", fontsize=16, fontweight="bold")

        # 1. Round Time Comparison
        ax1 = axes[0, 0]
        rounds = [r["round"] for r in without_zkp["round_metrics"]]
        no_zkp_times = [r["round_total_time"] for r in without_zkp["round_metrics"]]

        ax1.plot(
            rounds, no_zkp_times, "o-", label="Without ZKP", linewidth=2, markersize=6
        )

        if with_zkp:
            zkp_times = [r["round_total_time"] for r in with_zkp["round_metrics"]]
            ax1.plot(
                rounds, zkp_times, "s--", label="With ZKP", linewidth=2, markersize=6
            )

        ax1.set_xlabel("Round")
        ax1.set_ylabel("Round Time (seconds)")
        ax1.set_title("Round Time Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Training vs Proof Time
        ax2 = axes[0, 1]

        # Aggregate data by round
        no_zkp_train = []
        no_zkp_proof = []
        zkp_train = []
        zkp_proof = []

        for round_num in rounds:
            # Without ZKP
            round_clients_no_zkp = [
                c for c in without_zkp["client_metrics"] if c["round"] == round_num
            ]
            no_zkp_train.append(
                np.mean([c["training_time"] for c in round_clients_no_zkp])
            )
            no_zkp_proof.append(
                np.mean([c["proof_time"] for c in round_clients_no_zkp])
            )

            # With ZKP
            if with_zkp:
                round_clients_zkp = [
                    c for c in with_zkp["client_metrics"] if c["round"] == round_num
                ]
                zkp_train.append(
                    np.mean([c["training_time"] for c in round_clients_zkp])
                )
                zkp_proof.append(np.mean([c["proof_time"] for c in round_clients_zkp]))

        x_pos = np.arange(len(rounds))
        width = 0.35

        ax2.bar(
            x_pos - width / 2, no_zkp_train, width, label="Training (No ZKP)", alpha=0.8
        )
        ax2.bar(
            x_pos - width / 2,
            no_zkp_proof,
            width,
            bottom=no_zkp_train,
            label="Proof (No ZKP)",
            alpha=0.8,
        )

        if with_zkp:
            ax2.bar(
                x_pos + width / 2, zkp_train, width, label="Training (ZKP)", alpha=0.8
            )
            ax2.bar(
                x_pos + width / 2,
                zkp_proof,
                width,
                bottom=zkp_train,
                label="Proof (ZKP)",
                alpha=0.8,
            )

        ax2.set_xlabel("Round")
        ax2.set_ylabel("Time (seconds)")
        ax2.set_title("Training vs Proof Time Breakdown")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"R{r}" for r in rounds])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Overhead Ratio
        ax3 = axes[0, 2]

        if with_zkp:
            overhead_ratios = [
                r["proof_overhead_ratio"] for r in with_zkp["round_metrics"]
            ]
            ax3.bar(rounds, overhead_ratios, alpha=0.8, color="red")
            ax3.set_xlabel("Round")
            ax3.set_ylabel("Proof Overhead Ratio")
            ax3.set_title("ZKP Overhead Ratio (Proof Time / Training Time)")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(
                0.5,
                0.5,
                "ZKP not tested",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_title("ZKP Overhead Ratio")

        # 4. Aggregation Time
        ax4 = axes[1, 0]

        no_zkp_agg = [r["aggregation_time"] for r in without_zkp["round_metrics"]]
        ax4.plot(
            rounds, no_zkp_agg, "o-", label="Without ZKP", linewidth=2, markersize=6
        )

        if with_zkp:
            zkp_agg = [r["aggregation_time"] for r in with_zkp["round_metrics"]]
            ax4.plot(
                rounds, zkp_agg, "s--", label="With ZKP", linewidth=2, markersize=6
            )

        ax4.set_xlabel("Round")
        ax4.set_ylabel("Aggregation Time (seconds)")
        ax4.set_title("Server Aggregation Time")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Client Performance Distribution
        ax5 = axes[1, 1]

        all_client_times_no_zkp = [
            c["total_time"] for c in without_zkp["client_metrics"]
        ]

        ax5.hist(
            all_client_times_no_zkp,
            bins=10,
            alpha=0.7,
            label="Without ZKP",
            density=True,
        )

        if with_zkp:
            all_client_times_zkp = [c["total_time"] for c in with_zkp["client_metrics"]]
            ax5.hist(
                all_client_times_zkp, bins=10, alpha=0.7, label="With ZKP", density=True
            )

        ax5.set_xlabel("Client Total Time (seconds)")
        ax5.set_ylabel("Density")
        ax5.set_title("Client Performance Distribution")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Summary Statistics
        ax6 = axes[1, 2]

        categories = ["Avg Round\nTime", "Avg Training\nTime", "Avg Aggregation\nTime"]
        no_zkp_values = [
            without_zkp["summary"]["avg_round_time"],
            without_zkp["summary"]["avg_training_time"],
            without_zkp["summary"]["avg_aggregation_time"],
        ]

        x_pos = np.arange(len(categories))
        ax6.bar(x_pos - 0.2, no_zkp_values, 0.4, label="Without ZKP", alpha=0.8)

        if with_zkp:
            zkp_values = [
                with_zkp["summary"]["avg_round_time"],
                with_zkp["summary"]["avg_training_time"],
                with_zkp["summary"]["avg_aggregation_time"],
            ]
            ax6.bar(x_pos + 0.2, zkp_values, 0.4, label="With ZKP", alpha=0.8)

        ax6.set_xlabel("Metric")
        ax6.set_ylabel("Time (seconds)")
        ax6.set_title("Average Performance Summary")
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(categories)
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / "performance_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"📊 Performance plot saved to {plot_file}")

        # Generate summary report
        self._generate_report(results)

    def _generate_report(self, results: dict[str, Any]):
        """Generate text summary report"""
        report_file = self.output_dir / "performance_report.txt"

        with open(report_file, "w") as f:
            f.write("SECURE FL PERFORMANCE BENCHMARK REPORT\n")
            f.write("=" * 50 + "\n\n")

            config = results["config"]
            f.write("Configuration:\n")
            f.write(f"  - Number of clients: {config['num_clients']}\n")
            f.write(f"  - Number of rounds: {config['num_rounds']}\n")
            f.write(f"  - Samples per client: {config['samples_per_client']}\n\n")

            # Without ZKP results
            if results["without_zkp"]:
                f.write("RESULTS WITHOUT ZKP (Baseline):\n")
                f.write("-" * 30 + "\n")
                summary = results["without_zkp"]["summary"]
                f.write(
                    f"  Total experiment time: {summary['total_experiment_time']:.2f} seconds\n"
                )
                f.write(
                    f"  Average round time: {summary['avg_round_time']:.3f} ± {np.std([r['round_total_time'] for r in results['without_zkp']['round_metrics']]):.3f} seconds\n"
                )
                f.write(
                    f"  Average client time: {summary['avg_client_time']:.3f} seconds\n"
                )
                f.write(
                    f"  Average training time: {summary['avg_training_time']:.3f} seconds\n"
                )
                f.write(
                    f"  Average aggregation time: {summary['avg_aggregation_time']:.3f} seconds\n\n"
                )

            # With ZKP results
            if results["with_zkp"]:
                f.write("RESULTS WITH ZKP:\n")
                f.write("-" * 20 + "\n")
                summary = results["with_zkp"]["summary"]
                f.write(
                    f"  Total experiment time: {summary['total_experiment_time']:.2f} seconds\n"
                )
                f.write(
                    f"  Average round time: {summary['avg_round_time']:.3f} ± {np.std([r['round_total_time'] for r in results['with_zkp']['round_metrics']]):.3f} seconds\n"
                )
                f.write(
                    f"  Average client time: {summary['avg_client_time']:.3f} seconds\n"
                )
                f.write(
                    f"  Average training time: {summary['avg_training_time']:.3f} seconds\n"
                )
                f.write(
                    f"  Average proof time: {summary['avg_proof_time']:.3f} seconds\n"
                )
                f.write(
                    f"  Average aggregation time: {summary['avg_aggregation_time']:.3f} seconds\n"
                )
                f.write(
                    f"  Total proof overhead: {summary['total_proof_overhead']:.2f} seconds\n"
                )
                f.write(
                    f"  Average overhead ratio: {summary['avg_proof_overhead_ratio']:.3f}\n\n"
                )

                # Calculate overhead comparison
                if results["without_zkp"]:
                    baseline_time = results["without_zkp"]["summary"][
                        "total_experiment_time"
                    ]
                    zkp_time = results["with_zkp"]["summary"]["total_experiment_time"]
                    overhead_percentage = (
                        (zkp_time - baseline_time) / baseline_time
                    ) * 100

                    f.write("OVERHEAD ANALYSIS:\n")
                    f.write("-" * 20 + "\n")
                    f.write(
                        f"  Total time increase: {zkp_time - baseline_time:.2f} seconds\n"
                    )
                    f.write(f"  Overhead percentage: {overhead_percentage:.1f}%\n")
                    f.write(
                        f"  Proof time as % of total: {(summary['total_proof_overhead'] / zkp_time) * 100:.1f}%\n\n"
                    )

            f.write(f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        logger.info(f"📝 Performance report saved to {report_file}")


def main():
    """Main function to run simple benchmark"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run simple Secure FL performance benchmark"
    )
    parser.add_argument("--clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default="simple_benchmark",
        help="Benchmark name for results directory",
    )
    parser.add_argument(
        "--no-zkp", action="store_true", help="Skip ZKP testing (baseline only)"
    )

    args = parser.parse_args()

    # Create benchmark runner
    benchmark = SimpleBenchmark(benchmark_name=args.benchmark_name)

    # Run benchmark
    results = benchmark.run_benchmark(
        num_clients=args.clients, num_rounds=args.rounds, test_zkp=not args.no_zkp
    )

    print("\n✅ Simple benchmark completed!")
    print(f"📊 Results saved to: {benchmark.output_dir}")
    print(f"📈 Performance plot: {benchmark.output_dir}/performance_analysis.png")
    print(f"📝 Summary report: {benchmark.output_dir}/performance_report.txt")


if __name__ == "__main__":
    main()
