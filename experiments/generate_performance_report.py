"""
Generate Comprehensive Performance Report for Secure FL

This script runs benchmarks and generates detailed performance analysis
including plots, tables, and report for the research paper.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class PerformanceReportGenerator:
    """Generate comprehensive performance analysis report"""

    def __init__(self, benchmark_name: str = "performance_analysis"):
        self.output_dir = Path("results") / benchmark_name
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = {}

    def run_all_benchmarks(self):
        """Run all performance benchmarks and collect results"""
        print("🚀 Starting Comprehensive Performance Analysis...")

        # Define benchmark suites
        benchmark_suites = [
            {
                "name": "proof_generation_scaling",
                "description": "Client Proof Generation Scaling",
                "test_pattern": "TestProofManagerBenchmarks::test_client_proof_generation_scaling",
                "timeout": 300,
            },
            {
                "name": "training_comparison",
                "description": "Training With/Without ZKP",
                "test_pattern": "TestClientTrainingBenchmarks::test_training_without_zkp TestClientTrainingBenchmarks::test_training_with_zkp",
                "timeout": 600,
            },
            {
                "name": "model_scaling",
                "description": "Model Size Scaling",
                "test_pattern": "TestClientTrainingBenchmarks::test_training_model_scaling",
                "timeout": 300,
            },
            {
                "name": "aggregation_scaling",
                "description": "Server Aggregation Scaling",
                "test_pattern": "TestServerAggregationBenchmarks::test_aggregation_scaling",
                "timeout": 180,
            },
            {
                "name": "end_to_end",
                "description": "End-to-End FL Rounds",
                "test_pattern": "TestEndToEndBenchmarks::test_complete_fl_round_without_zkp TestEndToEndBenchmarks::test_complete_fl_round_with_zkp",
                "timeout": 900,
            },
        ]

        # Run each benchmark suite
        for suite in benchmark_suites:
            print(f"\n📊 Running {suite['description']}...")
            result_file = self.output_dir / f"{suite['name']}.json"

            cmd = [
                sys.executable,
                "-m",
                "pytest",
                f"experiments/test_performance_benchmarks.py::{suite['test_pattern']}",
                "--benchmark-only",
                "--benchmark-json",
                str(result_file),
                "--benchmark-sort=mean",
                "-v",
            ]

            try:
                # Replace :: with space for proper pytest syntax
                cmd_str = " ".join(cmd).replace("::", " -k ")
                if " -k " not in cmd_str:
                    # Single test pattern
                    cmd = [
                        sys.executable,
                        "-m",
                        "pytest",
                        f"experiments/test_performance_benchmarks.py::{suite['test_pattern']}",
                        "--benchmark-only",
                        "--benchmark-json",
                        str(result_file),
                        "--benchmark-sort=mean",
                        "-v",
                    ]
                else:
                    # Multiple test patterns - use -k
                    test_names = suite["test_pattern"].replace("::", "")
                    cmd = [
                        sys.executable,
                        "-m",
                        "pytest",
                        "experiments/test_performance_benchmarks.py",
                        "-k",
                        test_names,
                        "--benchmark-only",
                        "--benchmark-json",
                        str(result_file),
                        "--benchmark-sort=mean",
                        "-v",
                    ]

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=suite["timeout"]
                )

                if result.returncode == 0:
                    print(f"✅ {suite['description']} completed")
                    if result_file.exists():
                        with open(result_file) as f:
                            self.results[suite["name"]] = json.load(f)
                else:
                    print(f"❌ {suite['description']} failed: {result.stderr}")

            except subprocess.TimeoutExpired:
                print(f"⏰ {suite['description']} timed out")
            except Exception as e:
                print(f"❌ {suite['description']} error: {e}")

        print(
            f"\n✅ Benchmark collection completed. Results: {list(self.results.keys())}"
        )

    def generate_plots(self):
        """Generate comprehensive performance plots"""
        print("\n📈 Generating performance plots...")

        # Create main analysis figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Secure FL Performance Analysis", fontsize=16, fontweight="bold")

        # Plot 1: Proof Generation Scaling
        self._plot_proof_scaling(axes[0, 0])

        # Plot 2: Training Time Comparison
        self._plot_training_comparison(axes[0, 1])

        # Plot 3: Model Size Scaling
        self._plot_model_scaling(axes[0, 2])

        # Plot 4: Aggregation Scaling
        self._plot_aggregation_scaling(axes[1, 0])

        # Plot 5: End-to-End Comparison
        self._plot_end_to_end(axes[1, 1])

        # Plot 6: Overhead Summary
        self._plot_overhead_summary(axes[1, 2])

        plt.tight_layout()
        plot_file = self.output_dir / "comprehensive_performance_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📊 Main analysis plot saved: {plot_file}")

        # Generate individual detailed plots
        self._generate_detailed_plots()

    def _plot_proof_scaling(self, ax):
        """Plot proof generation scaling"""
        if "proof_generation_scaling" not in self.results:
            ax.text(0.5, 0.5, "No proof scaling data", ha="center", va="center")
            ax.set_title("Proof Generation Scaling")
            return

        data = self._extract_benchmark_data("proof_generation_scaling")
        if not data:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.set_title("Proof Generation Scaling")
            return

        # Extract parameter sizes and times
        param_sizes = []
        mean_times = []
        std_times = []

        for bench in data:
            if "param_size" in str(bench.get("name", "")):
                # Extract parameter size from name
                name = bench["name"]
                if "[" in name and "]" in name:
                    size_str = name.split("[")[1].split("]")[0]
                    try:
                        param_size = int(size_str)
                        param_sizes.append(param_size)

                        stats = bench.get("stats", {})
                        mean_times.append(stats.get("mean", 0) * 1000)  # Convert to ms
                        std_times.append(stats.get("stddev", 0) * 1000)
                    except ValueError:
                        continue

        if param_sizes and mean_times:
            # Sort by parameter size
            sorted_data = sorted(zip(param_sizes, mean_times, std_times))
            param_sizes, mean_times, std_times = zip(*sorted_data)

            ax.errorbar(
                param_sizes,
                mean_times,
                yerr=std_times,
                marker="o",
                capsize=5,
                capthick=2,
                linewidth=2,
            )
            ax.set_xlabel("Parameter Size")
            ax.set_ylabel("Proof Generation Time (ms)")
            ax.set_title("ZKP Proof Generation Scaling")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Unable to parse data", ha="center", va="center")
            ax.set_title("Proof Generation Scaling")

    def _plot_training_comparison(self, ax):
        """Plot training time comparison with/without ZKP"""
        if "training_comparison" not in self.results:
            ax.text(0.5, 0.5, "No training comparison data", ha="center", va="center")
            ax.set_title("Training Time Comparison")
            return

        data = self._extract_benchmark_data("training_comparison")
        if not data:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.set_title("Training Time Comparison")
            return

        zkp_time = None
        no_zkp_time = None

        for bench in data:
            name = bench.get("name", "")
            stats = bench.get("stats", {})
            mean_time = stats.get("mean", 0) * 1000  # Convert to ms

            if "without_zkp" in name:
                no_zkp_time = mean_time
            elif "with_zkp" in name:
                zkp_time = mean_time

        if zkp_time is not None and no_zkp_time is not None:
            categories = ["Without ZKP", "With ZKP"]
            times = [no_zkp_time, zkp_time]
            colors = ["#2E86AB", "#A23B72"]

            bars = ax.bar(categories, times, color=colors, alpha=0.8)
            ax.set_ylabel("Training Time (ms)")
            ax.set_title("Training Time: ZKP vs Baseline")
            ax.set_yscale("log")

            # Add value labels on bars
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{time:.1f}ms",
                    ha="center",
                    va="bottom",
                )

            # Add overhead ratio
            overhead = zkp_time / no_zkp_time
            ax.text(
                0.5,
                0.8,
                f"Overhead: {overhead:.0f}x",
                transform=ax.transAxes,
                ha="center",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
            )
        else:
            ax.text(0.5, 0.5, "Incomplete comparison data", ha="center", va="center")
            ax.set_title("Training Time Comparison")

        ax.grid(True, alpha=0.3)

    def _plot_model_scaling(self, ax):
        """Plot model size scaling"""
        if "model_scaling" not in self.results:
            ax.text(0.5, 0.5, "No model scaling data", ha="center", va="center")
            ax.set_title("Model Size Scaling")
            return

        data = self._extract_benchmark_data("model_scaling")
        if not data:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.set_title("Model Size Scaling")
            return

        model_sizes = []
        mean_times = []

        size_order = ["small", "medium", "large"]

        for size in size_order:
            for bench in data:
                name = bench.get("name", "")
                if f"[{size}]" in name:
                    stats = bench.get("stats", {})
                    mean_time = stats.get("mean", 0) * 1000  # Convert to ms
                    model_sizes.append(size.capitalize())
                    mean_times.append(mean_time)
                    break

        if model_sizes and mean_times:
            bars = ax.bar(model_sizes, mean_times, alpha=0.8)
            ax.set_ylabel("Training Time (ms)")
            ax.set_title("Training Time vs Model Size")

            # Add value labels
            for bar, time in zip(bars, mean_times):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{time:.1f}ms",
                    ha="center",
                    va="bottom",
                )
        else:
            ax.text(0.5, 0.5, "Unable to parse model data", ha="center", va="center")
            ax.set_title("Model Size Scaling")

        ax.grid(True, alpha=0.3)

    def _plot_aggregation_scaling(self, ax):
        """Plot server aggregation scaling"""
        if "aggregation_scaling" not in self.results:
            ax.text(0.5, 0.5, "No aggregation scaling data", ha="center", va="center")
            ax.set_title("Server Aggregation Scaling")
            return

        data = self._extract_benchmark_data("aggregation_scaling")
        if not data:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.set_title("Server Aggregation Scaling")
            return

        num_clients = []
        mean_times = []

        for bench in data:
            name = bench.get("name", "")
            if "[" in name and "]" in name:
                client_str = name.split("[")[1].split("]")[0]
                try:
                    clients = int(client_str)
                    num_clients.append(clients)

                    stats = bench.get("stats", {})
                    mean_time = stats.get("mean", 0) * 1000  # Convert to ms
                    mean_times.append(mean_time)
                except ValueError:
                    continue

        if num_clients and mean_times:
            # Sort by number of clients
            sorted_data = sorted(zip(num_clients, mean_times))
            num_clients, mean_times = zip(*sorted_data)

            ax.plot(num_clients, mean_times, "o-", linewidth=2, markersize=6)
            ax.set_xlabel("Number of Clients")
            ax.set_ylabel("Aggregation Time (ms)")
            ax.set_title("Server Aggregation Scaling")
        else:
            ax.text(
                0.5, 0.5, "Unable to parse aggregation data", ha="center", va="center"
            )
            ax.set_title("Server Aggregation Scaling")

        ax.grid(True, alpha=0.3)

    def _plot_end_to_end(self, ax):
        """Plot end-to-end FL round comparison"""
        if "end_to_end" not in self.results:
            ax.text(0.5, 0.5, "No end-to-end data", ha="center", va="center")
            ax.set_title("End-to-End FL Round")
            return

        data = self._extract_benchmark_data("end_to_end")
        if not data:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.set_title("End-to-End FL Round")
            return

        zkp_time = None
        no_zkp_time = None

        for bench in data:
            name = bench.get("name", "")
            stats = bench.get("stats", {})
            mean_time = stats.get("mean", 0) * 1000  # Convert to ms

            if "without_zkp" in name:
                no_zkp_time = mean_time
            elif "with_zkp" in name:
                zkp_time = mean_time

        if zkp_time is not None and no_zkp_time is not None:
            categories = ["Without ZKP", "With ZKP"]
            times = [no_zkp_time, zkp_time]
            colors = ["#F18F01", "#C73E1D"]

            bars = ax.bar(categories, times, color=colors, alpha=0.8)
            ax.set_ylabel("FL Round Time (ms)")
            ax.set_title("End-to-End FL Round")
            ax.set_yscale("log")

            # Add value labels
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{time:.0f}ms",
                    ha="center",
                    va="bottom",
                )
        else:
            ax.text(0.5, 0.5, "Incomplete end-to-end data", ha="center", va="center")
            ax.set_title("End-to-End FL Round")

        ax.grid(True, alpha=0.3)

    def _plot_overhead_summary(self, ax):
        """Plot overhead summary"""
        # Calculate overhead ratios from available data
        overheads = {}

        # Training overhead
        if "training_comparison" in self.results:
            data = self._extract_benchmark_data("training_comparison")
            zkp_time = no_zkp_time = None

            for bench in data:
                name = bench.get("name", "")
                stats = bench.get("stats", {})
                mean_time = stats.get("mean", 0)

                if "without_zkp" in name:
                    no_zkp_time = mean_time
                elif "with_zkp" in name:
                    zkp_time = mean_time

            if zkp_time and no_zkp_time:
                overheads["Training"] = zkp_time / no_zkp_time

        # End-to-end overhead
        if "end_to_end" in self.results:
            data = self._extract_benchmark_data("end_to_end")
            zkp_time = no_zkp_time = None

            for bench in data:
                name = bench.get("name", "")
                stats = bench.get("stats", {})
                mean_time = stats.get("mean", 0)

                if "without_zkp" in name:
                    no_zkp_time = mean_time
                elif "with_zkp" in name:
                    zkp_time = mean_time

            if zkp_time and no_zkp_time:
                overheads["End-to-End"] = zkp_time / no_zkp_time

        if overheads:
            categories = list(overheads.keys())
            ratios = list(overheads.values())

            bars = ax.bar(categories, ratios, alpha=0.8, color="red")
            ax.set_ylabel("Overhead Ratio")
            ax.set_title("ZKP Overhead Summary")
            ax.set_yscale("log")

            # Add value labels
            for bar, ratio in zip(bars, ratios):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{ratio:.0f}x",
                    ha="center",
                    va="bottom",
                )
        else:
            ax.text(0.5, 0.5, "No overhead data available", ha="center", va="center")
            ax.set_title("ZKP Overhead Summary")

        ax.grid(True, alpha=0.3)

    def _generate_detailed_plots(self):
        """Generate additional detailed plots"""
        print("📊 Generating detailed plots...")

        # 1. Detailed proof scaling plot
        self._create_proof_scaling_detail()

        # 2. Performance breakdown chart
        self._create_performance_breakdown()

        # 3. Overhead analysis chart
        self._create_overhead_analysis()

    def _create_proof_scaling_detail(self):
        """Create detailed proof scaling analysis"""
        if "proof_generation_scaling" not in self.results:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        data = self._extract_benchmark_data("proof_generation_scaling")
        param_sizes = []
        mean_times = []
        ops_per_sec = []

        for bench in data:
            name = bench.get("name", "")
            if "[" in name and "]" in name:
                size_str = name.split("[")[1].split("]")[0]
                try:
                    param_size = int(size_str)
                    param_sizes.append(param_size)

                    stats = bench.get("stats", {})
                    mean_time = stats.get("mean", 0)
                    mean_times.append(mean_time * 1000)  # ms

                    ops = stats.get("ops", 0)
                    ops_per_sec.append(ops)
                except ValueError:
                    continue

        if param_sizes:
            # Sort data
            sorted_data = sorted(zip(param_sizes, mean_times, ops_per_sec))
            param_sizes, mean_times, ops_per_sec = zip(*sorted_data)

            # Plot 1: Time vs Parameters (log-log)
            ax1.loglog(param_sizes, mean_times, "o-", linewidth=2, markersize=8)
            ax1.set_xlabel("Number of Parameters")
            ax1.set_ylabel("Proof Generation Time (ms)")
            ax1.set_title("ZKP Proof Scaling (Log-Log)")
            ax1.grid(True, alpha=0.3)

            # Fit power law
            if len(param_sizes) > 2:
                log_sizes = np.log10(param_sizes)
                log_times = np.log10(mean_times)
                coeffs = np.polyfit(log_sizes, log_times, 1)

                # Plot fit line
                fit_sizes = np.logspace(
                    np.log10(min(param_sizes)), np.log10(max(param_sizes)), 100
                )
                fit_times = 10 ** (coeffs[0] * np.log10(fit_sizes) + coeffs[1])
                ax1.plot(
                    fit_sizes, fit_times, "--", alpha=0.7, label=f"O(n^{coeffs[0]:.2f})"
                )
                ax1.legend()

            # Plot 2: Throughput
            ax2.semilogx(
                param_sizes,
                ops_per_sec,
                "s-",
                linewidth=2,
                markersize=8,
                color="orange",
            )
            ax2.set_xlabel("Number of Parameters")
            ax2.set_ylabel("Proofs per Second")
            ax2.set_title("ZKP Proof Throughput")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = self.output_dir / "proof_scaling_detailed.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 Detailed proof scaling plot: {plot_file}")

    def _create_performance_breakdown(self):
        """Create performance breakdown chart"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Collect all timing data
        categories = []
        baseline_times = []
        zkp_times = []

        # Training times
        if "training_comparison" in self.results:
            data = self._extract_benchmark_data("training_comparison")
            no_zkp = zkp = None

            for bench in data:
                name = bench.get("name", "")
                stats = bench.get("stats", {})
                mean_time = stats.get("mean", 0) * 1000

                if "without_zkp" in name:
                    no_zkp = mean_time
                elif "with_zkp" in name:
                    zkp = mean_time

            if no_zkp and zkp:
                categories.append("Client Training")
                baseline_times.append(no_zkp)
                zkp_times.append(zkp)

        # End-to-end times
        if "end_to_end" in self.results:
            data = self._extract_benchmark_data("end_to_end")
            no_zkp = zkp = None

            for bench in data:
                name = bench.get("name", "")
                stats = bench.get("stats", {})
                mean_time = stats.get("mean", 0) * 1000

                if "without_zkp" in name:
                    no_zkp = mean_time
                elif "with_zkp" in name:
                    zkp = mean_time

            if no_zkp and zkp:
                categories.append("FL Round")
                baseline_times.append(no_zkp)
                zkp_times.append(zkp)

        if categories:
            x = np.arange(len(categories))
            width = 0.35

            bars1 = ax.bar(
                x - width / 2, baseline_times, width, label="Without ZKP", alpha=0.8
            )
            bars2 = ax.bar(x + width / 2, zkp_times, width, label="With ZKP", alpha=0.8)

            ax.set_xlabel("Operation")
            ax.set_ylabel("Time (ms)")
            ax.set_title("Performance Breakdown: ZKP vs Baseline")
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.0f}",
                        ha="center",
                        va="bottom",
                    )

        plt.tight_layout()
        plot_file = self.output_dir / "performance_breakdown.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📊 Performance breakdown plot: {plot_file}")

    def _create_overhead_analysis(self):
        """Create overhead analysis chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Overhead ratios
        overheads = []
        labels = []

        # Training overhead
        if "training_comparison" in self.results:
            data = self._extract_benchmark_data("training_comparison")
            no_zkp = zkp = None

            for bench in data:
                name = bench.get("name", "")
                stats = bench.get("stats", {})
                mean_time = stats.get("mean", 0)

                if "without_zkp" in name:
                    no_zkp = mean_time
                elif "with_zkp" in name:
                    zkp = mean_time

            if no_zkp and zkp:
                overheads.append(zkp / no_zkp)
                labels.append("Training")

        # End-to-end overhead
        if "end_to_end" in self.results:
            data = self._extract_benchmark_data("end_to_end")
            no_zkp = zkp = None

            for bench in data:
                name = bench.get("name", "")
                stats = bench.get("stats", {})
                mean_time = stats.get("mean", 0)

                if "without_zkp" in name:
                    no_zkp = mean_time
                elif "with_zkp" in name:
                    zkp = mean_time

            if no_zkp and zkp:
                overheads.append(zkp / no_zkp)
                labels.append("FL Round")

        if overheads:
            # Plot 1: Overhead ratios
            colors = plt.cm.viridis(np.linspace(0, 1, len(overheads)))
            bars = ax1.bar(labels, overheads, color=colors, alpha=0.8)
            ax1.set_ylabel("Overhead Ratio")
            ax1.set_title("ZKP Overhead by Operation")
            ax1.set_yscale("log")

            for bar, overhead in zip(bars, overheads):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{overhead:.0f}x",
                    ha="center",
                    va="bottom",
                )

            # Plot 2: Time breakdown pie chart for training
            if "training_comparison" in self.results:
                data = self._extract_benchmark_data("training_comparison")
                no_zkp = zkp = None

                for bench in data:
                    name = bench.get("name", "")
                    stats = bench.get("stats", {})
                    mean_time = stats.get("mean", 0) * 1000

                    if "without_zkp" in name:
                        no_zkp = mean_time
                    elif "with_zkp" in name:
                        zkp = mean_time

                if no_zkp and zkp:
                    proof_time = zkp - no_zkp
                    sizes = [no_zkp, proof_time]
                    pie_labels = ["Training", "Proof Generation"]
                    colors = ["#66c2a5", "#fc8d62"]

                    ax2.pie(sizes, labels=pie_labels, autopct="%1.1f%%", colors=colors)
                    ax2.set_title("Time Distribution (With ZKP)")

        plt.tight_layout()
        plot_file = self.output_dir / "overhead_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 Overhead analysis plot: {plot_file}")

    def _extract_benchmark_data(self, suite_name: str) -> List[Dict]:
        """Extract benchmark data from results"""
        if suite_name not in self.results:
            return []

        result_data = self.results[suite_name]
        benchmarks = result_data.get("benchmarks", [])
        return benchmarks

    def generate_report(self):
        """Generate comprehensive text report"""
        print("\n📝 Generating performance report...")

        report_file = self.output_dir / "performance_analysis_report.md"

        with open(report_file, "w") as f:
            f.write("# Secure FL Performance Analysis Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")

            # Calculate key metrics
            training_overhead = self._calculate_training_overhead()
            if training_overhead:
                f.write(
                    f"- **ZKP Training Overhead:** {training_overhead:.0f}x increase in training time\n"
                )

            end_to_end_overhead = self._calculate_end_to_end_overhead()
            if end_to_end_overhead:
                f.write(
                    f"- **FL Round Overhead:** {end_to_end_overhead:.0f}x increase in FL round time\n"
                )

            proof_scaling = self._analyze_proof_scaling()
            if proof_scaling:
                f.write(
                    f"- **Proof Scaling:** O(n^{proof_scaling:.2f}) with parameter count\n"
                )

            f.write("\n## Detailed Analysis\n\n")

            # Training Performance
            f.write("### Training Performance\n\n")
            self._write_training_analysis(f)

            # Proof Generation
            f.write("### Proof Generation Scaling\n\n")
            self._write_proof_analysis(f)

            # Server Performance
            f.write("### Server Performance\n\n")
            self._write_server_analysis(f)

            # End-to-End Performance
            f.write("### End-to-End FL Performance\n\n")
            self._write_end_to_end_analysis(f)

            # Recommendations
            f.write("## Recommendations\n\n")
            self._write_recommendations(f)

            f.write("## Methodology\n\n")
            f.write(
                "Benchmarks were conducted using pytest-benchmark with the following specifications:\n"
            )
            f.write("- **Hardware:** Apple Silicon Mac\n")
            f.write("- **Python:** 3.12\n")
            f.write("- **Framework:** PyTorch + Flower\n")
            f.write("- **ZKP Library:** PySNARK\n")
            f.write("- **Measurement:** Wall clock time with statistical analysis\n")

        print(f"📄 Performance report saved: {report_file}")

    def _calculate_training_overhead(self) -> float:
        """Calculate training overhead ratio"""
        if "training_comparison" not in self.results:
            return None

        data = self._extract_benchmark_data("training_comparison")
        no_zkp = zkp = None

        for bench in data:
            name = bench.get("name", "")
            stats = bench.get("stats", {})
            mean_time = stats.get("mean", 0)

            if "without_zkp" in name:
                no_zkp = mean_time
            elif "with_zkp" in name:
                zkp = mean_time

        if no_zkp and zkp:
            return zkp / no_zkp
        return None

    def _calculate_end_to_end_overhead(self) -> float:
        """Calculate end-to-end overhead ratio"""
        if "end_to_end" not in self.results:
            return None

        data = self._extract_benchmark_data("end_to_end")
        no_zkp = zkp = None

        for bench in data:
            name = bench.get("name", "")
            stats = bench.get("stats", {})
            mean_time = stats.get("mean", 0)

            if "without_zkp" in name:
                no_zkp = mean_time
            elif "with_zkp" in name:
                zkp = mean_time

        if no_zkp and zkp:
            return zkp / no_zkp
        return None

    def _analyze_proof_scaling(self) -> float:
        """Analyze proof generation scaling exponent"""
        if "proof_generation_scaling" not in self.results:
            return None

        data = self._extract_benchmark_data("proof_generation_scaling")
        param_sizes = []
        mean_times = []

        for bench in data:
            name = bench.get("name", "")
            if "[" in name and "]" in name:
                size_str = name.split("[")[1].split("]")[0]
                try:
                    param_size = int(size_str)
                    param_sizes.append(param_size)

                    stats = bench.get("stats", {})
                    mean_time = stats.get("mean", 0)
                    mean_times.append(mean_time)
                except ValueError:
                    continue

        if len(param_sizes) > 2:
            # Fit power law in log space
            log_sizes = np.log10(param_sizes)
            log_times = np.log10(mean_times)
            coeffs = np.polyfit(log_sizes, log_times, 1)
            return coeffs[0]  # Scaling exponent

        return None

    def _write_training_analysis(self, f):
        """Write training performance analysis"""
        if "training_comparison" not in self.results:
            f.write("No training comparison data available.\n\n")
            return

        data = self._extract_benchmark_data("training_comparison")
        no_zkp = zkp = None

        for bench in data:
            name = bench.get("name", "")
            stats = bench.get("stats", {})
            mean_time = stats.get("mean", 0) * 1000  # ms
            std_time = stats.get("stddev", 0) * 1000

            if "without_zkp" in name:
                no_zkp = (mean_time, std_time)
            elif "with_zkp" in name:
                zkp = (mean_time, std_time)

        if no_zkp and zkp:
            f.write(f"- **Baseline Training:** {no_zkp[0]:.1f} ± {no_zkp[1]:.1f} ms\n")
            f.write(f"- **ZKP Training:** {zkp[0]:.1f} ± {zkp[1]:.1f} ms\n")
            f.write(f"- **Overhead:** {zkp[0] / no_zkp[0]:.1f}x increase\n")
            f.write(
                f"- **Proof Time:** ~{zkp[0] - no_zkp[0]:.0f} ms ({(zkp[0] - no_zkp[0]) / zkp[0] * 100:.1f}% of total)\n\n"
            )
        else:
            f.write("Incomplete training comparison data.\n\n")

    def _write_proof_analysis(self, f):
        """Write proof generation analysis"""
        if "proof_generation_scaling" not in self.results:
            f.write("No proof generation scaling data available.\n\n")
            return

        data = self._extract_benchmark_data("proof_generation_scaling")

        f.write("**Proof Generation Times by Parameter Count:**\n\n")
        f.write("| Parameters | Time (ms) | Throughput (proofs/sec) |\n")
        f.write("|------------|-----------|-------------------------|\n")

        for bench in data:
            name = bench.get("name", "")
            if "[" in name and "]" in name:
                size_str = name.split("[")[1].split("]")[0]
                try:
                    param_size = int(size_str)
                    stats = bench.get("stats", {})
                    mean_time = stats.get("mean", 0) * 1000  # ms
                    ops = stats.get("ops", 0)

                    f.write(f"| {param_size:,} | {mean_time:.2f} | {ops:.1f} |\n")
                except ValueError:
                    continue

        f.write("\n")

        scaling_exp = self._analyze_proof_scaling()
        if scaling_exp:
            f.write(f"**Scaling Behavior:** O(n^{scaling_exp:.2f})\n\n")

    def _write_server_analysis(self, f):
        """Write server performance analysis"""
        if "aggregation_scaling" not in self.results:
            f.write("No server aggregation scaling data available.\n\n")
            return

        data = self._extract_benchmark_data("aggregation_scaling")

        f.write("**Server Aggregation Performance:**\n\n")
        f.write("| Clients | Aggregation Time (ms) |\n")
        f.write("|---------|----------------------|\n")

        for bench in data:
            name = bench.get("name", "")
            if "[" in name and "]" in name:
                client_str = name.split("[")[1].split("]")[0]
                try:
                    clients = int(client_str)
                    stats = bench.get("stats", {})
                    mean_time = stats.get("mean", 0) * 1000  # ms

                    f.write(f"| {clients} | {mean_time:.2f} |\n")
                except ValueError:
                    continue

        f.write("\n")

    def _write_end_to_end_analysis(self, f):
        """Write end-to-end analysis"""
        if "end_to_end" not in self.results:
            f.write("No end-to-end FL round data available.\n\n")
            return

        data = self._extract_benchmark_data("end_to_end")
        no_zkp = zkp = None

        for bench in data:
            name = bench.get("name", "")
            stats = bench.get("stats", {})
            mean_time = stats.get("mean", 0) * 1000  # ms
            std_time = stats.get("stddev", 0) * 1000

            if "without_zkp" in name:
                no_zkp = (mean_time, std_time)
            elif "with_zkp" in name:
                zkp = (mean_time, std_time)

        if no_zkp and zkp:
            f.write(f"- **Baseline FL Round:** {no_zkp[0]:.0f} ± {no_zkp[1]:.0f} ms\n")
            f.write(f"- **ZKP FL Round:** {zkp[0]:.0f} ± {zkp[1]:.0f} ms\n")
            f.write(
                f"- **End-to-End Overhead:** {zkp[0] / no_zkp[0]:.1f}x increase\n\n"
            )
        else:
            f.write("Incomplete end-to-end comparison data.\n\n")

    def _write_recommendations(self, f):
        """Write performance recommendations"""
        training_overhead = self._calculate_training_overhead()

        f.write("Based on the performance analysis:\n\n")

        if training_overhead and training_overhead > 100:
            f.write(
                "1. **High ZKP Overhead:** Consider proof batching or asynchronous generation\n"
            )

        f.write(
            "2. **Model Optimization:** Smaller models significantly reduce proof generation time\n"
        )
        f.write(
            "3. **Parameter Pruning:** Reducing parameter count has quadratic benefits for ZKP\n"
        )
        f.write(
            "4. **Hybrid Approach:** Use ZKP selectively for critical rounds only\n"
        )
        f.write(
            "5. **Hardware Acceleration:** GPU-based ZKP libraries may improve performance\n\n"
        )


def main():
    """Main function to generate comprehensive performance report"""
    generator = PerformanceReportGenerator("comprehensive_performance")

    # Run all benchmarks
    generator.run_all_benchmarks()

    # Generate plots
    generator.generate_plots()

    # Generate report
    generator.generate_report()

    print(f"\n✅ Performance analysis completed!")
    print(f"📁 Results directory: {generator.output_dir}")
    print(
        f"📊 Main plot: {generator.output_dir}/comprehensive_performance_analysis.png"
    )
    print(f"📄 Report: {generator.output_dir}/performance_analysis_report.md")


if __name__ == "__main__":
    main()
