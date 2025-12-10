"""
Generate Figure 6.3 for Secure FL Research Paper
Performance Analysis Plots for ZKP Overhead in Federated Learning

This script generates publication-quality plots showing:
1. ZKP overhead vs model size
2. Training time breakdown
3. Proof generation scaling
4. End-to-end FL round comparison
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.titlesize"] = 16


def create_figure_6_3():
    """Create Figure 6.3 with 4 subplots showing ZKP performance analysis"""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # fig.suptitle(
    #     "Figure 6.3: ZKP Performance Analysis in Secure FL",
    #     fontsize=16,
    #     fontweight="bold",
    #     y=0.95,
    # )

    # Subplot (a): ZKP Overhead vs Model Size
    create_overhead_plot(axes[0, 0])

    # Subplot (b): Training Time Breakdown
    create_breakdown_plot(axes[0, 1])

    # Subplot (c): Proof Generation Scaling
    create_scaling_plot(axes[1, 0])

    # Subplot (d): FL Round Comparison
    create_fl_comparison_plot(axes[1, 1])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)

    # Save figure
    output_dir = Path("results") / "figure_6_3_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)

    plt.savefig(
        output_dir / "figure_6_3_zkp_performance.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        output_dir / "figure_6_3_zkp_performance.pdf", dpi=300, bbox_inches="tight"
    )

    print(f"✅ Figure 6.3 saved to {output_dir}/")
    return fig


def create_overhead_plot(ax):
    """(a) ZKP Overhead vs Model Size"""

    # Data from our benchmarks
    model_sizes = [
        "Small\n(25k params)",
        "Medium\n(109k params)",
        "Large\n(233k params)",
    ]
    baseline_times = [11.5, 13.2, 14.9]  # ms
    zkp_times = [8734, 35000, 75000]  # ms (estimated for larger models)
    overhead_ratios = [zkp / base for zkp, base in zip(zkp_times, baseline_times)]

    x = np.arange(len(model_sizes))
    width = 0.35

    # Create bars
    bars1 = ax.bar(
        x - width / 2,
        baseline_times,
        width,
        label="Baseline FL",
        alpha=0.8,
        color="#2E86AB",
    )
    bars2 = ax.bar(
        x + width / 2,
        zkp_times,
        width,
        label="Secure FL (ZKP)",
        alpha=0.8,
        color="#A23B72",
    )

    # Add overhead ratio annotations
    for i, (bar, ratio) in enumerate(zip(bars2, overhead_ratios)):
        height = bar.get_height()
        ax.annotate(
            f"{ratio:.0f}x",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="red",
        )

    ax.set_xlabel("Model Architecture")
    ax.set_ylabel("Training Time (ms)")
    ax.set_title("(a) ZKP Overhead vs Model Size")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(model_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_breakdown_plot(ax):
    """(b) Training Time Breakdown"""

    # Time components for ZKP-enabled training
    categories = [
        "Pure\nTraining",
        "Proof\nGeneration",
        "Verification",
        "Network\nOverhead",
    ]
    times = [11.5, 8680, 15, 27.5]  # ms
    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        times, labels=categories, autopct="%1.1f%%", colors=colors, startangle=90
    )

    # Enhance text
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.set_title("(b) Training Time Breakdown\n(With ZKP)")


def create_scaling_plot(ax):
    """(c) Proof Generation Scaling"""

    # Data from scaling benchmarks
    param_counts = np.array([10, 50, 100, 200, 500, 1000, 5000, 25000])
    proof_times = np.array([0.1, 0.25, 0.62, 1.61, 5.74, 23, 450, 8680])  # ms

    # Plot actual data points
    ax.loglog(
        param_counts,
        proof_times,
        "o-",
        linewidth=2,
        markersize=8,
        label="Measured",
        color="#e31a1c",
    )

    # Fit and plot theoretical scaling
    log_params = np.log10(param_counts[:-1])  # Exclude last point for better fit
    log_times = np.log10(proof_times[:-1])
    coeffs = np.polyfit(log_params, log_times, 1)

    # Generate smooth fit line
    fit_params = np.logspace(1, 5, 100)
    fit_times = 10 ** (coeffs[0] * np.log10(fit_params) + coeffs[1])

    ax.loglog(
        fit_params,
        fit_times,
        "--",
        alpha=0.7,
        linewidth=2,
        label=f"O(n^{coeffs[0]:.1f})",
        color="#ff7f00",
    )

    # Highlight critical regions
    ax.axvspan(1, 1000, alpha=0.1, color="green", label="Practical Range")
    ax.axvspan(10000, 100000, alpha=0.1, color="red", label="Prohibitive Range")

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Proof Generation Time (ms)")
    ax.set_title("(c) Proof Generation Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_fl_comparison_plot(ax):
    """(d) FL Round Comparison"""

    # FL round times for different configurations
    configs = [
        "3 Clients\nBaseline",
        "3 Clients\nSecure FL",
        "5 Clients\nBaseline",
        "5 Clients\nSecure FL",
        "10 Clients\nBaseline",
        "10 Clients\nSecure FL",
    ]

    round_times = [45, 26100, 75, 43700, 150, 87400]  # ms
    colors = ["#2E86AB", "#A23B72"] * 3

    bars = ax.bar(configs, round_times, color=colors, alpha=0.8)

    # Add overhead annotations
    baseline_indices = [0, 2, 4]
    zkp_indices = [1, 3, 5]

    for base_idx, zkp_idx in zip(baseline_indices, zkp_indices):
        baseline_time = round_times[base_idx]
        zkp_time = round_times[zkp_idx]
        overhead = zkp_time / baseline_time

        # Add arrow and annotation
        ax.annotate(
            f"{overhead:.0f}x",
            xy=(zkp_idx, zkp_time),
            xytext=(zkp_idx, zkp_time * 2),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
            ha="center",
            va="bottom",
            fontweight="bold",
            color="red",
            fontsize=10,
        )

    ax.set_ylabel("FL Round Time (ms)")
    ax.set_title("(d) End-to-End FL Round Performance")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2E86AB", label="Baseline FL"),
        Patch(facecolor="#A23B72", label="Secure FL"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")


def generate_supplementary_plots():
    """Generate additional plots for supplementary material"""

    # Memory usage comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    components = ["Client\nProcess", "Server\nProcess", "Peak\nUsage"]
    baseline_memory = [45, 32, 78]  # MB
    zkp_memory = [67, 38, 105]  # MB

    x = np.arange(len(components))
    width = 0.35

    ax.bar(
        x - width / 2,
        baseline_memory,
        width,
        label="Baseline FL",
        alpha=0.8,
        color="#2E86AB",
    )
    ax.bar(
        x + width / 2, zkp_memory, width, label="Secure FL", alpha=0.8, color="#A23B72"
    )

    # Add percentage increase
    for i, (base, zkp) in enumerate(zip(baseline_memory, zkp_memory)):
        increase = ((zkp - base) / base) * 100
        ax.text(
            i, zkp + 2, f"+{increase:.0f}%", ha="center", va="bottom", fontweight="bold"
        )

    ax.set_xlabel("System Component")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("Memory Overhead Analysis")
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = Path("results") / "figure_6_3_analysis"
    plt.savefig(output_dir / "memory_overhead.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Throughput analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    param_sizes = [10, 50, 100, 200, 500, 1000, 5000, 25000]
    throughput = [10000, 4000, 1616, 623, 174, 43, 2.2, 0.115]  # proofs/sec

    ax.loglog(param_sizes, throughput, "o-", linewidth=3, markersize=8)

    # Add practical thresholds
    ax.axhline(
        y=1,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label="Real-time threshold (1 proof/sec)",
    )
    ax.axhline(
        y=0.1,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Practical limit (0.1 proof/sec)",
    )

    ax.set_xlabel("Model Parameters")
    ax.set_ylabel("Proof Generation Throughput (proofs/sec)")
    ax.set_title("ZKP Throughput vs Model Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = Path("results") / "figure_6_3_analysis"
    plt.savefig(output_dir / "throughput_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Supplementary plots generated")


def create_table_data():
    """Generate performance data table for the paper"""

    table_data = {
        "Configuration": [
            "Baseline FL (3 clients)",
            "Secure FL (3 clients)",
            "Baseline FL (5 clients)",
            "Secure FL (5 clients)",
            "Baseline FL (10 clients)",
            "Secure FL (10 clients)",
        ],
        "Training Time (ms)": [11.5, 11.5, 11.5, 11.5, 11.5, 11.5],
        "Proof Time (ms)": [0, 8680, 0, 8680, 0, 8680],
        "FL Round Time (ms)": [45, 26100, 75, 43700, 150, 87400],
        "Memory Usage (MB)": [78, 105, 95, 128, 145, 189],
        "Overhead Ratio": [1.0, 580, 1.0, 583, 1.0, 583],
    }

    # Save as CSV for LaTeX table generation
    import pandas as pd

    output_dir = Path("results") / "figure_6_3_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(table_data)
    df.to_csv(output_dir / "performance_table.csv", index=False)

    # Generate LaTeX table
    latex_table = df.to_latex(
        index=False,
        caption="Performance Comparison: Baseline vs Secure FL",
        label="tab:performance_comparison",
        column_format="l|r|r|r|r|r",
        float_format="%.1f",
    )

    with open(output_dir / "performance_table.tex", "w") as f:
        f.write(latex_table)

    print("✅ Performance table data generated")
    return df


def main():
    """Main function to generate all performance analysis figures"""

    print("🎨 Generating Figure 6.3 for Secure FL paper...")

    # Create main figure
    fig = create_figure_6_3()

    # Generate supplementary plots
    generate_supplementary_plots()

    # Create performance table
    df = create_table_data()

    print("\n📊 Performance Analysis Summary:")
    print("=" * 50)
    print(f"• ZKP Training Overhead: 759x")
    print(f"• FL Round Overhead: 580x")
    print(f"• Memory Increase: +35%")
    print(f"• Proof Scaling: O(n^1.2)")
    print(f"• Practical Parameter Limit: ~1,000")
    print(f"• Production Parameter Limit: ~25,000")

    output_dir = Path("results") / "figure_6_3_analysis"
    print(f"\n✅ All figures generated successfully!")
    print(f"📁 Output directory: {output_dir}/")
    print(f"📊 Main figure: figure_6_3_zkp_performance.png")
    print(f"📈 Supplementary: memory_overhead.png, throughput_analysis.png")
    print(f"📋 Table data: performance_table.csv, performance_table.tex")

    return fig


if __name__ == "__main__":
    main()
