"""
Experiments Directory for Secure FL

This directory contains experiments, benchmarks, and training scripts
that are NOT bundled with the package distribution.

Use these scripts to test and evaluate the Secure FL framework.

Available experiments:
- benchmark.py: Multi-dataset benchmarking with performance analysis
- demo.py: Quick demonstration scripts
- train.py: Full training experiments with various configurations
"""

# This directory is not included in the package build
# Use these scripts for testing and evaluation purposes

__version__ = "1.0.0"

# Experiment metadata
EXPERIMENTS = {
    "benchmark": {
        "description": "Multi-dataset benchmarking suite",
        "datasets": ["mnist", "cifar10", "synthetic"],
        "features": ["performance_analysis", "zkp_overhead", "convergence_plots"],
    },
    "demo": {
        "description": "Quick demonstration and testing",
        "datasets": ["synthetic"],
        "features": ["quick_setup", "basic_fl"],
    },
    "train": {
        "description": "Full training experiments",
        "datasets": ["mnist", "cifar10", "synthetic"],
        "features": ["full_zkp", "blockchain_integration", "comprehensive_logging"],
    },
}


def list_experiments():
    """List available experiments"""
    print("Available Secure FL Experiments:")
    print("=" * 40)
    for name, info in EXPERIMENTS.items():
        print(f"\n{name}.py:")
        print(f"  Description: {info['description']}")
        print(f"  Datasets: {', '.join(info['datasets'])}")
        print(f"  Features: {', '.join(info['features'])}")


if __name__ == "__main__":
    list_experiments()
