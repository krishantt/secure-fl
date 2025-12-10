"""
Secure Federated Learning (secure-fl)

A dual-verifiable framework for federated learning using zero-knowledge proofs.
This package provides a complete implementation of federated learning with
dual zero-knowledge proof verification using zk-STARKs (client-side) and
zk-SNARKs (server-side).

Main components:
- SecureFlowerServer: FL server with FedJSCM aggregation and ZKP verification
- SecureFlowerClient: FL client with local training and zk-STARK proof generation
- FedJSCMAggregator: Momentum-based aggregation algorithm
- ClientProofManager/ServerProofManager: ZKP proof generation and verification
- StabilityMonitor: Dynamic proof rigor adjustment
- Utility functions for parameter handling and quantization

Example usage:
    # Server setup
    from secure_fl import SecureFlowerServer, create_server_strategy

    strategy = create_server_strategy(
        model_fn=lambda: MyModel(),
        enable_zkp=True,
        proof_rigor="high"
    )
    server = SecureFlowerServer(strategy=strategy, num_rounds=10)
    server.start()

    # Client setup
    from secure_fl import SecureFlowerClient, create_client, start_client

    client = create_client(
        client_id="client_1",
        model_fn=lambda: MyModel(),
        train_data=train_dataset,
        enable_zkp=True
    )
    start_client(client, server_address="localhost:8080")
"""

# Version information
from ._version import __version__, get_build_info, print_version_info
from .aggregation import FedJSCMAggregator
from .client import SecureFlowerClient, create_client, start_client
from .proof_manager import ClientProofManager, ProofManagerBase, ServerProofManager
from .quantization import (
    FixedPointQuantizer,
    GradientAwareQuantizer,
    QuantizationConfig,
    compute_quantization_error,
    dequantize_parameters,
    quantize_parameters,
)
from .server import SecureFlowerServer, SecureFlowerStrategy, create_server_strategy
from .stability_monitor import StabilityMetrics, StabilityMonitor
from .utils import (
    aggregate_weighted_average,
    compute_hash,
    compute_parameter_diff,
    compute_parameter_norm,
    deserialize_parameters,
    get_parameter_stats,
    ndarrays_to_parameters,
    ndarrays_to_torch,
    parameters_to_ndarrays,
    serialize_parameters,
    torch_to_ndarrays,
    validate_parameters,
)

__author__ = "Krishant Timilsina, Bindu Paudel"
__email__ = "krishantt@example.com, bigya01@example.com"

# Package metadata
__title__ = "secure-fl"
__description__ = (
    "Dual-Verifiable Framework for Federated Learning using Zero-Knowledge Proofs"
)
__url__ = "https://github.com/krishantt/secure-fl"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Krishant Timilsina, Bindu Paudel"

# Export main classes and functions
__all__ = [
    # Version information
    "__version__",
    "get_build_info",
    "print_version_info",
    # Main FL components
    "SecureFlowerServer",
    "SecureFlowerStrategy",
    "SecureFlowerClient",
    "create_server_strategy",
    "create_client",
    "start_client",
    # Aggregation
    "FedJSCMAggregator",
    # ZKP components
    "ProofManagerBase",
    "ClientProofManager",
    "ServerProofManager",
    # Stability monitoring
    "StabilityMonitor",
    "StabilityMetrics",
    # Quantization
    "FixedPointQuantizer",
    "GradientAwareQuantizer",
    "QuantizationConfig",
    "quantize_parameters",
    "dequantize_parameters",
    "compute_quantization_error",
    # Utilities
    "parameters_to_ndarrays",
    "ndarrays_to_parameters",
    "torch_to_ndarrays",
    "ndarrays_to_torch",
    "compute_parameter_norm",
    "compute_parameter_diff",
    "compute_hash",
    "serialize_parameters",
    "deserialize_parameters",
    "aggregate_weighted_average",
    "validate_parameters",
    "get_parameter_stats",
]

# Configuration constants
DEFAULT_CONFIG = {
    "server": {
        "host": "localhost",
        "port": 8080,
        "num_rounds": 10,
    },
    "strategy": {
        "min_fit_clients": 2,
        "min_evaluate_clients": 2,
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
    },
    "aggregation": {
        "momentum": 0.9,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "adaptive_momentum": False,
    },
    "zkp": {
        "enable_zkp": True,
        "proof_rigor": "high",
        "blockchain_verification": False,
        "quantize_weights": True,
        "quantization_bits": 8,
    },
    "stability": {
        "window_size": 10,
        "stability_threshold_high": 0.9,
        "stability_threshold_medium": 0.7,
        "convergence_patience": 5,
        "min_rounds_for_adjustment": 3,
    },
}


def get_default_config():
    """Get default configuration dictionary"""
    return DEFAULT_CONFIG.copy()


def print_system_info():
    """Print system information and component status"""
    print(f"Secure FL v{__version__}")
    print(f"Description: {__description__}")
    print(f"Authors: {__author__}")
    print()

    # Check component availability
    components = {
        "Flower": True,  # Always available if imported
        "PyTorch": True,
        "Cairo": False,  # Would need to check installation
        "Circom": False,  # Would need to check installation
        "SnarkJS": False,  # Would need to check installation
    }

    try:
        import torch  # noqa: F401

        components["PyTorch"] = True
    except ImportError:
        components["PyTorch"] = False

    try:
        import flwr  # noqa: F401

        components["Flower"] = True
    except ImportError:
        components["Flower"] = False

    # Check external tools
    import subprocess

    try:
        result = subprocess.run(
            ["cairo-compile", "--version"], capture_output=True, timeout=5
        )
        components["Cairo"] = result.returncode == 0
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass

    try:
        result = subprocess.run(["circom", "--version"], capture_output=True, timeout=5)
        components["Circom"] = result.returncode == 0
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass

    try:
        result = subprocess.run(["snarkjs", "help"], capture_output=True, timeout=5)
        components["SnarkJS"] = result.returncode == 0
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass

    print("Component Status:")
    for component, available in components.items():
        status = "✓ Available" if available else "✗ Not Found"
        print(f"  {component}: {status}")

    print()
    print("Note: Cairo, Circom, and SnarkJS are required for full ZKP functionality")
    print("Install them separately for complete proof generation capabilities")


# Convenience function for quick setup


def create_secure_fl_setup(
    model_fn,
    train_datasets,
    val_datasets=None,
    config=None,
):
    """
    Create a complete secure FL setup with server and multiple clients

    Args:
        model_fn: Function that returns a PyTorch model
        train_datasets: List of training datasets for each client
        val_datasets: Optional list of validation datasets
        config: Optional configuration dictionary

    Returns:
        Tuple of (server, clients) ready for training
    """
    if config is None:
        config = get_default_config()

    # Create server strategy
    strategy = create_server_strategy(
        model_fn=model_fn, **config["aggregation"], **config["zkp"]
    )

    # Create server
    server = SecureFlowerServer(strategy=strategy, **config["server"])

    # Create clients
    clients = []
    for i, train_data in enumerate(train_datasets):
        val_data = val_datasets[i] if val_datasets else None

        client = create_client(
            client_id=f"client_{i}",
            model_fn=model_fn,
            train_data=train_data,
            val_data=val_data,
            **config["zkp"],
        )
        clients.append(client)

    return server, clients
