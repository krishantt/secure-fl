"""
Pytest Configuration and Fixtures for Secure Federated Learning Tests

This module provides common fixtures, configuration, and utilities used across
all test modules in the secure-fl test suite.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from secure_fl.quantization import QuantizationConfig

# Import secure_fl components
# Import test configuration
from tests import TEST_CONFIG, check_optional_dependencies

# ==================== Pytest Configuration ====================


def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Add custom markers
    markers = [
        "unit: Unit tests for individual components",
        "integration: Integration tests for component interactions",
        "e2e: End-to-end system tests",
        "slow: Tests that take longer than 5 seconds",
        "zkp: Tests requiring zero-knowledge proof tools",
        "gpu: Tests requiring GPU acceleration",
        "blockchain: Tests requiring blockchain setup",
        "network: Tests requiring network connectivity",
        "docker: Tests requiring Docker",
        "external: Tests requiring external services",
        "smoke: Basic functionality smoke tests",
        "regression: Regression tests for bug fixes",
        "performance: Performance benchmarking tests",
        "security: Security-focused tests",
        "stress: Stress and load tests",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and handle skips"""
    deps_status = check_optional_dependencies()

    for item in items:
        # Auto-mark tests based on file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Skip tests based on dependencies
        if item.get_closest_marker("zkp") and not deps_status.get("cairo", False):
            item.add_marker(pytest.mark.skip(reason="Cairo not available"))

        if item.get_closest_marker("gpu") and not deps_status.get("cuda", False):
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))

        if TEST_CONFIG["skip_zkp"] and item.get_closest_marker("zkp"):
            item.add_marker(pytest.mark.skip(reason="ZKP tests disabled"))

        if TEST_CONFIG["skip_gpu"] and item.get_closest_marker("gpu"):
            item.add_marker(pytest.mark.skip(reason="GPU tests disabled"))


# ==================== Session-level Fixtures ====================


@pytest.fixture(scope="session")
def test_config():
    """Test configuration dictionary"""
    return TEST_CONFIG.copy()


@pytest.fixture(scope="session")
def dependencies_status():
    """Status of optional dependencies"""
    return check_optional_dependencies()


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for the test session"""
    temp_path = tempfile.mkdtemp(prefix="secure_fl_test_")
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="session")
def test_data_dir(temp_dir):
    """Create test data directory"""
    data_dir = temp_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# ==================== Model Fixtures ====================

# Model moved to secure_fl.models


class SimpleTestModel(nn.Module):
    """Simple neural network for testing"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 5, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


@pytest.fixture
def model_parameters():
    """Create sample model parameters as numpy arrays"""
    # Create a simple model to get realistic parameter shapes
    model = SimpleTestModel(input_dim=10, hidden_dim=5, output_dim=2)

    # Convert to numpy arrays
    params = []
    for param in model.parameters():
        params.append(param.detach().numpy().astype(np.float32))

    return params


@pytest.fixture
def server_config():
    """Server configuration for testing"""
    return {
        "fraction_fit": 0.5,
        "fraction_evaluate": 0.5,
        "min_fit_clients": 2,
        "min_evaluate_clients": 2,
        "min_available_clients": 2,
        "enable_monitoring": True,
        "enable_zkp": False,  # Disabled for most tests
        "aggregation_config": {
            "momentum": 0.9,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
        },
    }


@pytest.fixture
def client_datasets():
    """Create mock datasets for multiple clients"""
    datasets = []

    # Create 3 client datasets with different sizes
    dataset_sizes = [100, 80, 120]

    for size in dataset_sizes:
        # Generate random data
        X = torch.randn(size, 10)  # 10 features
        y = torch.randint(0, 2, (size,))  # Binary classification

        dataset = TensorDataset(X, y)
        datasets.append(dataset)

    return datasets


@pytest.fixture
def quantization_config():
    """Quantization configuration for testing"""
    return QuantizationConfig(bits=8, symmetric=True, per_channel=False, signed=True)


@pytest.fixture
def simple_model():
    """Simple PyTorch model for testing"""
    return SimpleTestModel(input_dim=10, hidden_dim=5, output_dim=2)


@pytest.fixture
def client_config():
    """Client configuration for testing"""
    return {
        "local_epochs": 1,
        "batch_size": 10,
        "learning_rate": 0.01,
        "enable_zkp": False,  # Disabled for most tests
        "quantize_weights": True,
    }


@pytest.fixture
def aggregation_config():
    """Aggregation configuration for testing"""
    return {
        "momentum": 0.9,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "adaptive_momentum": False,
    }


@pytest.fixture
def zkp_config():
    """ZKP configuration for testing"""
    return {
        "enable_zkp": True,
        "proof_rigor": "medium",
        "blockchain_verification": False,
        "quantize_weights": True,
        "quantization_bits": 8,
        "proof_timeout": 60,
    }


# ==================== Mock Fixtures ====================


@pytest.fixture
def mock_client_proof_manager():
    """Mock client proof manager"""
    mock = Mock()
    mock.generate_training_proof.return_value = {
        "proof": "mock_client_proof_data",
        "public_inputs": [1, 2, 3],
        "proof_time": 0.1,
    }
    mock.verify_proof.return_value = True
    return mock


@pytest.fixture
def mock_server_proof_manager():
    """Mock server proof manager"""
    mock = Mock()
    mock.generate_aggregation_proof.return_value = {
        "proof": "mock_server_proof_data",
        "public_inputs": [4, 5, 6],
        "proof_time": 0.2,
    }
    mock.verify_proof.return_value = True
    return mock


@pytest.fixture
def mock_blockchain_verifier():
    """Mock blockchain verifier"""
    mock = Mock()
    mock.submit_proof.return_value = True
    mock.verify_on_chain.return_value = True
    mock.get_transaction_hash.return_value = "0x1234567890abcdef"
    return mock


# ==================== Utility Fixtures ====================


@pytest.fixture
def sample_client_updates(model_parameters):
    """Sample client parameter updates"""
    updates = []
    for i in range(3):  # 3 clients
        client_update = []
        for param in model_parameters:
            # Add some noise to simulate updates
            noise = np.random.normal(0, 0.01, param.shape)
            updated_param = param + noise
            client_update.append(updated_param.astype(np.float32))
        updates.append(client_update)
    return updates


@pytest.fixture
def client_weights():
    """Client weights for aggregation"""
    return [0.4, 0.35, 0.25]  # Sum to 1.0


@pytest.fixture
def training_metrics():
    """Sample training metrics"""
    return {
        "train_loss": 1.5,
        "train_accuracy": 0.75,
        "val_loss": 1.8,
        "val_accuracy": 0.70,
        "gradient_norm": 0.1,
        "parameter_norm": 2.3,
        "proof_time": 0.5,
        "communication_time": 0.2,
    }


# ==================== Logging and Cleanup ====================


@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """Setup logging for tests"""
    caplog.set_level(logging.INFO)
    # Disable some verbose loggers during testing
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def cleanup_environment():
    """Cleanup environment after each test"""
    # Store original environment
    original_env = os.environ.copy()

    yield

    # Restore environment
    os.environ.clear()
    os.environ.update(original_env)


# ==================== Performance Testing Fixtures ====================


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests"""
    return {
        "warmup_rounds": 2,
        "measurement_rounds": 5,
        "timeout_seconds": 60,
        "memory_threshold_mb": 500,
        "cpu_threshold_percent": 80,
    }


# ==================== Parametrized Fixtures ====================


@pytest.fixture(params=[1, 3, 5])
def num_clients(request):
    """Parametrized number of clients"""
    return request.param


@pytest.fixture(params=["low", "medium", "high"])
def proof_rigor(request):
    """Parametrized proof rigor levels"""
    return request.param


@pytest.fixture(params=[4, 8, 16])
def quantization_bits(request):
    """Parametrized quantization bits"""
    return request.param


# ==================== Utility Functions ====================


def assert_parameters_equal(
    params1: list[np.ndarray], params2: list[np.ndarray], rtol: float = 1e-5
):
    """Assert that two parameter lists are equal"""
    assert len(params1) == len(params2), "Parameter lists have different lengths"

    for i, (p1, p2) in enumerate(zip(params1, params2)):
        assert p1.shape == p2.shape, (
            f"Parameter {i} shapes don't match: {p1.shape} vs {p2.shape}"
        )
        assert np.allclose(p1, p2, rtol=rtol), f"Parameter {i} values don't match"


def create_mock_fl_result(success: bool = True, metrics: dict | None = None):
    """Create a mock FL training result"""
    default_metrics = {"train_loss": 1.0, "train_accuracy": 0.8, "num_examples": 100}

    if metrics:
        default_metrics.update(metrics)

    from flwr.common import Code, FitRes, Status

    status = Status(code=Code.OK if success else Code.UNKNOWN, message="")
    return FitRes(
        status=status,
        parameters=[],  # Empty for mock
        num_examples=default_metrics["num_examples"],
        metrics=default_metrics,
    )


@pytest.fixture
def mock_fl_result():
    """Mock FL training result"""
    return create_mock_fl_result()
