"""
Test Suite for Secure Federated Learning Framework

This package contains comprehensive tests for the secure-fl framework including:
- Unit tests for individual components
- Integration tests for component interactions
- End-to-end tests for full system functionality
- Performance and security tests

Test structure:
    tests/
    ├── __init__.py              # This file
    ├── conftest.py              # Pytest configuration and fixtures
    ├── unit/                    # Unit tests
    │   ├── test_aggregation.py
    │   ├── test_client.py
    │   ├── test_quantization.py
    │   ├── test_server.py
    │   ├── test_stability_monitor.py
    │   ├── test_proof_manager.py
    │   └── test_utils.py
    ├── integration/             # Integration tests
    │   ├── test_fl_workflow.py
    │   ├── test_zkp_integration.py
    │   └── test_server_client.py
    ├── e2e/                     # End-to-end tests
    │   ├── test_complete_training.py
    │   ├── test_cli_commands.py
    │   └── test_experiments.py
    ├── fixtures/                # Test data and fixtures
    │   ├── models.py
    │   ├── datasets.py
    │   └── configs.py
    └── utils/                   # Test utilities
        ├── __init__.py
        ├── helpers.py
        └── mocks.py

Running tests:
    # All tests
    pytest

    # Unit tests only
    pytest tests/unit/

    # Integration tests only
    pytest tests/integration/

    # Specific test file
    pytest tests/unit/test_aggregation.py

    # Tests with coverage
    pytest --cov=secure_fl --cov-report=html

    # Specific markers
    pytest -m unit
    pytest -m "not slow"
    pytest -m "zkp and not gpu"

Environment variables for testing:
    SECURE_FL_TEST_MODE=1        # Enable test mode
    SECURE_FL_LOG_LEVEL=DEBUG    # Set log level
    SECURE_FL_SKIP_ZKP=1         # Skip ZKP tests if tools not available
    SECURE_FL_SKIP_GPU=1         # Skip GPU tests
    SECURE_FL_TEST_DATA_DIR      # Custom test data directory
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path for testing
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "secure_fl"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Test configuration
TEST_CONFIG = {
    "test_mode": os.getenv("SECURE_FL_TEST_MODE", "0") == "1",
    "log_level": os.getenv("SECURE_FL_LOG_LEVEL", "INFO"),
    "skip_zkp": os.getenv("SECURE_FL_SKIP_ZKP", "0") == "1",
    "skip_gpu": os.getenv("SECURE_FL_SKIP_GPU", "0") == "1",
    "test_data_dir": os.getenv(
        "SECURE_FL_TEST_DATA_DIR", str(TEST_DIR / "fixtures" / "data")
    ),
    "timeout": int(os.getenv("SECURE_FL_TEST_TIMEOUT", "300")),
}

# Version info
__version__ = "0.1.0"
__author__ = "Secure FL Test Team"

# Export test configuration
__all__ = ["TEST_CONFIG", "TEST_DIR", "PROJECT_ROOT", "SRC_DIR"]


def setup_test_environment():
    """Setup test environment with proper configuration"""
    # Set test mode environment variable
    os.environ["SECURE_FL_TEST_MODE"] = "1"

    # Configure logging for tests
    import logging

    logging.basicConfig(
        level=getattr(logging, TEST_CONFIG["log_level"]),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Disable some noisy loggers during testing
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # Create test data directory if it doesn't exist
    test_data_dir = Path(TEST_CONFIG["test_data_dir"])
    test_data_dir.mkdir(parents=True, exist_ok=True)

    return TEST_CONFIG


def check_test_dependencies():
    """Check if all required test dependencies are available"""
    missing_deps = []

    try:
        import pytest
    except ImportError:
        missing_deps.append("pytest")

    try:
        import torch
    except ImportError:
        missing_deps.append("torch")

    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    if missing_deps:
        raise ImportError(
            f"Missing test dependencies: {', '.join(missing_deps)}. "
            "Please install with: pip install -e '.[dev]'"
        )

    return True


def check_optional_dependencies():
    """Check optional dependencies and return availability status"""
    deps_status = {}

    # ZKP tools
    try:
        import subprocess

        result = subprocess.run(
            ["cairo-compile", "--version"], capture_output=True, timeout=5
        )
        deps_status["cairo"] = result.returncode == 0
    except:
        deps_status["cairo"] = False

    try:
        import subprocess

        result = subprocess.run(["circom", "--version"], capture_output=True, timeout=5)
        deps_status["circom"] = result.returncode == 0
    except:
        deps_status["circom"] = False

    # GPU availability
    try:
        import torch

        deps_status["cuda"] = torch.cuda.is_available()
    except:
        deps_status["cuda"] = False

    # Medical datasets
    try:
        import medmnist

        deps_status["medmnist"] = True
    except ImportError:
        deps_status["medmnist"] = False

    # Blockchain tools
    try:
        import web3

        deps_status["web3"] = True
    except ImportError:
        deps_status["web3"] = False

    return deps_status


# Initialize test environment when module is imported
if TEST_CONFIG["test_mode"]:
    setup_test_environment()
    check_test_dependencies()
