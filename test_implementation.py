#!/usr/bin/env python3
"""
Basic Test Suite for Secure Federated Learning Implementation

This file tests the core components of the secure FL framework to ensure
everything is working correctly.

Usage:
    python test_implementation.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from secure_fl.aggregation import FedJSCMAggregator
from secure_fl.client import SecureFlowerClient, create_client
from secure_fl.proof_manager import ClientProofManager, ServerProofManager
from secure_fl.quantization import FixedPointQuantizer, QuantizationConfig
from secure_fl.server import SecureFlowerStrategy, create_server_strategy
from secure_fl.stability_monitor import StabilityMonitor
from secure_fl.utils import (
    compute_parameter_norm,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    validate_parameters,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model moved to secure_fl.models
from secure_fl.models import SimpleModel


class SimpleTestModel(nn.Module):
    """Simple model for testing"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def test_basic_imports():
    """Test that all modules can be imported"""
    logger.info("Testing basic imports...")

    try:
        from secure_fl import (
            FedJSCMAggregator,
            SecureFlowerClient,
            SecureFlowerServer,
            StabilityMonitor,
        )

        logger.info("✓ All core modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False


def test_utils():
    """Test utility functions"""
    logger.info("Testing utility functions...")

    try:
        # Test parameter conversion
        model = SimpleTestModel()
        params = [p.detach().numpy() for p in model.parameters()]

        # Test conversion functions
        flower_params = ndarrays_to_parameters(params)
        recovered_params = parameters_to_ndarrays(flower_params)

        assert len(params) == len(recovered_params)
        for orig, recovered in zip(params, recovered_params):
            assert np.allclose(orig, recovered, rtol=1e-5)

        # Test parameter norm
        norm = compute_parameter_norm(params)
        assert norm > 0

        # Test parameter validation
        assert validate_parameters(params)

        logger.info("✓ Utility functions working correctly")
        return True

    except Exception as e:
        logger.error(f"✗ Utility test failed: {e}")
        return False


def test_quantization():
    """Test quantization functionality"""
    logger.info("Testing quantization...")

    try:
        # Create test parameters
        params = [
            np.random.randn(5, 3).astype(np.float32),
            np.random.randn(3).astype(np.float32),
        ]

        # Test quantization
        config = QuantizationConfig(bits=8, symmetric=True)
        quantizer = FixedPointQuantizer(config)

        quantized, metadata = quantizer.quantize(params)
        dequantized = quantizer.dequantize(quantized, metadata)

        # Check that quantized are integers
        for qp in quantized:
            assert qp.dtype in [np.int32, np.int64]

        # Check shapes are preserved
        assert len(params) == len(dequantized)
        for orig, deq in zip(params, dequantized):
            assert orig.shape == deq.shape

        logger.info("✓ Quantization working correctly")
        return True

    except Exception as e:
        logger.error(f"✗ Quantization test failed: {e}")
        return False


def test_aggregation():
    """Test FedJSCM aggregation"""
    logger.info("Testing FedJSCM aggregation...")

    try:
        aggregator = FedJSCMAggregator(momentum=0.9, learning_rate=0.01)

        # Create test client updates
        client1_update = [np.random.randn(5, 3), np.random.randn(3)]
        client2_update = [np.random.randn(5, 3), np.random.randn(3)]
        client_updates = [client1_update, client2_update]

        client_weights = [0.6, 0.4]
        global_params = [np.random.randn(5, 3), np.random.randn(3)]

        # Test aggregation
        updated_params = aggregator.aggregate(
            client_updates=client_updates,
            client_weights=client_weights,
            server_round=1,
            global_params=global_params,
        )

        # Check output
        assert len(updated_params) == len(global_params)
        for up, gp in zip(updated_params, global_params):
            assert up.shape == gp.shape

        # Test momentum state
        momentum_state = aggregator.get_momentum_state()
        assert momentum_state["initialized"] == True

        logger.info("✓ FedJSCM aggregation working correctly")
        return True

    except Exception as e:
        logger.error(f"✗ Aggregation test failed: {e}")
        return False


def test_stability_monitor():
    """Test stability monitoring"""
    logger.info("Testing stability monitor...")

    try:
        monitor = StabilityMonitor(window_size=5)

        # Simulate training rounds
        for round_num in range(1, 8):
            params = [
                np.random.randn(3, 2) * (1.0 / round_num),
                np.random.randn(2) * (1.0 / round_num),
            ]
            metrics = {
                "train_loss": 1.0 / round_num,
                "gradient_norm": 1.0 / round_num,
                "proof_time": 0.5,
            }

            stability_metrics = monitor.update(params, round_num, metrics)
            recommended_rigor = monitor.get_recommended_rigor()

            assert stability_metrics.overall_stability >= 0
            assert recommended_rigor in ["low", "medium", "high"]

        # Check final state
        assert monitor.get_stability_score() >= 0

        logger.info("✓ Stability monitor working correctly")
        return True

    except Exception as e:
        logger.error(f"✗ Stability monitor test failed: {e}")
        return False


def test_proof_managers():
    """Test proof manager initialization (without actual proof generation)"""
    logger.info("Testing proof managers...")

    try:
        # Test client proof manager initialization
        client_pm = ClientProofManager()
        assert client_pm is not None

        # Test server proof manager initialization
        server_pm = ServerProofManager()
        assert server_pm is not None

        # Note: We don't test actual proof generation since it requires
        # Cairo and Circom to be properly installed

        logger.info("✓ Proof managers initialized correctly")
        return True

    except Exception as e:
        logger.error(f"✗ Proof manager test failed: {e}")
        return False


def test_server_strategy():
    """Test server strategy creation"""
    logger.info("Testing server strategy...")

    try:
        strategy = create_server_strategy(
            model_fn=SimpleTestModel,
            momentum=0.9,
            enable_zkp=False,  # Disable ZKP for testing
            proof_rigor="medium",
        )

        assert strategy is not None
        assert isinstance(strategy, SecureFlowerStrategy)

        logger.info("✓ Server strategy created correctly")
        return True

    except Exception as e:
        logger.error(f"✗ Server strategy test failed: {e}")
        return False


def test_client_creation():
    """Test client creation"""
    logger.info("Testing client creation...")

    try:
        # Create dummy dataset
        from torch.utils.data import TensorDataset

        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)

        client = create_client(
            client_id="test_client",
            model_fn=SimpleTestModel,
            train_data=dataset,
            enable_zkp=False,  # Disable ZKP for testing
            local_epochs=1,
        )

        assert client is not None
        assert isinstance(client, SecureFlowerClient)
        assert client.client_id == "test_client"

        logger.info("✓ Client created correctly")
        return True

    except Exception as e:
        logger.error(f"✗ Client creation test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    logger.info("=" * 60)
    logger.info("SECURE FEDERATED LEARNING - IMPLEMENTATION TEST SUITE")
    logger.info("=" * 60)

    tests = [
        test_basic_imports,
        test_utils,
        test_quantization,
        test_aggregation,
        test_stability_monitor,
        test_proof_managers,
        test_server_strategy,
        test_client_creation,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"✗ {test_func.__name__} failed with exception: {e}")

    logger.info("=" * 60)
    logger.info(f"TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Implementation is working correctly.")
        logger.info("\nNext steps:")
        logger.info("1. Install ZKP tools (Cairo, Circom) for full functionality")
        logger.info("2. Run experiments: cd experiments && python train_secure_fl.py")
        logger.info("3. Check the documentation in docs/ folder")
    else:
        logger.warning("⚠️  Some tests failed. Check the error messages above.")
        logger.info(
            "This might be due to missing dependencies or system configuration."
        )

    logger.info("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
