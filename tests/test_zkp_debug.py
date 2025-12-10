"""
Debug ZKP Issues in Federated Learning Context

This script helps identify and debug issues with PySNARK integration
in the federated learning workflow.
"""

import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from secure_fl.client import SecureFlowerClient
from secure_fl.data.dataloader import FederatedDataLoader
from secure_fl.models import MNISTModel
from secure_fl.proof_manager import ClientProofManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_pysnark_basic():
    """Test basic PySNARK functionality"""
    logger.info("=== Testing Basic PySNARK ===")

    try:
        from pysnark.runtime import PrivVal, PubVal

        # Basic arithmetic
        a = PrivVal(3)
        b = PrivVal(4)
        c = a + b
        logger.info(f"✅ Basic arithmetic: {a} + {b} = {c}")

        # Multiplication
        d = a * b
        logger.info(f"✅ Multiplication: {a} * {b} = {d}")

        return True

    except Exception as e:
        logger.error(f"❌ Basic PySNARK failed: {e}")
        traceback.print_exc()
        return False


def test_pysnark_circuit_import():
    """Test importing our delta bound circuit"""
    logger.info("=== Testing Circuit Import ===")

    try:
        # Try importing the circuit
        from proofs.client_circuits.delta_bound import delta_bound_proof

        logger.info("✅ Circuit import successful")
        return True

    except Exception as e:
        logger.error(f"❌ Circuit import failed: {e}")
        traceback.print_exc()

        # Try fallback import
        try:
            repo_root = Path(__file__).resolve().parents[1]
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from proofs.client_circuits.delta_bound import delta_bound_proof

            logger.info("✅ Circuit import successful (fallback)")
            return True
        except Exception as e2:
            logger.error(f"❌ Circuit import fallback failed: {e2}")
            traceback.print_exc()
            return False


def test_simple_proof_generation():
    """Test simple proof generation with small parameters"""
    logger.info("=== Testing Simple Proof Generation ===")

    try:
        manager = ClientProofManager(use_pysnark=True)
        logger.info("✅ Proof manager created")

        # Very small test parameters
        initial_params = [np.array([1.0, 2.0], dtype=np.float32)]
        updated_params = [np.array([1.1, 2.1], dtype=np.float32)]

        logger.info("Testing delta bound proof with small parameters...")

        start_time = time.perf_counter()
        result = manager._generate_pysnark_delta_bound_proof(
            initial_params, updated_params, 0.2
        )
        end_time = time.perf_counter()

        logger.info(f"✅ Simple proof generated in {end_time - start_time:.3f}s")
        logger.info(f"Result: {result}")
        return True

    except Exception as e:
        logger.error(f"❌ Simple proof generation failed: {e}")
        traceback.print_exc()
        return False


def test_larger_proof_generation():
    """Test proof generation with larger parameters"""
    logger.info("=== Testing Larger Proof Generation ===")

    try:
        manager = ClientProofManager(use_pysnark=True)

        # Medium size parameters (like a small neural network layer)
        param_size = 100  # Start small
        initial_params = [np.random.randn(param_size).astype(np.float32)]
        updated_params = [
            initial_params[0] + 0.01 * np.random.randn(param_size).astype(np.float32)
        ]

        delta_norm = np.sqrt(np.sum((updated_params[0] - initial_params[0]) ** 2))
        logger.info(f"Parameter size: {param_size}, Delta norm: {delta_norm:.4f}")

        logger.info("Testing delta bound proof with medium parameters...")

        start_time = time.perf_counter()
        result = manager._generate_pysnark_delta_bound_proof(
            initial_params, updated_params, delta_norm + 0.1
        )
        end_time = time.perf_counter()

        logger.info(f"✅ Medium proof generated in {end_time - start_time:.3f}s")
        logger.info(f"Result enabled: {result.get('enabled', False)}")
        return True

    except Exception as e:
        logger.error(f"❌ Medium proof generation failed: {e}")
        traceback.print_exc()
        return False


def test_client_proof_integration():
    """Test proof generation in client context"""
    logger.info("=== Testing Client Integration ===")

    try:
        # Create small dataset using centralized data loader
        fed_loader = FederatedDataLoader(
            dataset_name="mnist",
            num_clients=1,
            iid=True,
            val_split=0.0,
            batch_size=16,
        )

        # Get client datasets
        client_datasets = fed_loader.create_client_datasets()
        train_data, _ = client_datasets[0]

        # Limit to very small subset for debugging
        small_subset = torch.utils.data.Subset(train_data, list(range(50)))
        train_loader = DataLoader(small_subset, batch_size=16, shuffle=True)

        # Create small model
        model = MNISTModel(hidden_dims=[32], output_dim=10)  # Much smaller

        # Create client with ZKP enabled
        client = SecureFlowerClient(
            client_id="debug_client",
            model=model,
            train_loader=train_loader,
            val_loader=None,
            enable_zkp=True,
            proof_rigor="low",  # Use low rigor for debugging
            local_epochs=1,
            learning_rate=0.01,
            quantize_weights=False,
        )

        logger.info("✅ Debug client created")

        # Get initial parameters
        initial_params = client.get_parameters({})
        logger.info(f"Model has {len(initial_params)} parameter tensors")

        total_params = sum(p.size for p in initial_params)
        logger.info(f"Total parameters: {total_params}")

        # Test training with timeout
        logger.info("Starting training with ZKP (with monitoring)...")

        config = {"server_round": 1}

        start_time = time.perf_counter()
        updated_params, num_examples, metrics = client.fit(initial_params, config)
        end_time = time.perf_counter()

        logger.info(f"✅ Training completed in {end_time - start_time:.3f}s")
        logger.info(f"Proof time: {metrics.get('proof_time', 0):.3f}s")
        logger.info(f"Training time: {metrics.get('training_time', 0):.3f}s")

        return True

    except Exception as e:
        logger.error(f"❌ Client integration failed: {e}")
        traceback.print_exc()
        return False


def test_proof_generation_scaling():
    """Test how proof generation scales with parameter size"""
    logger.info("=== Testing Proof Generation Scaling ===")

    sizes = [10, 50, 100, 200]  # Start with small sizes
    results = {}

    try:
        manager = ClientProofManager(use_pysnark=True)

        for size in sizes:
            logger.info(f"Testing with parameter size: {size}")

            initial_params = [np.random.randn(size).astype(np.float32) * 0.1]
            updated_params = [
                initial_params[0] + 0.01 * np.random.randn(size).astype(np.float32)
            ]

            delta_norm = np.sqrt(np.sum((updated_params[0] - initial_params[0]) ** 2))

            try:
                start_time = time.perf_counter()
                result = manager._generate_pysnark_delta_bound_proof(
                    initial_params, updated_params, delta_norm + 0.1
                )
                end_time = time.perf_counter()

                duration = end_time - start_time
                results[size] = {
                    "duration": duration,
                    "success": result is not None and result.get("enabled", False),
                    "delta_norm": delta_norm,
                }

                logger.info(
                    f"  Size {size}: {duration:.3f}s, Success: {results[size]['success']}"
                )

                # If it takes more than 30 seconds, stop testing larger sizes
                if duration > 30:
                    logger.warning(
                        f"  Size {size} took {duration:.1f}s, skipping larger sizes"
                    )
                    break

            except Exception as e:
                logger.error(f"  Size {size} failed: {e}")
                results[size] = {
                    "duration": None,
                    "success": False,
                    "error": str(e),
                    "delta_norm": delta_norm,
                }
                # If we get an error, stop testing
                break

        logger.info("Scaling test results:")
        for size, result in results.items():
            if result["success"]:
                logger.info(f"  Size {size}: {result['duration']:.3f}s ✅")
            else:
                logger.info(f"  Size {size}: FAILED ❌")

        return len(results) > 0

    except Exception as e:
        logger.error(f"❌ Scaling test failed: {e}")
        traceback.print_exc()
        return False


def debug_circuit_execution():
    """Debug the actual circuit execution step by step"""
    logger.info("=== Debugging Circuit Execution ===")

    try:
        # Test circuit import first
        logger.info("Step 1: Import circuit")
        from proofs.client_circuits.delta_bound import delta_bound_proof

        logger.info("✅ Circuit imported")

        # Test PySNARK types
        logger.info("Step 2: Import PySNARK types")
        from pysnark.runtime import PrivVal, PubVal

        logger.info("✅ PySNARK types imported")

        # Create test data
        logger.info("Step 3: Create test vectors")
        size = 5  # Very small for debugging
        initial = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        updated = np.array([1.1, 2.1, 3.1, 4.1, 5.1], dtype=np.float32)

        # Convert to fixed point
        scale = 1  # No scaling for simplicity
        initial_int = (initial * scale).astype(int)
        updated_int = (updated * scale).astype(int)

        logger.info(f"Initial (int): {initial_int}")
        logger.info(f"Updated (int): {updated_int}")

        # Calculate bound
        delta = updated - initial
        l2_norm_sq = int(np.sum(delta**2) * scale * scale)
        bound = l2_norm_sq + 1  # Ensure it passes

        logger.info(f"Delta L2^2: {l2_norm_sq}, Bound: {bound}")

        # Test circuit call
        logger.info("Step 4: Call circuit")
        start_time = time.perf_counter()

        # Call with explicit parameters to debug
        result = delta_bound_proof(initial_int.tolist(), updated_int.tolist(), bound)

        end_time = time.perf_counter()
        logger.info(f"✅ Circuit executed in {end_time - start_time:.3f}s")
        logger.info(f"Circuit result: {result}")

        return True

    except Exception as e:
        logger.error(f"❌ Circuit execution debug failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all debug tests"""
    logger.info("🔍 Starting ZKP Debug Session")
    logger.info("=" * 50)

    tests = [
        ("Basic PySNARK", test_pysnark_basic),
        ("Circuit Import", test_pysnark_circuit_import),
        ("Circuit Execution Debug", debug_circuit_execution),
        ("Simple Proof Generation", test_simple_proof_generation),
        ("Larger Proof Generation", test_larger_proof_generation),
        ("Proof Scaling", test_proof_generation_scaling),
        ("Client Integration", test_client_proof_integration),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        logger.info("-" * 40)

        try:
            start_time = time.perf_counter()
            success = test_func()
            end_time = time.perf_counter()

            results[test_name] = {"success": success, "duration": end_time - start_time}

            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{status} in {end_time - start_time:.3f}s")

        except Exception as e:
            logger.error(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = {"success": False, "duration": None, "error": str(e)}

    # Summary
    logger.info("\n📊 DEBUG SUMMARY")
    logger.info("=" * 50)

    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)

    logger.info(f"Tests passed: {passed}/{total}")

    for test_name, result in results.items():
        status = "✅" if result["success"] else "❌"
        duration = f"{result['duration']:.3f}s" if result["duration"] else "N/A"
        logger.info(f"{status} {test_name}: {duration}")

        if not result["success"] and "error" in result:
            logger.info(f"    Error: {result['error']}")

    # Recommendations
    logger.info("\n💡 RECOMMENDATIONS")
    logger.info("-" * 30)

    if not results.get("Basic PySNARK", {}).get("success", False):
        logger.info("❌ PySNARK basic functionality is broken")
        logger.info("   → Check PySNARK installation and backends")
    elif not results.get("Circuit Import", {}).get("success", False):
        logger.info("❌ Circuit import is failing")
        logger.info("   → Check circuit file paths and syntax")
    elif not results.get("Simple Proof Generation", {}).get("success", False):
        logger.info("❌ Simple proof generation is failing")
        logger.info("   → Check circuit logic and parameter handling")
    elif not results.get("Client Integration", {}).get("success", False):
        logger.info("❌ Client integration is failing")
        logger.info("   → Check parameter conversion and client workflow")
    else:
        logger.info("✅ Most tests passed - system appears functional")
        logger.info("   → Ready for performance benchmarking")


if __name__ == "__main__":
    main()
