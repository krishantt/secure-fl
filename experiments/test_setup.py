#!/usr/bin/env python3
"""
Test Script for Secure FL Experiments Setup

This script verifies that the new experiments directory structure works correctly
and tests the multi-dataset benchmark functionality.
"""

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_project_structure():
    """Check that the project structure is correct"""
    logger.info("Checking project structure...")

    # Get project root
    project_root = Path(__file__).parent.parent

    # Check required directories
    required_dirs = ["secure_fl", "experiments", "proofs", "blockchain", "docs"]

    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
        else:
            logger.info(f"✓ Found {dir_name}/")

    if missing_dirs:
        logger.error(f"Missing directories: {missing_dirs}")
        return False

    # Check experiments files
    experiments_dir = project_root / "experiments"
    required_experiment_files = [
        "__init__.py",
        "benchmark.py",
        "demo.py",
        "train.py",
        "README.md",
        "config.yaml",
    ]

    missing_files = []
    for file_name in required_experiment_files:
        file_path = experiments_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            logger.info(f"✓ Found experiments/{file_name}")

    if missing_files:
        logger.error(f"Missing experiment files: {missing_files}")
        return False

    logger.info("✓ Project structure is correct")
    return True


def check_package_import():
    """Test that secure_fl package imports correctly"""
    logger.info("Testing package imports...")

    try:
        # Test main package import
        import secure_fl  # noqa: F401

        logger.info("✓ secure_fl package imports successfully")

        # Test key components
        from secure_fl.models import CIFAR10Model, MNISTModel, SimpleModel  # noqa: F401

        logger.info("✓ Model imports working")

        from secure_fl.server import (  # noqa: F401
            SecureFlowerServer,
            SecureFlowerStrategy,
        )

        logger.info("✓ Server imports working")

        from secure_fl.client import SecureFlowerClient  # noqa: F401

        logger.info("✓ Client imports working")

        from secure_fl.aggregation import FedJSCMAggregator  # noqa: F401

        logger.info("✓ Aggregation imports working")

        return True

    except ImportError as e:
        logger.error(f"Package import failed: {e}")
        return False


def test_model_creation():
    """Test that models can be created correctly"""
    logger.info("Testing model creation...")

    try:
        from secure_fl.models import CIFAR10Model, MNISTModel, SimpleModel

        # Test MNIST model
        mnist_model = MNISTModel()
        logger.info(f"✓ MNISTModel created: {mnist_model.__class__.__name__}")

        # Test CIFAR-10 model
        cifar_model = CIFAR10Model()
        logger.info(f"✓ CIFAR10Model created: {cifar_model.__class__.__name__}")

        # Test Simple model
        simple_model = SimpleModel(input_dim=784, output_dim=10)
        logger.info(f"✓ SimpleModel created: {simple_model.__class__.__name__}")

        return True

    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        return False


def test_dataset_manager():
    """Test the DatasetManager from benchmark.py"""
    logger.info("Testing DatasetManager...")

    try:
        # Add current directory to path to import from experiments
        experiments_dir = Path(__file__).parent
        sys.path.insert(0, str(experiments_dir))

        from benchmark import DatasetManager

        # Create dataset manager
        with tempfile.TemporaryDirectory() as temp_dir:
            dm = DatasetManager(data_dir=temp_dir)
            logger.info("✓ DatasetManager created successfully")

            # Test model creation for each dataset
            for dataset in ["mnist", "cifar10", "synthetic"]:
                try:
                    model = dm.get_model(dataset)
                    logger.info(
                        f"✓ {dataset} model created: {model.__class__.__name__}"
                    )
                except Exception as e:
                    logger.error(f"Failed to create {dataset} model: {e}")
                    return False

            # Test dataset info
            for dataset in ["mnist", "cifar10", "synthetic"]:
                info = dm.get_dataset_info(dataset)
                # model_class is now a string after JSON serialization fix
                model_class_name = info["model_class"]
                logger.info(f"✓ {dataset} info: {model_class_name}")

        return True

    except Exception as e:
        logger.error(f"DatasetManager test failed: {e}")
        return False
    finally:
        # Remove from path
        if str(experiments_dir) in sys.path:
            sys.path.remove(str(experiments_dir))


def test_benchmark_script():
    """Test running the benchmark script with --help"""
    logger.info("Testing benchmark script...")

    try:
        benchmark_script = Path(__file__).parent / "benchmark.py"

        # Test help command
        result = subprocess.run(
            [sys.executable, str(benchmark_script), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            logger.info("✓ Benchmark script help command works")
            return True
        else:
            logger.error(f"Benchmark script help failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Benchmark script help command timed out")
        return False
    except Exception as e:
        logger.error(f"Benchmark script test failed: {e}")
        return False


def test_quick_benchmark():
    """Test running a very quick benchmark"""
    logger.info("Testing quick benchmark execution...")

    try:
        benchmark_script = Path(__file__).parent / "benchmark.py"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Run quick synthetic benchmark
            result = subprocess.run(
                [
                    sys.executable,
                    str(benchmark_script),
                    "--datasets",
                    "synthetic",
                    "--quick",
                    "--configs",
                    "baseline_iid",
                    "--output-dir",
                    temp_dir,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                logger.info("✓ Quick benchmark completed successfully")

                # Check if results file was created
                results_file = Path(temp_dir) / "multi_dataset_benchmark_results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        results = json.load(f)
                    logger.info(f"✓ Results saved with {len(results)} datasets")
                    return True
                else:
                    logger.warning("Results file not found, but benchmark completed")
                    return True
            else:
                logger.error(f"Quick benchmark failed: {result.stderr}")
                return False

    except subprocess.TimeoutExpired:
        logger.error("Quick benchmark timed out")
        return False
    except Exception as e:
        logger.error(f"Quick benchmark test failed: {e}")
        return False


def check_build_exclusion():
    """Check that experiments are excluded from build"""
    logger.info("Checking build configuration...")

    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

        if pyproject_path.exists():
            with open(pyproject_path) as f:
                content = f.read()

            # Check that experiments is in excludes
            if '"experiments/",' in content or "**/experiments/" in content:
                logger.info("✓ Experiments directory excluded from build")
                return True
            else:
                logger.warning("⚠ Experiments directory may not be excluded from build")
                return False
        else:
            logger.error("pyproject.toml not found")
            return False

    except Exception as e:
        logger.error(f"Build configuration check failed: {e}")
        return False


def run_all_tests():
    """Run all tests and return summary"""
    logger.info("=" * 60)
    logger.info("SECURE FL EXPERIMENTS SETUP TEST")
    logger.info("=" * 60)

    tests = [
        ("Project Structure", check_project_structure),
        ("Package Import", check_package_import),
        ("Model Creation", test_model_creation),
        ("DatasetManager", test_dataset_manager),
        ("Benchmark Script", test_benchmark_script),
        ("Quick Benchmark", test_quick_benchmark),
        ("Build Exclusion", check_build_exclusion),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = 0
    total = len(tests)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:20s}: {status}")
        if result:
            passed += 1

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Experiments setup is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the setup.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
