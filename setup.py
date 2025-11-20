#!/usr/bin/env python3
"""
Setup and Deployment Script for Secure Federated Learning Framework

This script provides automated setup, installation, and deployment capabilities
for the Secure FL framework with dual ZKP verification.

Usage:
    python setup.py install        # Install dependencies
    python setup.py setup-zkp      # Setup ZKP tools (Cairo, Circom)
    python setup.py test           # Run basic tests
    python setup.py demo           # Run a quick demo
    python setup.py clean          # Clean up temporary files
"""

import os
import sys
import subprocess
import shutil
import logging
import argparse
from pathlib import Path
import platform
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SecureFLSetup:
    """Setup and deployment manager for Secure FL"""

    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.system = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")

        logger.info(
            f"Secure FL Setup - System: {self.system}, Python: {self.python_version}"
        )

    def install_python_deps(self):
        """Install Python dependencies"""
        logger.info("Installing Python dependencies...")

        try:
            # Upgrade pip first
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
            )

            # Install requirements
            requirements_file = self.root_dir / "requirements.txt"
            if requirements_file.exists():
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(requirements_file),
                    ]
                )
            else:
                # Install core dependencies manually
                core_deps = [
                    "torch>=2.0.0",
                    "torchvision>=0.15.0",
                    "flwr>=1.5.0",
                    "numpy>=1.24.0",
                    "scikit-learn>=1.3.0",
                    "pandas>=2.0.0",
                    "matplotlib>=3.7.0",
                    "seaborn>=0.12.0",
                    "tqdm>=4.65.0",
                    "pyyaml>=6.0",
                    "web3>=6.0.0",
                    "py-ecc>=6.0.0",
                    "psutil>=5.9.0",
                    "fastapi>=0.100.0",
                    "uvicorn>=0.23.0",
                ]

                for dep in core_deps:
                    try:
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", dep]
                        )
                    except subprocess.CalledProcessError:
                        logger.warning(f"Failed to install {dep}, continuing...")

            logger.info("✓ Python dependencies installed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to install Python dependencies: {e}")
            return False

    def setup_zkp_tools(self):
        """Setup ZKP tools (Cairo and Circom)"""
        logger.info("Setting up ZKP tools...")

        success = True

        # Setup Cairo
        if not self._setup_cairo():
            logger.warning("Cairo setup failed - client zk-STARKs won't work")
            success = False

        # Setup Circom and SnarkJS
        if not self._setup_circom():
            logger.warning("Circom/SnarkJS setup failed - server zk-SNARKs won't work")
            success = False

        if success:
            logger.info("✓ ZKP tools setup completed successfully")
        else:
            logger.warning("⚠ ZKP tools setup completed with warnings")

        return success

    def _setup_cairo(self):
        """Setup Cairo for zk-STARKs"""
        logger.info("Setting up Cairo...")

        try:
            # Check if Cairo is already installed
            result = subprocess.run(
                ["cairo-compile", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info("✓ Cairo already installed")
                return True
        except FileNotFoundError:
            pass

        try:
            # Install Cairo via pip (if available)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "cairo-lang"]
            )
            logger.info("✓ Cairo installed via pip")
            return True

        except subprocess.CalledProcessError:
            logger.warning("Cairo installation via pip failed")
            logger.info(
                "Please install Cairo manually from: https://github.com/starkware-libs/cairo"
            )
            return False

    def _setup_circom(self):
        """Setup Circom and SnarkJS for zk-SNARKs"""
        logger.info("Setting up Circom and SnarkJS...")

        try:
            # Check if Node.js is installed
            result = subprocess.run(
                ["node", "--version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error("Node.js not found. Please install Node.js first.")
                return False

            # Check if npm is available
            result = subprocess.run(
                ["npm", "--version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error("npm not found. Please install npm.")
                return False

            # Install Circom and SnarkJS globally
            try:
                subprocess.check_call(["npm", "install", "-g", "circom"])
                subprocess.check_call(["npm", "install", "-g", "snarkjs"])
                logger.info("✓ Circom and SnarkJS installed successfully")
                return True

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install Circom/SnarkJS: {e}")
                logger.info("You may need to run with sudo or as administrator")
                return False

        except Exception as e:
            logger.error(f"Circom setup failed: {e}")
            return False

    def run_tests(self):
        """Run basic tests to verify installation"""
        logger.info("Running basic tests...")

        try:
            test_script = self.root_dir / "test_implementation.py"
            if test_script.exists():
                result = subprocess.run(
                    [sys.executable, str(test_script)], capture_output=True, text=True
                )

                if result.returncode == 0:
                    logger.info("✓ All tests passed!")
                    return True
                else:
                    logger.error("✗ Some tests failed:")
                    logger.error(result.stdout)
                    logger.error(result.stderr)
                    return False
            else:
                logger.warning("Test script not found, skipping tests")
                return False

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False

    def run_demo(self):
        """Run a quick demo to showcase functionality"""
        logger.info("Running Secure FL demo...")

        try:
            demo_script = self.root_dir / "experiments" / "train_secure_fl.py"
            if demo_script.exists():
                # Run a minimal demo
                cmd = [
                    sys.executable,
                    str(demo_script),
                    "--num-clients",
                    "3",
                    "--rounds",
                    "3",
                    "--dataset",
                    "synthetic",
                    "--enable-zkp=false",  # Disable ZKP for quick demo
                ]

                logger.info("Starting demo (this may take a few minutes)...")
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )

                if result.returncode == 0:
                    logger.info("✓ Demo completed successfully!")
                    logger.info("Check the experiments/results/ directory for outputs")
                    return True
                else:
                    logger.error("✗ Demo failed:")
                    logger.error(result.stderr)
                    return False
            else:
                logger.error("Demo script not found")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Demo timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error(f"Demo execution failed: {e}")
            return False

    def clean(self):
        """Clean up temporary files and caches"""
        logger.info("Cleaning up temporary files...")

        patterns_to_clean = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/results/temp_*",
            "**/proofs/**/*_compiled.json",
            "**/proofs/**/witness",
            "**/proofs/**/circuit.wasm",
            "**/proofs/**/circuit_js",
        ]

        cleaned_count = 0

        for pattern in patterns_to_clean:
            for path in self.root_dir.glob(pattern):
                try:
                    if path.is_file():
                        path.unlink()
                        cleaned_count += 1
                    elif path.is_dir():
                        shutil.rmtree(path)
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Could not clean {path}: {e}")

        logger.info(f"✓ Cleaned {cleaned_count} files/directories")
        return True

    def create_config(self):
        """Create default configuration files"""
        logger.info("Creating default configuration...")

        # Create experiments directory if it doesn't exist
        experiments_dir = self.root_dir / "experiments"
        experiments_dir.mkdir(exist_ok=True)

        # Create results directory
        results_dir = experiments_dir / "results"
        results_dir.mkdir(exist_ok=True)

        # Create data directory
        data_dir = self.root_dir / "data"
        data_dir.mkdir(exist_ok=True)

        logger.info("✓ Directory structure created")
        return True

    def check_system_requirements(self):
        """Check system requirements and dependencies"""
        logger.info("Checking system requirements...")

        checks = []

        # Python version
        if sys.version_info >= (3, 8):
            checks.append("✓ Python 3.8+: OK")
        else:
            checks.append("✗ Python 3.8+: FAILED")

        # Available memory
        try:
            import psutil

            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb >= 4:
                checks.append(f"✓ Memory: {memory_gb:.1f}GB (OK)")
            else:
                checks.append(f"⚠ Memory: {memory_gb:.1f}GB (Recommended: 4GB+)")
        except:
            checks.append("? Memory: Unknown")

        # Disk space
        try:
            disk_space = shutil.disk_usage(self.root_dir).free / (1024**3)
            if disk_space >= 2:
                checks.append(f"✓ Disk space: {disk_space:.1f}GB (OK)")
            else:
                checks.append(f"⚠ Disk space: {disk_space:.1f}GB (Recommended: 2GB+)")
        except:
            checks.append("? Disk space: Unknown")

        # Node.js (for ZKP tools)
        try:
            result = subprocess.run(
                ["node", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                checks.append(f"✓ Node.js: {version}")
            else:
                checks.append("✗ Node.js: Not found")
        except:
            checks.append("✗ Node.js: Not found")

        # Git (for cloning dependencies)
        try:
            result = subprocess.run(
                ["git", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                checks.append("✓ Git: Available")
            else:
                checks.append("⚠ Git: Not available")
        except:
            checks.append("⚠ Git: Not available")

        logger.info("System Requirements Check:")
        for check in checks:
            logger.info(f"  {check}")

        return True


def main():
    parser = argparse.ArgumentParser(description="Secure FL Setup and Deployment")
    parser.add_argument(
        "command",
        nargs="?",
        default="help",
        choices=[
            "install",
            "setup-zkp",
            "test",
            "demo",
            "clean",
            "check",
            "config",
            "help",
        ],
        help="Command to execute",
    )

    args = parser.parse_args()

    if args.command == "help":
        parser.print_help()
        print("\nCommands:")
        print("  install   - Install Python dependencies")
        print("  setup-zkp - Setup ZKP tools (Cairo, Circom)")
        print("  test      - Run basic functionality tests")
        print("  demo      - Run a quick demonstration")
        print("  clean     - Clean temporary files")
        print("  check     - Check system requirements")
        print("  config    - Create default configurations")
        return

    try:
        setup = SecureFLSetup()

        if args.command == "install":
            success = setup.install_python_deps()
        elif args.command == "setup-zkp":
            success = setup.setup_zkp_tools()
        elif args.command == "test":
            success = setup.run_tests()
        elif args.command == "demo":
            success = setup.run_demo()
        elif args.command == "clean":
            success = setup.clean()
        elif args.command == "check":
            success = setup.check_system_requirements()
        elif args.command == "config":
            success = setup.create_config()
        else:
            logger.error(f"Unknown command: {args.command}")
            success = False

        if success:
            logger.info("✓ Command completed successfully")
        else:
            logger.error("✗ Command failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
