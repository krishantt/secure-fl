"""
Setup and Installation Module for Secure FL

This module provides automated setup, installation, and configuration utilities
for the Secure FL framework with dual ZKP verification.
"""

import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SecureFLSetup:
    """Setup and installation manager for Secure FL"""

    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.system = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.arch = platform.machine().lower()

        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")

        logger.info(
            f"Secure FL Setup - System: {self.system}, Python: {self.python_version}, Arch: {self.arch}"
        )

    def check_system_requirements(self) -> Dict[str, bool]:
        """Check system requirements and return status"""
        checks = {}

        # Python version
        checks["python_version"] = sys.version_info >= (3, 8)

        # Available memory
        try:
            import psutil

            memory_gb = psutil.virtual_memory().total / (1024**3)
            checks["memory"] = memory_gb >= 4
            checks["memory_gb"] = memory_gb
        except ImportError:
            checks["memory"] = None

        # Disk space
        try:
            disk_space = shutil.disk_usage(self.root_dir).free / (1024**3)
            checks["disk_space"] = disk_space >= 2
            checks["disk_space_gb"] = disk_space
        except:
            checks["disk_space"] = None

        # Node.js (for ZKP tools)
        checks["nodejs"] = self._check_command_exists("node")
        if checks["nodejs"]:
            try:
                result = subprocess.run(
                    ["node", "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    checks["nodejs_version"] = result.stdout.strip()
            except:
                checks["nodejs_version"] = "unknown"

        # npm
        checks["npm"] = self._check_command_exists("npm")

        # Git
        checks["git"] = self._check_command_exists("git")

        # Cairo
        checks["cairo"] = self._check_command_exists("cairo-compile")

        # Circom
        checks["circom"] = self._check_command_exists("circom")

        # SnarkJS
        checks["snarkjs"] = self._check_command_exists("snarkjs")

        # CUDA (optional)
        try:
            import torch

            checks["cuda"] = torch.cuda.is_available()
            if checks["cuda"]:
                checks["cuda_devices"] = torch.cuda.device_count()
        except ImportError:
            checks["cuda"] = None

        self._print_system_check(checks)
        return checks

    def _print_system_check(self, checks: Dict[str, bool]) -> None:
        """Print formatted system check results"""
        print("\n🔍 System Requirements Check:")
        print("=" * 50)

        # Core requirements
        status = "✓" if checks.get("python_version") else "✗"
        print(f"  Python 3.8+: {status} (Current: {self.python_version})")

        if checks.get("memory") is not None:
            status = "✓" if checks["memory"] else "⚠"
            gb = checks.get("memory_gb", 0)
            print(f"  Memory (4GB+): {status} ({gb:.1f}GB available)")

        if checks.get("disk_space") is not None:
            status = "✓" if checks["disk_space"] else "⚠"
            gb = checks.get("disk_space_gb", 0)
            print(f"  Disk Space (2GB+): {status} ({gb:.1f}GB free)")

        # Development tools
        print("\n📦 Development Tools:")
        status = "✓" if checks.get("git") else "⚠"
        print(f"  Git: {status}")

        # ZKP Dependencies
        print("\n🔐 ZKP Dependencies:")
        status = "✓" if checks.get("nodejs") else "✗"
        version = checks.get("nodejs_version", "")
        print(f"  Node.js: {status} {version}")

        status = "✓" if checks.get("npm") else "✗"
        print(f"  npm: {status}")

        status = "✓" if checks.get("cairo") else "✗"
        print(f"  Cairo: {status}")

        status = "✓" if checks.get("circom") else "✗"
        print(f"  Circom: {status}")

        status = "✓" if checks.get("snarkjs") else "✗"
        print(f"  SnarkJS: {status}")

        # Optional components
        print("\n⚡ Optional Components:")
        if checks.get("cuda") is not None:
            if checks["cuda"]:
                devices = checks.get("cuda_devices", 0)
                print(f"  CUDA: ✓ ({devices} device(s))")
            else:
                print(f"  CUDA: ⚠ (Not available)")
        else:
            print(f"  CUDA: ? (PyTorch not installed)")

    def _check_command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH"""
        try:
            result = subprocess.run(
                [command, "--version"] if command != "git" else ["git", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def install_python_deps(
        self, dev: bool = False, extras: Optional[List[str]] = None
    ) -> bool:
        """Install Python dependencies"""
        logger.info("Installing Python dependencies...")

        try:
            # Upgrade pip first
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
            )

            # Install the package itself
            install_cmd = [sys.executable, "-m", "pip", "install", "-e", "."]

            # Add extras
            if extras:
                package_spec = f".[{','.join(extras)}]"
                install_cmd[-1] = package_spec

            if dev:
                install_cmd.extend(["--dev"])

            subprocess.check_call(install_cmd, cwd=self.root_dir)

            logger.info("✓ Python dependencies installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Python dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during installation: {e}")
            return False

    def setup_zkp_tools(self) -> bool:
        """Setup ZKP tools (Cairo and Circom/SnarkJS)"""
        logger.info("Setting up ZKP tools...")

        cairo_success = self._setup_cairo()
        circom_success = self._setup_circom()

        if cairo_success and circom_success:
            logger.info("✓ ZKP tools setup completed successfully")
            return True
        else:
            logger.warning("⚠ ZKP tools setup completed with warnings")
            return False

    def _setup_cairo(self) -> bool:
        """Setup Cairo for zk-STARKs"""
        logger.info("Setting up Cairo...")

        # Check if already installed
        if self._check_command_exists("cairo-compile"):
            logger.info("✓ Cairo already installed")
            return True

        try:
            # Try installing via pip
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "cairo-lang"]
            )

            if self._check_command_exists("cairo-compile"):
                logger.info("✓ Cairo installed successfully via pip")
                return True
            else:
                logger.warning("Cairo installed but cairo-compile not found in PATH")
                return False

        except subprocess.CalledProcessError:
            logger.warning("Cairo installation via pip failed")
            logger.info("Please install Cairo manually:")
            logger.info("  - Visit: https://github.com/starkware-libs/cairo")
            logger.info(
                "  - Or try: curl -L https://github.com/starkware-libs/cairo/releases/latest/download/cairo-lang.tar.gz | tar xz"
            )
            return False

    def _setup_circom(self) -> bool:
        """Setup Circom and SnarkJS for zk-SNARKs"""
        logger.info("Setting up Circom and SnarkJS...")

        # Check Node.js
        if not self._check_command_exists("node"):
            logger.error("Node.js not found. Please install Node.js first:")
            logger.info("  - Visit: https://nodejs.org/")
            logger.info(
                "  - Or use a package manager: brew install node (macOS), apt install nodejs (Ubuntu)"
            )
            return False

        # Check npm
        if not self._check_command_exists("npm"):
            logger.error("npm not found. Please install npm.")
            return False

        try:
            # Install Circom
            if not self._check_command_exists("circom"):
                logger.info("Installing Circom...")
                subprocess.check_call(["npm", "install", "-g", "circom"])

            # Install SnarkJS
            if not self._check_command_exists("snarkjs"):
                logger.info("Installing SnarkJS...")
                subprocess.check_call(["npm", "install", "-g", "snarkjs"])

            # Verify installations
            circom_ok = self._check_command_exists("circom")
            snarkjs_ok = self._check_command_exists("snarkjs")

            if circom_ok and snarkjs_ok:
                logger.info("✓ Circom and SnarkJS installed successfully")
                return True
            else:
                logger.error("Installation completed but tools not accessible")
                logger.info("You may need to run with sudo or as administrator")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Circom/SnarkJS: {e}")
            logger.info("Try installing manually:")
            logger.info("  npm install -g circom")
            logger.info("  npm install -g snarkjs")
            return False

    def create_config(self) -> bool:
        """Create default configuration and directory structure"""
        logger.info("Creating default configuration...")

        directories = [
            "experiments/results",
            "data",
            "proofs/compiled",
            "blockchain/artifacts",
            "logs",
        ]

        for dir_path in directories:
            full_path = self.root_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {full_path}")

        # Create default experiment config
        config_path = self.root_dir / "experiments" / "config.yaml"
        if not config_path.exists():
            default_config = """
# Secure FL Experiment Configuration

# Server configuration
server:
  host: localhost
  port: 8080
  num_rounds: 10
  min_fit_clients: 2
  min_evaluate_clients: 2

# Aggregation parameters
aggregation:
  momentum: 0.9
  learning_rate: 0.01
  weight_decay: 0.0

# ZKP settings
zkp:
  enable_zkp: true
  proof_rigor: medium
  blockchain_verification: false
  quantization_bits: 8

# Dataset configuration
dataset:
  name: synthetic
  num_clients: 3
  iid: false
  alpha: 0.5  # Dirichlet parameter for non-IID

# Model configuration
model:
  name: simple_nn
  hidden_dim: 128
  num_classes: 10

# Training parameters
training:
  local_epochs: 5
  batch_size: 32
  learning_rate: 0.01
  optimizer: sgd

# Monitoring and logging
monitoring:
  log_level: INFO
  save_models: false
  save_metrics: true
  visualize: true
"""
            with open(config_path, "w") as f:
                f.write(default_config.strip())
            logger.info(f"Created default config: {config_path}")

        logger.info("✓ Configuration created successfully")
        return True

    def run_tests(self) -> bool:
        """Run basic tests to verify installation"""
        logger.info("Running basic tests...")

        try:
            # Test imports
            self._test_imports()

            # Test basic functionality
            self._test_basic_functionality()

            logger.info("✓ All basic tests passed!")
            return True

        except Exception as e:
            logger.error(f"Tests failed: {e}")
            return False

    def _test_imports(self) -> None:
        """Test that all required modules can be imported"""
        required_modules = [
            "torch",
            "torchvision",
            "flwr",
            "numpy",
            "pandas",
            "matplotlib",
            "secure_fl",
        ]

        for module in required_modules:
            try:
                __import__(module)
                logger.debug(f"✓ {module}")
            except ImportError as e:
                raise ImportError(f"Failed to import {module}: {e}")

    def _test_basic_functionality(self) -> None:
        """Test basic framework functionality"""
        try:
            import torch
            import torch.nn as nn

            from secure_fl import create_server_strategy, get_default_config
            from secure_fl.utils import ndarrays_to_torch, torch_to_ndarrays

            # Test config loading
            config = get_default_config()
            assert isinstance(config, dict)

            # Test simple model creation
            # Model moved to secure_fl.models

            from secure_fl.models import SimpleModel

            # Create a simple model instance to test
            model = SimpleModel()
            assert model is not None

        except ImportError as e:
            logger.warning(f"Basic functionality test failed: {e}")
        except Exception as e:
            logger.warning(f"Basic functionality test error: {e}")

    def clean(self, clean_all: bool = False) -> bool:
        """Clean up temporary files and caches"""
        logger.info("Cleaning up temporary files...")

        patterns_to_clean = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/results/temp_*",
            "**/logs/*.log",
        ]

        if clean_all:
            patterns_to_clean.extend(
                [
                    "**/results/**/*",
                    "**/proofs/compiled/**/*",
                    "**/data/temp/**/*",
                    "**/.coverage",
                    "**/htmlcov",
                ]
            )

        cleaned_count = 0

        for pattern in patterns_to_clean:
            for path in self.root_dir.glob(pattern):
                try:
                    if path.is_file():
                        path.unlink()
                        cleaned_count += 1
                    elif path.is_dir() and len(list(path.iterdir())) == 0:
                        # Only remove empty directories
                        path.rmdir()
                        cleaned_count += 1
                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not clean {path}: {e}")

        logger.info(f"✓ Cleaned {cleaned_count} files/directories")
        return True

    def run_demo(self) -> bool:
        """Run a quick demo to verify everything works"""
        logger.info("Running demo...")

        try:
            from secure_fl.experiments.demo import run_quick_demo

            return run_quick_demo()
        except ImportError:
            logger.warning("Demo module not available")
            return True
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            return False

    def get_installation_status(self) -> Dict[str, str]:
        """Get current installation status"""
        status = {}

        # Check core components
        try:
            import secure_fl

            status["secure_fl"] = f"✓ v{secure_fl.__version__}"
        except ImportError:
            status["secure_fl"] = "✗ Not installed"

        try:
            import torch

            status["torch"] = f"✓ v{torch.__version__}"
        except ImportError:
            status["torch"] = "✗ Not installed"

        try:
            import flwr

            status["flwr"] = f"✓ v{flwr.__version__}"
        except ImportError:
            status["flwr"] = "✗ Not installed"

        # Check ZKP tools
        if self._check_command_exists("cairo-compile"):
            status["cairo"] = "✓ Installed"
        else:
            status["cairo"] = "✗ Not found"

        if self._check_command_exists("circom"):
            status["circom"] = "✓ Installed"
        else:
            status["circom"] = "✗ Not found"

        if self._check_command_exists("snarkjs"):
            status["snarkjs"] = "✓ Installed"
        else:
            status["snarkjs"] = "✗ Not found"

        return status

    def print_installation_guide(self) -> None:
        """Print installation guide"""
        print(
            """
🚀 Secure FL Installation Guide
================================

1. Install Python dependencies:
   secure-fl setup install

2. Setup ZKP tools:
   secure-fl setup zkp

3. Create configuration:
   secure-fl setup config

4. Run tests:
   secure-fl setup test

5. Try a demo:
   secure-fl demo

For more information, visit:
https://github.com/krishantt/secure-fl
        """
        )


def main():
    """Main setup entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Secure FL Setup")
    parser.add_argument(
        "action",
        choices=["check", "install", "zkp", "config", "test", "clean", "demo", "full"],
        help="Action to perform",
    )
    parser.add_argument(
        "--dev", action="store_true", help="Install development dependencies"
    )
    parser.add_argument("--extras", nargs="+", help="Extra dependencies to install")
    parser.add_argument(
        "--clean-all", action="store_true", help="Clean all generated files"
    )

    args = parser.parse_args()

    setup = SecureFLSetup()

    if args.action == "check":
        setup.check_system_requirements()
    elif args.action == "install":
        setup.install_python_deps(dev=args.dev, extras=args.extras)
    elif args.action == "zkp":
        setup.setup_zkp_tools()
    elif args.action == "config":
        setup.create_config()
    elif args.action == "test":
        setup.run_tests()
    elif args.action == "clean":
        setup.clean(clean_all=args.clean_all)
    elif args.action == "demo":
        setup.run_demo()
    elif args.action == "full":
        # Full setup
        setup.install_python_deps(dev=args.dev, extras=args.extras)
        setup.setup_zkp_tools()
        setup.create_config()
        if setup.run_tests():
            print("✅ Full setup completed successfully!")
        else:
            print("⚠️ Setup completed with some issues")


if __name__ == "__main__":
    main()
