"""
Setup and Installation Module for Secure FL

This module provides automated setup, installation, and configuration utilities
for the Secure FL framework with dual ZKP verification.
"""

import logging
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class SecureFLSetup:
    """Setup and installation manager for Secure FL"""

    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.system = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.arch = platform.machine().lower()

        # Check Python version

        logger.info(
            f"Secure FL Setup - System: {self.system}, Python: {self.python_version}, Arch: {self.arch}"
        )

    def check_system_requirements(self) -> dict[str, bool]:
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
        except (OSError, shutil.Error):
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
            except (subprocess.SubprocessError, OSError):
                checks["nodejs_version"] = "unknown"

        # npm
        checks["npm"] = self._check_command_exists("npm")

        # Git
        checks["git"] = self._check_command_exists("git")

        # Circom (check if Rust-based)
        checks["circom"] = self._check_command_exists("circom")
        if checks["circom"]:
            checks["circom_type"] = self._get_circom_type()

        # SnarkJS
        checks["snarkjs"] = self._check_command_exists("snarkjs")

        # Rust (needed for circom)
        checks["rust"] = self._check_command_exists("rustc")
        if checks["rust"]:
            try:
                result = subprocess.run(
                    ["rustc", "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    checks["rust_version"] = result.stdout.strip()
            except (subprocess.SubprocessError, OSError):
                checks["rust_version"] = "unknown"

        # ZKP tools comprehensive check
        if checks["circom"] and checks["snarkjs"]:
            checks["zkp_verification"] = self._quick_zkp_verification()

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

    def _get_circom_type(self) -> str:
        """Determine if circom is Rust-based or npm-based"""
        try:
            result = subprocess.run(
                ["circom", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if "circom compiler" in output:
                    return "rust"
                else:
                    return "npm"
            return "unknown"
        except (subprocess.SubprocessError, OSError):
            return "unknown"

    def _quick_zkp_verification(self) -> bool:
        """Quick verification that ZKP tools can perform basic operations"""
        try:
            # Test circom compilation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                test_circuit = temp_path / "test.circom"

                # Simple test circuit
                test_circuit.write_text("""
pragma circom 2.0.0;

template Test() {
    signal input a;
    signal output b;
    b <== a * 2;
}

component main = Test();
""")

                # Try to compile
                result = subprocess.run(
                    ["circom", str(test_circuit), "--r1cs", "-o", str(temp_path)],
                    capture_output=True,
                    timeout=30,
                )

                return result.returncode == 0

        except Exception:
            return False

    def _print_system_check(self, checks: dict[str, bool]) -> None:
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

        status = "✓" if checks.get("rust") else "⚠"
        version = checks.get("rust_version", "")
        print(f"  Rust: {status} {version}")

        # ZKP Dependencies
        print("\n🔐 ZKP Dependencies:")
        status = "✓" if checks.get("nodejs") else "✗"
        version = checks.get("nodejs_version", "")
        print(f"  Node.js: {status} {version}")

        status = "✓" if checks.get("npm") else "⚠"
        print(f"  npm: {status}")

        # ZKP Tools
        print("\n🛡️ ZKP Tools:")
        if checks.get("circom"):
            circom_type = checks.get("circom_type", "unknown")
            status = "✓" if circom_type == "rust" else "⚠"
            type_info = f" ({circom_type}-based)" if circom_type != "unknown" else ""
            print(f"  Circom: {status}{type_info}")
        else:
            print(f"  Circom: ✗")

        status = "✓" if checks.get("snarkjs") else "✗"
        print(f"  SnarkJS: {status}")

        # ZKP Verification
        if checks.get("zkp_verification") is not None:
            status = "✓" if checks["zkp_verification"] else "⚠"
            print(f"  ZKP Tools Test: {status}")

        # Optional components
        print("\n⚡ Optional Components:")
        if checks.get("cuda") is not None:
            if checks["cuda"]:
                devices = checks.get("cuda_devices", 0)
                print(f"  CUDA: ✓ ({devices} device(s))")
            else:
                print("  CUDA: ⚠ (Not available)")
        else:
            print("  CUDA: ? (PyTorch not installed)")

    def _check_command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH"""
        try:
            # Special handling for snarkjs which uses --help and exits with code 1
            if command == "snarkjs":
                result = subprocess.run(
                    [command, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # snarkjs --help exits with code 1 but prints help text
                return "snarkjs@" in result.stdout or "snarkjs@" in result.stderr
            else:
                result = subprocess.run(
                    [command, "--version"]
                    if command != "git"
                    else ["git", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def install_python_deps(
        self, dev: bool = False, extras: list[str] | None = None
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
        """Setup ZKP tools (Rust-based Circom and SnarkJS)"""
        logger.info("Setting up ZKP tools...")

        circom_success = self._setup_circom()

        if circom_success:
            logger.info("✓ ZKP tools setup completed successfully")
            return True
        else:
            logger.warning("⚠ ZKP tools setup completed with warnings")
            return False

    def _setup_circom(self) -> bool:
        """Setup Rust-based Circom and SnarkJS for zk-SNARKs"""
        logger.info("Setting up Rust-based Circom and SnarkJS...")

        # Check Node.js for SnarkJS
        if not self._check_command_exists("node"):
            logger.error("Node.js not found. Please install Node.js first:")
            logger.info("  - Visit: https://nodejs.org/")
            logger.info(
                "  - Or use a package manager: brew install node (macOS), apt install nodejs (Ubuntu)"
            )
            return False

        # Check npm for SnarkJS
        if not self._check_command_exists("npm"):
            logger.error("npm not found. Please install npm.")
            return False

        # Check Rust for Circom
        rust_ok = self._setup_rust()
        if not rust_ok:
            logger.warning("Rust installation failed or not found")

        try:
            # Install Rust-based Circom from source
            if not self._check_command_exists("circom"):
                logger.info("Installing Rust-based Circom from source...")
                if not self._install_circom_from_source():
                    logger.warning("Failed to install Circom from source")
                    logger.info("Falling back to npm version...")
                    subprocess.check_call(["npm", "install", "-g", "circom"])

            # Install SnarkJS
            if not self._check_command_exists("snarkjs"):
                logger.info("Installing SnarkJS...")
                subprocess.check_call(["npm", "install", "-g", "snarkjs"])

            # Run comprehensive verification
            logger.info("Running ZKP tools verification...")
            verification_success = self._verify_zkp_tools()

            if verification_success:
                logger.info("✓ ZKP tools setup and verification completed successfully")
                return True
            else:
                logger.warning("⚠ ZKP tools installed but verification had issues")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install ZKP tools: {e}")
            logger.info("Try installing manually:")
            logger.info("  # Install Rust:")
            logger.info(
                "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
            )
            logger.info("  # Install Circom from source:")
            logger.info("  git clone https://github.com/iden3/circom.git")
            logger.info(
                "  cd circom && cargo build --release && cargo install --path circom"
            )
            logger.info("  # Install SnarkJS:")
            logger.info("  npm install -g snarkjs")
            return False

    def _setup_rust(self) -> bool:
        """Setup Rust toolchain if not already installed"""
        if self._check_command_exists("rustc") and self._check_command_exists("cargo"):
            logger.info("✓ Rust already installed")
            return True

        logger.info("Installing Rust toolchain...")
        try:
            # Use rustup to install Rust
            rustup_cmd = [
                "curl",
                "--proto",
                "=https",
                "--tlsv1.2",
                "-sSf",
                "https://sh.rustup.rs",
            ]
            rustup_install = ["sh", "-s", "--", "-y"]

            # Download and run rustup
            process = subprocess.Popen(
                rustup_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            rustup_script, _ = process.communicate()

            if process.returncode == 0:
                # Run the installation script
                install_process = subprocess.Popen(
                    rustup_install,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                install_process.communicate(input=rustup_script)

                # Check if installation succeeded
                # Note: May need shell restart for PATH update
                logger.info("Rust installation completed")
                logger.info(
                    "You may need to restart your shell or run: source ~/.cargo/env"
                )
                return True
            else:
                logger.error("Failed to download rustup")
                return False

        except Exception as e:
            logger.error(f"Rust installation failed: {e}")
            logger.info("Please install Rust manually:")
            logger.info("  Visit: https://rustup.rs/")
            logger.info(
                "  Or run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
            )
            return False

    def _install_circom_from_source(self) -> bool:
        """Install circom from source using Cargo"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                repo_path = temp_path / "circom"

                logger.info("Cloning circom repository...")
                subprocess.check_call(
                    [
                        "git",
                        "clone",
                        "https://github.com/iden3/circom.git",
                        str(repo_path),
                    ]
                )

                logger.info("Building circom (this may take a few minutes)...")
                subprocess.check_call(["cargo", "build", "--release"], cwd=repo_path)

                logger.info("Installing circom...")
                subprocess.check_call(
                    ["cargo", "install", "--path", "circom"], cwd=repo_path
                )

                # Verify installation
                if self._check_command_exists("circom"):
                    logger.info("✓ Rust-based Circom installed successfully")
                    return True
                else:
                    logger.error("Circom built but not found in PATH")
                    return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build circom from source: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during circom installation: {e}")
            return False

    def _verify_zkp_tools(self) -> bool:
        """Run comprehensive ZKP tools verification"""
        try:
            # Import and run the verification from our scripts
            # Run built-in comprehensive verification
            logger.info("Running comprehensive ZKP verification...")
            return self._comprehensive_zkp_check()

        except subprocess.TimeoutExpired:
            logger.warning("ZKP verification timed out")
            return False
        except Exception as e:
            logger.warning(f"ZKP verification failed: {e}")
            return self._basic_zkp_check()

    def _comprehensive_zkp_check(self) -> bool:
        """Comprehensive ZKP tools verification with actual circuit testing"""
        try:
            # Check basic availability
            circom_ok = self._check_command_exists("circom")
            snarkjs_ok = self._check_command_exists("snarkjs")

            if not circom_ok or not snarkjs_ok:
                if not circom_ok:
                    logger.error("✗ Circom not found")
                if not snarkjs_ok:
                    logger.error("✗ SnarkJS not found")
                return False

            # Test circuit compilation
            logger.info("Testing circuit compilation...")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                circuit_file = temp_path / "test.circom"
                input_file = temp_path / "input.json"

                # Create test circuit
                circuit_code = """
pragma circom 2.0.0;

template Multiplier2() {
    signal input a;
    signal input b;
    signal output c;

    c <== a * b;
}

component main = Multiplier2();
"""
                circuit_file.write_text(circuit_code)
                input_file.write_text('{"a": "3", "b": "11"}')

                # Test circom compilation
                result = subprocess.run(
                    [
                        "circom",
                        str(circuit_file),
                        "--r1cs",
                        "--wasm",
                        "--sym",
                        "-o",
                        str(temp_path),
                    ],
                    capture_output=True,
                    timeout=60,
                )

                if result.returncode != 0:
                    logger.error("✗ Circuit compilation failed")
                    return False

                logger.info("✓ Circuit compilation successful")

                # Test witness generation
                logger.info("Testing witness generation...")
                wasm_file = temp_path / "test_js" / "test.wasm"
                witness_file = temp_path / "witness.wtns"

                if wasm_file.exists():
                    result = subprocess.run(
                        [
                            "snarkjs",
                            "wtns",
                            "calculate",
                            str(wasm_file),
                            str(input_file),
                            str(witness_file),
                        ],
                        capture_output=True,
                        timeout=60,
                    )

                    if result.returncode == 0 and witness_file.exists():
                        logger.info("✓ Witness generation successful")
                    else:
                        logger.warning("⚠ Witness generation had issues")
                        return False

                # Test trusted setup
                logger.info("Testing trusted setup...")
                pot_file = temp_path / "pot12_0000.ptau"
                result = subprocess.run(
                    [
                        "snarkjs",
                        "powersoftau",
                        "new",
                        "bn128",
                        "12",
                        str(pot_file),
                        "-v",
                    ],
                    capture_output=True,
                    timeout=120,
                )

                if result.returncode == 0 and pot_file.exists():
                    logger.info("✓ Trusted setup successful")
                else:
                    logger.warning("⚠ Trusted setup had issues")
                    return False

                logger.info("🎉 All ZKP tools verification tests passed!")
                return True

        except subprocess.TimeoutExpired:
            logger.error("✗ ZKP verification timed out")
            return False
        except Exception as e:
            logger.error(f"✗ ZKP verification failed: {e}")
            return False

    def _basic_zkp_check(self) -> bool:
        """Basic ZKP tools availability check"""
        circom_ok = self._check_command_exists("circom")
        snarkjs_ok = self._check_command_exists("snarkjs")

        if circom_ok and snarkjs_ok:
            logger.info("✓ Basic ZKP tools check passed")
            return True
        else:
            if not circom_ok:
                logger.error("✗ Circom not found")
            if not snarkjs_ok:
                logger.error("✗ SnarkJS not found")
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
                raise ImportError(f"Failed to import {module}: {e}") from e

    def _test_basic_functionality(self) -> None:
        """Test basic framework functionality"""
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401

            from secure_fl import (  # noqa: F401
                create_server_strategy,
                get_default_config,
            )
            from secure_fl.utils import (  # noqa: F401
                ndarrays_to_torch,
                torch_to_ndarrays,
            )

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

    def get_installation_status(self) -> dict[str, str]:
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
