# Installation Guide for Secure FL

This guide provides comprehensive installation instructions for the Secure FL framework, a dual-verifiable federated learning system with zero-knowledge proofs.

## Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.9+ recommended)
- **Node.js 18+** (for ZKP tools)
- **Rust 1.75+** (for circom compilation)
- **Git** (for development)
- **4GB+ RAM** (8GB+ recommended)
- **2GB+ free disk space**

### Option 1: Install from PyPI (Recommended)

```bash
# Install the package
pip install secure-fl

# Setup ZKP tools (required for full functionality)
# First install Rust
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
source ~/.cargo/env

# Install circom from source
git clone https://github.com/iden3/circom.git
cd circom && cargo build --release && cargo install --path circom

# Install snarkjs
npm install -g snarkjs

# Verify ZKP tools installation
uv run python -m secure_fl.setup check

# Run a quick demo
secure-fl demo
```

### Option 2: Install with UV (For Development)

```bash
# Clone the repository
git clone https://github.com/krishantt/secure-fl.git
cd secure-fl

# Install UV if you don't have it
pip install uv

# Install dependencies
uv sync --all-extras --dev

# Setup ZKP tools
# Install Rust
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
source ~/.cargo/env

# Install circom from source
git clone https://github.com/iden3/circom.git /tmp/circom
cd /tmp/circom && cargo build --release && cargo install --path circom

# Install snarkjs
npm install -g snarkjs

# Verify ZKP tools installation
uv run python -m secure_fl.setup check

# Run tests
uv run pytest

# Run demo
uv run python -m secure_fl.cli demo
```

### Option 3: Install from Source

```bash
# Clone the repository
git clone https://github.com/krishantt/secure-fl.git
cd secure-fl

# Install in development mode
pip install -e .

# Setup environment
secure-fl setup full
```

## Detailed Installation

### 1. System Requirements

Check your system meets the requirements:

```bash
secure-fl setup check
```

This will verify:
- Python version and availability
- System memory and disk space
- Required development tools
- ZKP dependencies status

### 2. Core Dependencies

Install core Python dependencies:

```bash
# Basic installation
pip install secure-fl

# With optional dependencies
pip install "secure-fl[dev,medical,notebook]"

# All optional dependencies
pip install "secure-fl[all]"
```

### 3. Zero-Knowledge Proof Tools

The framework uses external ZKP tools that need separate installation:

#### Cairo (for client-side zk-STARKs)

```bash
# Automatic installation
secure-fl setup zkp

# Manual installation
pip install cairo-lang

# Verify installation
cairo-compile --version
```

#### Circom & SnarkJS (for server-side zk-SNARKs)

```bash
# Prerequisites: Node.js and npm
node --version  # Should be 16+
npm --version

# Install globally
npm install -g circom snarkjs

# Verify installation
circom --version
snarkjs help
```

### 4. Optional Dependencies

#### Medical Datasets

```bash
pip install "secure-fl[medical]"
```

#### Jupyter Notebook Support

```bash
pip install "secure-fl[notebook]"
```

#### Development Tools

```bash
pip install "secure-fl[dev]"
```

#### Advanced Quantization

```bash
pip install "secure-fl[quantization]"
```

#### Blockchain Integration

```bash
pip install "secure-fl[blockchain]"
```

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# System dependencies
sudo apt update
sudo apt install python3-dev python3-pip nodejs npm git

# Install secure-fl
pip3 install secure-fl

# Setup ZKP tools
secure-fl setup zkp
```

### macOS

```bash
# Using Homebrew
brew install python nodejs git

# Install secure-fl
pip install secure-fl

# Setup ZKP tools
secure-fl setup zkp
```

### Windows

```bash
# Using Chocolatey (run as Administrator)
choco install python nodejs git

# Or use Python installer + Node.js installer from official websites

# Install secure-fl
pip install secure-fl

# Setup ZKP tools
secure-fl setup zkp
```

### Docker

```bash
# Build Docker image
docker build -t secure-fl .

# Run container
docker run -it secure-fl secure-fl demo
```

## Development Installation

For developers who want to contribute or modify the framework:

### Using PDM (Recommended)

```bash
# Clone repository
git clone https://github.com/krishantt/secure-fl.git
cd secure-fl

# Install PDM
pip install pdm

# Install with all development dependencies
pdm install -d

# Install pre-commit hooks
pdm run pre-commit install

# Run tests
pdm run test

# Format code
pdm run format
```

### Using pip

```bash
# Clone repository
git clone https://github.com/krishantt/secure-fl.git
cd secure-fl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Setup ZKP tools
secure-fl setup zkp

# Run tests
pytest
```

## Verification

After installation, verify everything works:

```bash
# Check installation status
secure-fl info

# Run system check
secure-fl setup check

# Run quick demo
secure-fl demo

# Run comprehensive test
secure-fl setup test
```

## Troubleshooting

### Common Issues

#### 1. Python Version Issues

```bash
# Check Python version
python --version

# If using older Python, install newer version
# Ubuntu/Debian: sudo apt install python3.9
# macOS: brew install python@3.9
# Windows: Download from python.org
```

#### 2. Node.js/npm Issues

```bash
# Check Node.js version
node --version

# Update Node.js if needed
# Ubuntu/Debian: curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
# macOS: brew install node
# Windows: Download from nodejs.org
```

#### 3. Cairo Installation Issues

```bash
# If Cairo installation fails via pip
# Try installing from source:
git clone https://github.com/starkware-libs/cairo.git
cd cairo
pip install .
```

#### 4. Circom/SnarkJS Installation Issues

```bash
# If global installation fails, try local installation
npm install circom snarkjs

# Or use npx to run without global installation
npx circom --version
npx snarkjs help
```

#### 5. Memory Issues

For systems with limited memory:

```bash
# Install with minimal dependencies
pip install secure-fl --no-deps
pip install torch torchvision flwr numpy

# Disable ZKP for testing
secure-fl experiment --disable-zkp
```

#### 6. Permission Issues

```bash
# On Unix systems, if you get permission errors:
pip install --user secure-fl

# Or use virtual environment (recommended):
python -m venv venv
source venv/bin/activate
pip install secure-fl
```

### Getting Help

If you encounter issues:

1. **Check system requirements**: `secure-fl setup check`
2. **View logs**: Check terminal output for error messages
3. **GitHub Issues**: https://github.com/krishantt/secure-fl/issues
4. **Documentation**: https://github.com/krishantt/secure-fl/blob/main/README.md

### Clean Installation

If you need to start fresh:

```bash
# Remove existing installation
pip uninstall secure-fl

# Clean pip cache
pip cache purge

# Reinstall
pip install secure-fl

# Reset configuration
secure-fl setup clean --all
```

## Next Steps

After successful installation:

1. **Run Demo**: `secure-fl demo`
2. **Try Experiments**: `secure-fl experiment --help`
3. **Start Server**: `secure-fl server --help`
4. **Connect Clients**: `secure-fl client --help`
5. **Read Documentation**: Check README.md and docs/

## Updates

To update to the latest version:

```bash
# Update from PyPI
pip install --upgrade secure-fl

# Update from source (development)
cd secure-fl
git pull
pdm install
```

## Uninstallation

To completely remove Secure FL:

```bash
# Uninstall Python package
pip uninstall secure-fl

# Remove ZKP tools (optional)
npm uninstall -g circom snarkjs
pip uninstall cairo-lang

# Remove data and config (optional)
rm -rf ~/.secure-fl
rm -rf ./secure_fl_data
```
