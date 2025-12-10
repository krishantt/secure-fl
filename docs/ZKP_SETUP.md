# ZKP Tools Setup Guide

This guide explains how to set up Zero-Knowledge Proof (ZKP) tools for the Secure FL project, specifically the Rust-based circom compiler and snarkjs.

## Overview

The Secure FL project uses two main ZKP tools:
- **Circom**: A circuit compiler written in Rust for creating zero-knowledge circuits
- **SnarkJS**: A JavaScript library for generating and verifying zk-SNARK proofs

## Prerequisites

### System Requirements
- **Node.js** 18 or higher (for snarkjs)
- **Rust** 1.75 or higher (for circom)
- **Git** (for cloning repositories)
- **Build tools** (gcc, make, etc.)

### Operating System Support
- ✅ Linux (Ubuntu, Debian, CentOS, etc.) - **Full CI/CD support**
- ✅ macOS (Intel and Apple Silicon) - **Full CI/CD support**
- ⚠️ Windows (manual setup only) - **Limited support, not tested in CI**

## Installation Methods

### Method 1: Automatic Installation (Recommended)

Use the built-in CLI setup command:

```bash
# Check system requirements and ZKP tools
uv run python -m secure_fl.cli setup --action check

# Automatically install ZKP tools
uv run python -m secure_fl.cli setup --action zkp

# Full setup (Python deps + ZKP tools + config)
uv run python -m secure_fl.cli setup --action full
```

The CLI setup command provides integrated installation and verification with detailed diagnostics.

### Method 2: Manual Installation

#### Step 1: Install Rust

```bash
# Install Rust using rustup (Linux/macOS)
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh

# Restart your shell or run:
source ~/.cargo/env
```

For Windows, download and run the installer from [rustup.rs](https://rustup.rs/).

#### Step 2: Install Circom (Rust-based)

```bash
# Clone the circom repository
git clone https://github.com/iden3/circom.git
cd circom

# Build and install circom
cargo build --release
cargo install --path circom

# Verify installation
circom --help
```

#### Step 3: Install Node.js

Install Node.js 18+ from [nodejs.org](https://nodejs.org/) or use a version manager:

```bash
# Using nvm (Linux/macOS)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18

# Using fnm (cross-platform)
cargo install fnm
fnm install 18
fnm use 18
```

#### Step 4: Install SnarkJS

```bash
# Install snarkjs globally
npm install -g snarkjs

# Verify installation
snarkjs --help
```

### Method 3: Docker Installation

Use the provided Docker setup that includes all tools:

```bash
# Build the Docker image
docker build -t secure-fl .

# Or use docker-compose
docker-compose build secure-fl-dev

# Run verification in container
docker run --rm secure-fl uv run python -m secure_fl.setup check
```

## Verification

After installation, verify that everything works:

```bash
# Quick verification
circom --version
snarkjs --version

# Comprehensive verification
uv run python -m secure_fl.setup check
```

The setup verification will:
1. Check that both tools are installed and accessible
2. Test basic circuit compilation with circom
3. Test witness generation with snarkjs
4. Test trusted setup generation

## Usage Examples

### Basic Circuit Compilation

```bash
# Create a simple circuit file (multiplier.circom)
cat > multiplier.circom << 'EOF'
pragma circom 2.0.0;

template Multiplier2() {
    signal private input a;
    signal private input b;
    signal output c;
    
    c <== a * b;
}

component main = Multiplier2();
EOF

# Compile the circuit
circom multiplier.circom --r1cs --wasm --sym
```

### Witness Generation

```bash
# Create input file
echo '{"a": "3", "b": "11"}' > input.json

# Generate witness
cd multiplier_js
snarkjs wtns calculate multiplier.wasm ../input.json witness.wtns
```

### Trusted Setup (Powers of Tau)

```bash
# Start powers of tau ceremony
snarkjs powersoftau new bn128 12 pot12_0000.ptau -v

# Contribute to the ceremony
snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau --name="First contribution" -v
```

## Troubleshooting

### Common Issues

#### 1. Circom not found after installation
```bash
# Make sure Rust's bin directory is in your PATH
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### 2. Permission errors during installation
```bash
# Use sudo for global npm installation if needed
sudo npm install -g snarkjs

# Or install locally and add to PATH
npm install snarkjs
export PATH="./node_modules/.bin:$PATH"
```

#### 3. Build errors on Ubuntu/Debian
```bash
# Install required build dependencies
sudo apt-get update
sudo apt-get install build-essential pkg-config libssl-dev
```

#### 4. Build errors on macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Or install via Homebrew
brew install pkg-config openssl
```

#### 5. Windows-specific issues
- **Recommended**: Use Windows Subsystem for Linux (WSL2) for full compatibility
- Install Visual Studio Build Tools if compiling natively on Windows
- Use PowerShell or Command Prompt as Administrator for npm installations
- **Note**: Windows builds are not automatically tested in CI - manual verification required

### Performance Tips

1. **Use SSD storage** for faster compilation
2. **Allocate sufficient RAM** (4GB+ recommended for large circuits)
3. **Use release builds** (`cargo build --release`) for production
4. **Cache compiled circuits** to avoid recompilation

### Environment Variables

Set these environment variables for optimal performance:

```bash
# Rust compilation settings
export CARGO_TARGET_DIR=/tmp/cargo-target
export RUSTFLAGS="-C target-cpu=native"

# Node.js settings
export NODE_OPTIONS="--max-old-space-size=8192"
```

## Integration with Secure FL

The Secure FL project automatically detects and uses these tools when available. Key integration points:

1. **Circuit compilation**: Automatic compilation of `.circom` files in `proofs/` directory
2. **Witness generation**: Automated during proof generation process  
3. **Trusted setup**: Managed through the FL protocol setup
4. **Proof verification**: Integrated with the FL aggregation process

### Configuration

Configure ZKP tools in your Secure FL config:

```python
# In your secure_fl config
zkp_config = {
    "circom_path": "circom",  # Will auto-detect if in PATH
    "snarkjs_path": "snarkjs",  # Will auto-detect if in PATH
    "circuit_dir": "./proofs/circuits/",
    "setup_dir": "./proofs/setup/",
}
```

## Getting Help

- **Documentation**: [Circom Docs](https://docs.circom.io/), [SnarkJS Docs](https://github.com/iden3/snarkjs)
- **Issues**: Report installation issues in the Secure FL repository
- **Community**: Join discussions in the project's community channels

## Advanced Setup

### Custom Circuit Libraries

```bash
# Clone additional circuit libraries
git clone https://github.com/iden3/circomlib.git
export CIRCOM_LIBRARY_PATH="./circomlib/circuits"
```

### Performance Optimizations

```bash
# Use system-wide trusted setup files
export SNARKJS_PTAU_PATH="/usr/local/share/snarkjs/ptau"

# Enable parallel compilation
export CIRCOM_PARALLEL_JOBS=$(nproc)
```

### Development Environment

For development, consider using the development Docker container:

```bash
# Start development environment
docker-compose run --rm secure-fl-dev bash

# All tools are pre-installed and configured
circom --help
snarkjs --help
uv run python -m secure_fl.setup check
```
