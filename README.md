# 🔐 Secure FL: Dual-Verifiable Federated Learning with Zero-Knowledge Proofs

This repository contains a complete implementation of a secure federated learning framework that uses **dual zero-knowledge proof verification** to ensure both client-side training correctness and server-side aggregation integrity.

## 🎯 Key Features

- **🛡️ Dual ZKP Verification**: Client-side zk-STARKs + Server-side zk-SNARKs
- **🚀 FedJSCM Aggregation**: Momentum-based federated aggregation for improved convergence
- **📊 Dynamic Proof Rigor**: Adaptive proof complexity based on training stability
- **🔗 Blockchain Integration**: On-chain verification for public auditability
- **📈 Comprehensive Experiments**: Built-in benchmarking and visualization tools

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client 1      │    │   Client 2      │    │   Client N      │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Local Training│ │    │ │Local Training│ │    │ │Local Training│ │
│ │ + zk-STARK  │ │    │ │ + zk-STARK  │ │    │ │ + zk-STARK  │ │
│ │   Proof     │ │    │ │   Proof     │ │    │ │   Proof     │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     FL Server           │
                    │ ┌─────────────────────┐ │
                    │ │ FedJSCM Aggregation │ │
                    │ │   + zk-SNARK Proof  │ │
                    │ │ + Stability Monitor │ │
                    │ └─────────────────────┘ │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Blockchain Verifier   │
                    │ ┌─────────────────────┐ │
                    │ │  Smart Contract     │ │
                    │ │ Proof Verification  │ │
                    │ └─────────────────────┘ │
                    └─────────────────────────┘
```

## 📌 Contributors
- [@krishantt](https://github.com/krishantt) - Krishant Timilsina
- [@bigya01](https://github.com/bigya01) - Bindu Paudel

---

## 📂 Repository Structure
```
secure-fl/
├── 📁 docs/              # Research papers and documentation
│   ├── concept-note/     # Initial concept and motivation
│   ├── project-proposal/ # Detailed project proposal
│   └── proposal-defense/ # Defense materials
├── 📁 fl/               # Core federated learning implementation
│   ├── server.py        # FL server with FedJSCM and ZKP integration
│   ├── client.py        # FL client with zk-STARK proof generation
│   ├── aggregation.py   # FedJSCM momentum-based aggregation
│   ├── proof_manager.py # ZKP proof generation and verification
│   ├── stability_monitor.py # Dynamic proof rigor adjustment
│   ├── quantization.py  # Parameter quantization for circuits
│   └── utils.py         # Utility functions
├── 📁 proofs/           # Zero-knowledge proof circuits
│   ├── client/          # zk-STARK circuits (Cairo)
│   │   └── sgd_full_trace.cairo
│   └── server/          # zk-SNARK circuits (Circom)
│       └── fedjscm_aggregation.circom
├── 📁 blockchain/       # Smart contracts for verification
│   └── FLVerifier.sol   # Solidity contract for proof verification
├── 📁 experiments/      # Experiment scripts and configs
│   ├── train_secure_fl.py # Main training experiment
│   └── config.yaml      # Experiment configuration
├── 📁 k8s/             # Kubernetes deployment manifests
├── 📁 infra/           # Infrastructure as Code configs
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js (for Circom/SnarkJS)
- Cairo compiler (for zk-STARKs)
- CUDA-capable GPU (optional, for acceleration)

### Installation

#### Option 1: Install from PyPI (Recommended)

```bash
# Install the package
pip install secure-fl

# Setup ZKP tools (optional but recommended)
secure-fl setup zkp

# Run a quick demo
secure-fl demo
```

#### Option 2: Install with PDM (For Development)

```bash
# Clone the repository
git clone https://github.com/krishantt/secure-fl.git
cd secure-fl

# Install PDM if you don't have it
pip install pdm

# Install dependencies
pdm install

# Setup ZKP tools
pdm run setup-zkp

# Run tests
pdm run test
```

#### Option 3: Install from Source

```bash
# Clone the repository
git clone https://github.com/krishantt/secure-fl.git
cd secure-fl

# Install in development mode
pip install -e .

# Setup environment
secure-fl setup full
```

### Research and Development

For research purposes, the project includes a comprehensive experiments directory with multi-dataset benchmarking:

```bash
# Run multi-dataset benchmark (development only)
cd secure-fl
python experiments/benchmark.py --datasets mnist cifar10 synthetic

# Quick benchmark
python experiments/benchmark.py --quick --configs baseline_iid

# See experiments/README.md for full documentation
```

**Note:** The `experiments/` directory is excluded from package distribution and contains standalone research scripts.

### Basic Usage

#### Command Line Interface

```bash
# Run a quick demo
secure-fl demo

# Run a federated learning experiment
secure-fl experiment --num-clients 3 --rounds 5 --dataset synthetic

# Start a server
secure-fl server --rounds 10 --enable-zkp

# Connect a client
secure-fl client --client-id client_1 --dataset mnist

# Check system requirements
secure-fl setup check
```

#### Python API

```python
from secure_fl import SecureFlowerServer, create_client, create_server_strategy
import torch.nn as nn

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    
    def forward(self, x):
        return self.fc(x.flatten(1))

# Create server
strategy = create_server_strategy(
    model_fn=lambda: MyModel(),
    enable_zkp=True,
    proof_rigor="medium"
)
server = SecureFlowerServer(strategy=strategy)

# Create clients
client = create_client(
    client_id="client_1",
    model_fn=lambda: MyModel(),
    train_data=your_train_data,
    enable_zkp=True
)
```

## 🔬 Technical Details

### Core Algorithms

#### FedJSCM Aggregation
Our momentum-based aggregation algorithm:

```
m^{(t+1)} = γ × m^{(t)} + Σ(p_i × Δ_i)
w^{(t+1)} = w^{(t)} + m^{(t+1)}
```

Where:
- `m^{(t)}` is server momentum at round t
- `γ` is momentum coefficient (0.9 by default)
- `p_i` are client weights (proportional to data size)
- `Δ_i` are client parameter updates

#### Dynamic Proof Rigor
The system automatically adjusts proof complexity based on training stability:

- **High Rigor**: Full SGD trace verification (early rounds, unstable training)
- **Medium Rigor**: Single-step verification (moderate stability)
- **Low Rigor**: Delta norm verification (stable/converged training)

### Zero-Knowledge Proof Systems

#### Client-side zk-STARKs
- **Language**: Cairo
- **Purpose**: Prove correct local SGD training
- **Features**: 
  - Post-quantum secure
  - Transparent (no trusted setup)
  - Scalable verification

#### Server-side zk-SNARKs
- **Scheme**: Groth16
- **Purpose**: Prove correct FedJSCM aggregation
- **Features**:
  - Succinct proofs (~200 bytes)
  - Fast verification
  - Blockchain-compatible

### Security Guarantees

1. **Training Integrity**: Clients cannot submit invalid parameter updates
2. **Aggregation Correctness**: Server cannot manipulate aggregation process
3. **Data Privacy**: No raw data is revealed, only computational correctness
4. **Public Auditability**: All proofs can be verified on-chain

## 📊 Experimental Results

### Performance Benchmarks

| Configuration | Proof Time | Verification Time | Communication Overhead |
|---------------|------------|-------------------|----------------------|
| High Rigor    | ~2.3s      | ~0.05s           | +15%                |
| Medium Rigor  | ~0.8s      | ~0.02s           | +8%                 |
| Low Rigor     | ~0.3s      | ~0.01s           | +3%                 |

### Accuracy Comparison

| Method | MNIST | CIFAR-10 | MedMNIST |
|--------|-------|----------|----------|
| Standard FL | 0.95 | 0.78 | 0.82 |
| Secure FL (Ours) | 0.94 | 0.76 | 0.81 |
| Overhead | -1% | -2.6% | -1.2% |

## 🛠️ Advanced Usage

### Custom Model Integration

```python
from secure_fl import create_server_strategy, SecureFlowerServer
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your model definition
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Your forward pass
        x = torch.relu(self.conv1(x))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Create server strategy
strategy = create_server_strategy(
    model_fn=lambda: MyCustomModel(),
    enable_zkp=True,
    proof_rigor="medium"
)

# Start server
server = SecureFlowerServer(strategy=strategy, num_rounds=20)
server.start()
```

### Blockchain Deployment

```solidity
// Deploy the FLVerifier contract
contract MyFLVerifier is FLVerifier {
    constructor() FLVerifier(
        3,  // min clients per round
        300,  // proof timeout (seconds)  
        0x1234...  // STARK verifying key hash
    ) {}
}
```

### Parameter Sweeps

```yaml
# config.yaml
parameter_sweep:
  enabled: true
  parameters:
    momentum: [0.5, 0.7, 0.9, 0.95]
    proof_rigor: ["low", "medium", "high"]
    num_clients: [3, 5, 10]
```

## 🔧 Configuration Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_clients` | int | 5 | Number of federated clients |
| `num_rounds` | int | 10 | Training rounds |
| `enable_zkp` | bool | true | Enable zero-knowledge proofs |
| `proof_rigor` | str | "high" | Proof complexity level |
| `momentum` | float | 0.9 | FedJSCM momentum coefficient |
| `blockchain_verification` | bool | false | Enable on-chain verification |

### ZKP Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantization_bits` | int | 8 | Bits for parameter quantization |
| `max_trace_length` | int | 1024 | Maximum STARK trace length |
| `circuit_size` | int | 1000 | SNARK circuit constraint count |
| `proof_timeout` | int | 120 | Proof generation timeout (seconds) |

## 🐛 Troubleshooting

### Common Issues

1. **Cairo Compilation Errors**
   ```bash
   # Ensure Cairo is properly installed
   cairo-compile --version
   # Reinstall if needed
   pip uninstall cairo-lang && pip install cairo-lang
   ```

2. **Circom Circuit Compilation**
   ```bash
   # Check Circom installation
   circom --version
   # Compile circuits manually
   cd proofs/server
   circom fedjscm_aggregation.circom --r1cs --wasm --sym
   ```

3. **Memory Issues with Large Models**
   ```yaml
   # Reduce model/circuit size in config.yaml
   model:
     hidden_dim: 64  # Reduce from default 128
   zkp:
     client_proof:
       max_trace_length: 512  # Reduce from 1024
   ```

4. **Client Connection Timeouts**
   ```yaml
   # Increase timeouts
   networking:
     client_timeout: 600  # Increase from 300
     max_retries: 5       # Increase from 3
   ```

## 📈 Monitoring and Visualization

### Built-in Metrics

The framework automatically tracks:
- Training convergence (loss, accuracy)
- Proof generation/verification times
- Communication overhead
- Client participation rates
- Model parameter stability
- Resource utilization

### Custom Metrics

```python
from secure_fl import StabilityMonitor

monitor = StabilityMonitor()
# Add custom metrics
monitor.update(parameters, round_num, custom_metrics={
    "gradient_norm": grad_norm,
    "privacy_budget": epsilon,
    "custom_score": score
})
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Using PDM (recommended)
pdm install -d
pdm run test
pdm run format
pdm run lint

# Using pip
pip install -e ".[dev]"
pytest
black secure_fl/
isort secure_fl/
mypy secure_fl/
```

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{timilsina2024secure,
  title={Dual-Verifiable Framework for Federated Learning using Zero-Knowledge Proofs},
  author={Timilsina, Krishant and Paudel, Bindu},
  year={2024},
  institution={Tribhuvan University, Institute of Engineering}
}
```

## 📦 Package Information

### PyPI Installation

```bash
pip install secure-fl
```

### Development Installation

```bash
git clone https://github.com/krishantt/secure-fl.git
cd secure-fl
pdm install -d
```

### Available Extras

- `dev`: Development dependencies (pytest, black, mypy, etc.)
- `medical`: Medical dataset support (medmnist, nibabel, etc.)  
- `notebook`: Jupyter notebook support
- `quantization`: Advanced quantization tools
- `blockchain`: Blockchain integration tools
- `all`: All optional dependencies

Example: `pip install "secure-fl[dev,medical,notebook]"`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Flower](https://flower.dev/) for the federated learning framework
- [StarkWare](https://starkware.co/) for Cairo and STARK technology
- [iden3](https://iden3.io/) for Circom and zk-SNARK tools
- Our supervisor, Dr. Arun Kumar Timalsina, for guidance and support
- Tribhuvan University, Institute of Engineering, Pulchowk Campus

---

**⚠️ Note**: This is a research prototype. For production use, additional security audits and optimizations are recommended.

**📫 Contact**: For questions or collaborations, reach out to [krishantt@example.com](mailto:krishantt@example.com) or [bigya01@example.com](mailto:bigya01@example.com)
