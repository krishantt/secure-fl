# Training Examples for Secure Federated Learning

This document provides examples and usage instructions for the updated `train.py` script that uses real datasets and models from the project instead of synthetic data.

## Features

The updated training script now supports:
- **Real Datasets**: MNIST, CIFAR-10, MedMNIST, and synthetic data
- **Proper Models**: Dataset-specific models (MNISTModel, CIFAR10Model, SimpleModel)
- **Advanced Data Loading**: New FederatedDataLoader class for unified dataset handling
- **Federated Partitioning**: Automatic IID and Non-IID data partitioning across clients
- **ZKP Integration**: Optional zero-knowledge proof verification
- **FedJSCM Aggregation**: Momentum-based federated aggregation
- **Comprehensive Metrics**: Detailed training metrics and results

## Basic Usage

### Quick Start

Run a basic federated learning experiment with MNIST:

```bash
cd secure-fl
python -m secure_fl.experiments.train --dataset mnist --num-clients 3 --rounds 5
```

### Command Line Options

```bash
python -m secure_fl.experiments.train [OPTIONS]

Options:
  --config CONFIG_FILE           Path to YAML configuration file
  --num-clients INTEGER         Number of federated clients (default: 5)
  --rounds INTEGER              Number of training rounds (default: 10)
  --dataset {mnist,cifar10,medmnist,synthetic}  Dataset to use (default: mnist)
  --enable-zkp                  Enable zero-knowledge proof verification
  --local-epochs INTEGER        Number of local training epochs (default: 3)
  --batch-size INTEGER          Batch size for training (default: 32)
  --learning-rate FLOAT         Learning rate (default: 0.01)
```

## Dataset Examples

### MNIST Handwritten Digits

```bash
# Basic MNIST training with 3 clients
python -m secure_fl.experiments.train \
    --dataset mnist \
    --num-clients 3 \
    --rounds 10 \
    --local-epochs 3 \
    --batch-size 64

# Output:
# Dataset: MNIST handwritten digits (28x28 grayscale)
# Model: MNISTModel (109,770 parameters)
# Automatic data partitioning across clients
```

### CIFAR-10 Natural Images

```bash
# CIFAR-10 training with smaller batch size (GPU memory intensive)
python -m secure_fl.experiments.train \
    --dataset cifar10 \
    --num-clients 5 \
    --rounds 15 \
    --local-epochs 2 \
    --batch-size 32 \
    --learning-rate 0.001

# Output:
# Dataset: CIFAR-10 natural images (32x32 RGB)
# Model: CIFAR10Model (653,194 parameters)
# Note: Downloads ~170MB on first run
```

### MedMNIST Medical Images

```bash
# Medical image classification (requires medmnist package)
python -m secure_fl.experiments.train \
    --dataset medmnist \
    --num-clients 4 \
    --rounds 8 \
    --local-epochs 3 \
    --batch-size 32

# Note: Requires: pip install medmnist
# Uses PathMNIST dataset by default
```

### Synthetic Data (for testing)

```bash
# Fast synthetic data for development/testing
python -m secure_fl.experiments.train \
    --dataset synthetic \
    --num-clients 2 \
    --rounds 3 \
    --local-epochs 1 \
    --batch-size 64
```

## Advanced Examples

### With Zero-Knowledge Proofs

```bash
# Enable ZKP verification for enhanced security
python -m secure_fl.experiments.train \
    --dataset mnist \
    --num-clients 3 \
    --rounds 5 \
    --enable-zkp \
    --local-epochs 2

# Features:
# - Client-side zk-STARK proof generation
# - Server-side zk-SNARK verification  
# - Cryptographic integrity guarantees
```

### High-Performance Configuration

```bash
# Optimized for faster training
python -m secure_fl.experiments.train \
    --dataset mnist \
    --num-clients 5 \
    --rounds 20 \
    --local-epochs 1 \
    --batch-size 128 \
    --learning-rate 0.05
```

### Large-Scale Simulation

```bash
# Simulate many clients (may require more memory)
python -m secure_fl.experiments.train \
    --dataset cifar10 \
    --num-clients 10 \
    --rounds 25 \
    --local-epochs 1 \
    --batch-size 16 \
    --learning-rate 0.001
```

## Configuration File Usage

Create a YAML configuration file for complex setups:

```yaml
# config.yaml
num_clients: 8
num_rounds: 20
dataset: "mnist"
enable_zkp: true
proof_rigor: "high"
local_epochs: 2
batch_size: 64
learning_rate: 0.01
```

Run with configuration:

```bash
python -m secure_fl.experiments.train --config config.yaml
```

## Output and Results

### Console Output

The script provides real-time progress information:

```
INFO: Dataset info: {'input_shape': (1, 28, 28), 'num_classes': 10, ...}
INFO: Model info: {'total_parameters': 109770, 'model_class': 'MNISTModel'}
INFO: Round 1/10 completed: Acc=0.875, Loss=0.421, Time=2.34s
...
============================================================
SECURE FL EXPERIMENT RESULTS
============================================================
Dataset: mnist
Model: MNISTModel  
Clients: 3
Rounds: 10
Final Accuracy: 0.9234
Total Time: 45.67s
Total Parameters: 109770
ZKP Enabled: False
Results saved to: results/training_results_mnist_3clients_20231203_143022.json
============================================================
```

### Results File

Detailed metrics are saved to `results/training_results_*.json`:

```json
{
  "training_history": [
    {
      "round": 1,
      "accuracy": 0.875,
      "loss": 0.421,
      "time": 2.34,
      "client_metrics": {
        "client_0": {"loss": 0.415, "accuracy": 0.881, "samples": 12000},
        "client_1": {"loss": 0.427, "accuracy": 0.869, "samples": 12000}
      }
    }
  ],
  "final_accuracy": 0.9234,
  "total_time": 45.67,
  "dataset_info": {...},
  "model_info": {...}
}
```

## Performance Considerations

### Memory Usage

- **MNIST**: ~50MB per client
- **CIFAR-10**: ~200MB per client  
- **MedMNIST**: ~100MB per client
- **Synthetic**: ~10MB per client

### Training Time (approximate)

| Dataset   | Clients | Rounds | Time per Round | Total Time |
|-----------|---------|---------|---------------|------------|
| MNIST     | 3       | 10      | 2-3s          | 30s        |
| CIFAR-10  | 3       | 10      | 25-30s        | 5-6 min    |
| MedMNIST  | 3       | 10      | 3-4s          | 40s        |
| Synthetic | 5       | 10      | 0.1s          | 2s         |

### Optimization Tips

1. **Batch Size**: Larger batches = faster training, more memory usage
2. **Local Epochs**: More epochs = better convergence, longer training
3. **Number of Clients**: More clients = more realistic FL, slower aggregation
4. **Dataset Choice**: Start with synthetic for debugging, then move to real datasets

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   --batch-size 16
   ```

2. **Dataset Download Fails**:
   ```bash
   # Check internet connection, datasets auto-download on first use
   # MNIST: ~60MB, CIFAR-10: ~170MB
   ```

3. **MedMNIST Not Available**:
   ```bash
   pip install medmnist
   ```

4. **Low Accuracy**:
   ```bash
   # Increase training rounds and local epochs
   --rounds 20 --local-epochs 5
   ```

### Debug Mode

For development and debugging, use synthetic data:

```bash
python -m secure_fl.experiments.train \
    --dataset synthetic \
    --num-clients 2 \
    --rounds 2 \
    --local-epochs 1
```

## New FederatedDataLoader Architecture

### Core Components

The training system now uses a modular data loading architecture:

- **FederatedDataLoader**: Main class in `secure_fl.data.dataloader`
  - Unified interface for multiple dataset types
  - Automatic downloading and preprocessing
  - IID and Non-IID partitioning strategies
  - Client-specific data distribution

- **Data Utilities**: Helper functions in `secure_fl.data.utils`
  - Dataset information and statistics
  - Data transformations and augmentations
  - Compatibility validation

### Usage Example

```python
from secure_fl.data import FederatedDataLoader

# Create federated data loader
loader = FederatedDataLoader(
    dataset_name="mnist",
    num_clients=5,
    iid=False,  # Use Non-IID partitioning
    val_split=0.2,
    batch_size=32
)

# Get client data loaders
client_loaders = loader.create_client_dataloaders()

# Get dataset information
info = loader.get_dataset_info()
stats = loader.get_client_stats()
```

## Integration with Other Components

The updated training script integrates with:

- **Data Loading**: Uses FederatedDataLoader for advanced dataset handling
- **Aggregation**: Uses FedJSCM momentum-based aggregation
- **Client Management**: Creates secure FL clients with proper data partitioning
- **Model Registry**: Automatically selects appropriate models for datasets
- **Proof Systems**: Optional ZKP integration for security
- **Monitoring**: Comprehensive metrics collection and logging

## Exploring the Data Loading System

### Demo Script

Run the comprehensive data loading demo:

```bash
python examples/data_loader_demo.py
```

This demonstrates:
- Basic FederatedDataLoader usage
- Different dataset types (MNIST, CIFAR-10, Synthetic)
- IID vs Non-IID partitioning comparison  
- PyTorch DataLoader integration
- Advanced configuration options

### Custom Data Integration

To add new datasets:

1. **Extend FederatedDataLoader**: Add new dataset loading methods
2. **Update Dataset Info**: Add metadata in `data/utils.py`
3. **Create Transforms**: Define appropriate preprocessing
4. **Test Integration**: Use the demo script to validate

## Next Steps

After running experiments:

1. **Analyze Results**: Check the generated JSON files in `results/`
2. **Tune Hyperparameters**: Experiment with different configurations
3. **Scale Up**: Try larger numbers of clients and rounds
4. **Enable Security**: Add `--enable-zkp` for production scenarios
5. **Custom Datasets**: Extend the FederatedDataLoader for your data
6. **Explore Partitioning**: Test different IID/Non-IID strategies

For more advanced usage, see the main project documentation and explore the individual modules in `secure_fl/`.