# Secure FL Experiments

This directory contains experiments, benchmarks, and training scripts for the Secure FL framework. These files are **NOT bundled** with the package distribution and are intended for development, testing, and research purposes.

## 📁 Directory Contents

- [`benchmark.py`](./benchmark.py) - Multi-dataset benchmarking suite
- [`demo.py`](./demo.py) - Quick demonstration scripts
- [`train.py`](./train.py) - Full training experiments
- [`__init__.py`](./__init__.py) - Experiments package metadata

## 🚀 Quick Start

### Prerequisites

Make sure you have Secure FL installed:

```bash
pip install secure-fl
# or from source
cd secure-fl && pip install -e .
```

### Running Experiments

#### 1. Multi-Dataset Benchmark

Run comprehensive benchmarks across multiple datasets with their corresponding models:

```bash
# Run all datasets with default configurations
python benchmark.py

# Run specific datasets
python benchmark.py --datasets mnist cifar10 fashion_mnist

# Run different categories
python benchmark.py --datasets synthetic_small text_classification medical

# List all available datasets
python benchmark.py --list-datasets

# Quick benchmark (fewer rounds)
python benchmark.py --quick

# Run specific configurations
python benchmark.py --configs baseline_iid zkp_medium_rigor

# Custom output directory
python benchmark.py --output-dir ./my_benchmark_results

# List all available configurations
python benchmark.py --list-configs
```

**Note:** The benchmark runs as a standalone script only. It is NOT integrated into the `secure-fl` CLI commands since experiments are not bundled with the package distribution.

**Supported Datasets & Models:**

**📂 Image Datasets:**
- **MNIST**: `MNISTModel` (fully connected) → MNIST handwritten digits (28x28 grayscale)
- **Fashion-MNIST**: `MNISTModel` (enhanced) → Fashion-MNIST clothing items (28x28 grayscale)  
- **CIFAR-10**: `CIFAR10Model` (CNN) → CIFAR-10 natural images (32x32 RGB)
- **CIFAR-100**: `CIFAR10Model` (CNN, 100 classes) → CIFAR-100 natural images (32x32 RGB, 100 classes)

**📂 Synthetic Datasets:**
- **Synthetic**: `SimpleModel` (basic MLP) → Synthetic data for basic testing
- **Synthetic Large**: `FlexibleMLP` (deep) → Large synthetic dataset with 20 classes
- **Synthetic Small**: `FlexibleMLP` (compact) → Small synthetic dataset for quick testing

**📂 Domain-Specific Datasets:**
- **Text Classification**: `FlexibleMLP` (GELU) → Simulated text classification (4 sentiment classes)
- **Medical**: `FlexibleMLP` (medical) → Simulated medical diagnosis (3 disease classes)
- **Financial**: `FlexibleMLP` (fraud) → Simulated financial fraud detection (binary)

**Total: 10 datasets with 5 different model architectures**

#### 2. Quick Demo

```bash
# Basic federated learning demo (standalone)
python demo.py

# Or use CLI (if running from development environment)
secure-fl demo
```

#### 3. Full Training Experiment

```bash
# Full training with configuration (standalone)
python train.py --config config.yaml

# Command line parameters
python train.py --num-clients 5 --rounds 10 --dataset mnist --enable-zkp
```

## 📊 Benchmark Features

The enhanced benchmark system provides:

### Multi-Dataset Support
- Automatic model selection based on dataset
- Proper data preprocessing and splitting
- IID and Non-IID data distributions

### Performance Metrics
- **Accuracy convergence** across rounds
- **Training time** analysis
- **Communication overhead** measurement
- **ZKP proof times** (simulated)
- **Memory usage** tracking

### Visualizations
- **Accuracy comparison** across configurations
- **ZKP overhead analysis** by rigor level
- **Convergence plots** for accuracy and loss
- **Model complexity vs performance**

### Configurations

Default benchmark configurations:

| Configuration | Clients | Rounds | ZKP | Rigor | Distribution | Description |
|---------------|---------|---------|-----|-------|--------------|-------------|
| `baseline_iid` | 3 | 5 | ❌ | - | IID | IID baseline without ZKP |
| `baseline_non_iid` | 3 | 5 | ❌ | - | Non-IID | Non-IID baseline without ZKP |
| `zkp_high_rigor` | 3 | 5 | ✅ | High | Non-IID | High ZKP rigor (~2.5s proof time) |
| `zkp_medium_rigor` | 3 | 5 | ✅ | Medium | Non-IID | Medium ZKP rigor (~1.2s proof time) |
| `zkp_low_rigor` | 3 | 5 | ✅ | Low | Non-IID | Low ZKP rigor (~0.4s proof time) |
| `scaled_up` | 5 | 8 | ✅ | Medium | Non-IID | Larger scale experiment |
| `quick_test` | 2 | 3 | ✅ | Low | IID | Quick test configuration |
| `performance_focused` | 4 | 10 | ❌ | - | Non-IID | Performance-focused without ZKP |

## 📈 Results and Outputs

Benchmark results are saved in the specified output directory:

```
results/
├── multi_dataset_benchmark_results.json  # Raw results data
├── accuracy_comparison.png               # Final accuracy comparison
├── zkp_overhead_analysis.png            # ZKP overhead analysis
├── convergence_analysis.png             # Training convergence plots
└── model_complexity_analysis.png        # Model complexity vs performance
```

## 🎯 Example Outputs

### Comprehensive Analysis Plots

1. **Accuracy Comparison**: Final accuracy across all datasets and configurations
2. **ZKP Overhead Analysis**: 
   - Proof generation times by rigor level (High: ~2.5s, Medium: ~1.2s, Low: ~0.4s)
   - Communication overhead from ZKP integration (~15% increase)
3. **Convergence Analysis**: Training accuracy and loss evolution across rounds
4. **Model Complexity**: Relationship between model architecture, parameters, and performance
5. **Dataset Difficulty Ranking**: Comparative analysis of dataset complexity
6. **Configuration Performance Matrix**: Heatmap of accuracy across datasets and configs

## 🔧 Customization

### Adding New Datasets

To add a new dataset to the benchmark:

1. **Add dataset configuration** to `DatasetManager`:
```python
"new_dataset": {
    "model_class": YourModel,  # Use existing or create new model
    "model_kwargs": {"param1": value1, "param2": value2},
    "input_shape": (channels, height, width),  # or (features,) for tabular
    "num_classes": num_classes,
    "data_loader": self._load_new_dataset_data,
    "description": "Brief description of the dataset",
}
```

2. **Implement data loader method**:
```python
def _load_new_dataset_data(self, num_clients: int, iid: bool):
    # Load your dataset (real or synthetic)
    # Split data across clients (IID or Non-IID)
    # Return list of (train_loader, val_loader) tuples
    return client_data_loaders
```

3. **Choose appropriate model**:
   - `SimpleModel`: Basic fully connected network
   - `MNISTModel`: Optimized for 28x28 grayscale images
   - `CIFAR10Model`: CNN for RGB images
   - `FlexibleMLP`: Highly configurable MLP for any tabular data
   - `ResNetBlock`: For advanced CNN architectures

4. **Update CLI choices**:
```python
choices=["mnist", "cifar10", "your_new_dataset", ...]
```

### Adding New Configurations

Add new benchmark configurations in `get_default_benchmark_configs()`:

```python
{
    "name": "your_config",
    "num_clients": 4,
    "num_rounds": 10,
    "local_epochs": 2,
    "learning_rate": 0.005,
    "enable_zkp": True,
    "proof_rigor": "high",
    "iid": False,
}
```

## 📋 Requirements

The experiments require additional dependencies beyond the base package:

```bash
# Core ML libraries (usually included with secure-fl)
torch>=2.0.0
torchvision>=0.15.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Data handling
pandas>=2.0.0
numpy>=1.24.0
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the correct directory and have secure-fl installed.

2. **Memory Issues**: Reduce batch sizes or use fewer clients for large models:
   ```bash
   python benchmark.py --quick --datasets synthetic
   ```

3. **Slow Execution**: Use quick mode or limit configurations:
   ```bash
   python benchmark.py --quick --configs baseline_iid zkp_medium_rigor
   ```

4. **Missing Data**: MNIST and CIFAR-10 will be downloaded automatically on first run.

### Performance Tips

- Use `--quick` flag for faster iteration during development
- Run single datasets first to test configurations
- Use synthetic data for rapid prototyping
- Consider GPU acceleration for larger models (CIFAR-10)

## 📝 Output Interpretation

### Accuracy Values
- **MNIST**: Typically 85-95% (fully-connected model)
- **Fashion-MNIST**: Typically 80-90% (more challenging than MNIST)
- **CIFAR-10**: Typically 60-80% (basic CNN, limited training)
- **CIFAR-100**: Typically 35-55% (100 classes, more challenging)
- **Synthetic**: 10-80% (depends on complexity and dataset size)
- **Text Classification**: 60-85% (simulated sentiment analysis)
- **Medical**: 70-90% (simulated diagnosis with class imbalance)
- **Financial**: 85-95% (binary fraud detection, imbalanced data)

### ZKP Overhead
- **Proof Time**: Simulated based on complexity (High > Medium > Low)
- **Communication**: ~15% increase for ZKP-enabled configurations
- **Training Time**: Includes simulated proof generation delays

### Convergence Patterns
- **IID**: Usually faster, smoother convergence
- **Non-IID**: May show oscillations, slower convergence
- **ZKP**: Similar convergence with additional overhead

## 🤝 Contributing

To contribute new experiments or improvements:

1. Add your experiment script to this directory
2. Update this README with usage instructions
3. Ensure your experiment follows the existing patterns
4. Test with multiple datasets and configurations
5. Submit a pull request with clear documentation

## 📚 Related Documentation

- [Main README](../README.md) - Project overview and installation
- [INSTALL.md](../INSTALL.md) - Detailed installation guide
- [API Documentation](../docs/) - Technical documentation
- [Examples](../examples/) - Additional usage examples

---

**Note**: This experiments directory is excluded from the package build and has no CLI integration. All experiments must be run as standalone Python scripts for development, research, and evaluation purposes.