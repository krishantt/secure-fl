# SecureFL Docker Compose Example

This directory contains a complete Docker Compose setup for running SecureFL federated learning with zero-knowledge proofs using the pre-built `ghcr.io/krishant/securefl` container image.

## Overview

This setup demonstrates a federated learning scenario with:
- 1 SecureFL server
- 3 SecureFL clients (with different data partitions)
- ZKP (Zero-Knowledge Proof) verification enabled
- MNIST dataset for training
- Optional monitoring service

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB of available RAM
- Internet connection for downloading the container image

### Running the Training

1. **Start the federated learning training:**
   ```bash
   docker-compose up
   ```

2. **Run in detached mode (background):**
   ```bash
   docker-compose up -d
   ```

3. **View logs in real-time:**
   ```bash
   # All services
   docker-compose logs -f

   # Specific service
   docker-compose logs -f securefl-server
   docker-compose logs -f securefl-client-1
   ```

4. **Stop the training:**
   ```bash
   docker-compose down
   ```

### Monitoring

To enable the monitoring service that displays system information:

```bash
docker-compose --profile monitoring up
```

## Configuration

### Server Configuration

The server is configured with:
- **Host**: `0.0.0.0` (accessible from all containers)
- **Port**: `8080` (exposed to host)
- **Training Rounds**: `10`
- **Minimum Clients**: `3`
- **Model**: MNIST neural network
- **ZKP Enabled**: Yes (medium rigor)

### Client Configuration

Each client is configured with:
- **Dataset**: MNIST with different partitions (0, 1, 2)
- **Local Epochs**: `5` per round
- **Batch Size**: `32`
- **Learning Rate**: `0.01`
- **ZKP Enabled**: Yes

### Customization

You can modify the training parameters by editing the `command` sections in `docker-compose.yml`:

#### Server Parameters
```yaml
command: [
  "python", "-m", "secure_fl.cli", "server",
  "--host", "0.0.0.0",
  "--port", "8080",
  "--rounds", "10",              # Number of training rounds
  "--min-clients", "3",          # Minimum clients required
  "--model", "mnist",            # Model type: mnist, cifar10
  "--enable-zkp",                # Enable ZKP verification
  "--proof-rigor", "medium"      # ZKP rigor: low, medium, high
]
```

#### Client Parameters
```yaml
command: [
  "python", "-m", "secure_fl.cli", "client",
  "--server-address", "securefl-server:8080",
  "--client-id", "client-1",
  "--dataset", "mnist",          # Dataset: mnist, cifar10, synthetic
  "--partition", "0",            # Data partition index
  "--enable-zkp",                # Enable ZKP generation
  "--epochs", "5",               # Local training epochs
  "--batch-size", "32",          # Training batch size
  "--learning-rate", "0.01"      # Local learning rate
]
```

## Available Models and Datasets

### Models
- `mnist`: Neural network for MNIST digit classification
- `cifar10`: CNN for CIFAR-10 image classification

### Datasets
- `mnist`: Handwritten digit classification (28x28 grayscale)
- `cifar10`: Natural image classification (32x32 RGB)
- `synthetic`: Synthetically generated data for testing

## Data Persistence

The setup creates local directories that are mounted into the containers:
- `./data/`: Dataset storage and client data partitions
- `./logs/`: Training logs and debug information
- `./results/`: Model checkpoints and training results

These directories will be created automatically when you run the containers.

## Scaling

### Adding More Clients

To add more clients, copy one of the existing client service definitions and modify:

```yaml
securefl-client-4:
  image: ghcr.io/krishant/securefl:latest
  container_name: securefl-client-4
  # ... same configuration as other clients
  command: [
    "python", "-m", "secure_fl.cli", "client",
    "--server-address", "securefl-server:8080",
    "--client-id", "client-4",     # Unique client ID
    "--partition", "3",            # Different partition
    # ... other parameters
  ]
```

Don't forget to update the server's `--min-clients` parameter if needed.

### Different Datasets per Client

You can have clients train on different datasets:

```yaml
# Client 1 with MNIST
command: ["python", "-m", "secure_fl.cli", "client", "--dataset", "mnist", ...]

# Client 2 with CIFAR-10
command: ["python", "-m", "secure_fl.cli", "client", "--dataset", "cifar10", ...]
```

## Troubleshooting

### Common Issues

1. **Server health check fails:**
   ```bash
   # Check if the server is starting properly
   docker-compose logs securefl-server
   ```

2. **Clients can't connect:**
   ```bash
   # Verify network connectivity
   docker-compose exec securefl-client-1 ping securefl-server
   ```

3. **Out of memory:**
   ```bash
   # Reduce batch size or disable ZKP temporarily
   # Edit docker-compose.yml and change --batch-size to 16
   # Or add --disable-zkp flag
   ```

4. **ZKP setup issues:**
   ```bash
   # Run setup command in a container
   docker run --rm ghcr.io/krishant/securefl python -m secure_fl.cli setup --action zkp
   ```

### Viewing Container Status

```bash
# Check all containers
docker-compose ps

# Check resource usage
docker stats $(docker-compose ps -q)

# Access container shell for debugging
docker-compose exec securefl-server bash
```

## Advanced Configuration

### Custom Configuration File

You can use a custom configuration file by mounting it into the containers:

```yaml
volumes:
  - ./config.yaml:/home/app/config.yaml
  - ./data:/home/app/data
  # ...
command: [
  "python", "-m", "secure_fl.cli", "server",
  "--config", "/home/app/config.yaml"
]
```

### Environment Variables

Additional environment variables you can set:

```yaml
environment:
  - SECURE_FL_ENV=production
  - SECURE_FL_LOG_LEVEL=INFO
  - SECURE_FL_ZKP_BACKEND=snark  # or stark
  - SECURE_FL_PROOF_PATH=/home/app/proofs
```

### Resource Limits

Add resource constraints to prevent containers from consuming too much memory:

```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 1G
      cpus: '0.5'
```

## Expected Output

When running successfully, you should see:
1. Server starts and waits for clients
2. Clients connect and register with the server
3. Training rounds begin with ZKP verification
4. Model updates are aggregated after each round
5. Final model performance metrics are displayed

The training typically takes 5-10 minutes depending on your hardware and the number of rounds configured.

## Security Notes

- ZKP verification adds computational overhead but ensures training integrity
- Each client trains on a different partition of the data (federated setting)
- All communications happen within the Docker network (isolated from host network by default)
- Proof artifacts are stored in the shared volumes for verification

## Next Steps

- Experiment with different models and datasets
- Adjust ZKP rigor levels for security/performance trade-offs
- Scale to more clients for larger federated learning scenarios
- Integrate with monitoring tools like Prometheus/Grafana
- Deploy on Kubernetes for production use cases