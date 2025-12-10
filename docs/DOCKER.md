# Docker Setup for Secure FL

This document provides comprehensive instructions for running Secure FL using Docker with the modern `uv` package manager and task runners.

## Quick Start

### Option 1: Interactive Setup
```bash
./docker-quickstart.sh
```

### Option 2: Command Line
```bash
# Run demo
./docker-quickstart.sh demo

# Setup development environment
./docker-quickstart.sh dev

# Setup full federated learning environment
./docker-quickstart.sh full
```

### Option 3: Make Commands
```bash
# Show all available commands
make help

# Run demo
make docker-demo

# Start development environment
make docker-dev

# Start full FL environment with docker-compose
make up
```

## Available Docker Images

The Docker setup creates multiple specialized images:

- `secure-fl:base` - Base image with core dependencies
- `secure-fl:production` - Production-ready image
- `secure-fl:server` - FL server image
- `secure-fl:client` - FL client image
- `secure-fl:development` - Development environment

## Manual Docker Commands

### Build Images
```bash
# Build base image
./docker-build.sh -t base

# Build all images
./docker-build.sh --all

# Clean build (no cache)
./docker-build.sh -c -t development
```

### Run Demo
```bash
docker run --rm \
  -v "$(pwd)/data:/home/app/data" \
  -v "$(pwd)/logs:/home/app/logs" \
  -v "$(pwd)/results:/home/app/results" \
  secure-fl:base \
  uv run python experiments/demo.py
```

### Development Environment
```bash
docker run --rm -it \
  -p 8080:8080 \
  -v "$(pwd):/home/app/workspace" \
  -v "$(pwd)/data:/home/app/data" \
  -v "$(pwd)/logs:/home/app/logs" \
  -v "$(pwd)/results:/home/app/results" \
  secure-fl:development \
  bash
```

## Docker Compose

### Start Full Environment
```bash
# Start server and clients
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Available Services

1. **secure-fl-server** - FL aggregation server (port 8080)
2. **secure-fl-client-1** - First FL client
3. **secure-fl-client-2** - Second FL client
4. **secure-fl-dev** - Development environment (port 8081)
5. **secure-fl-demo** - One-time demo execution

## Task Runners in Docker

### Using Make (Traditional)
```bash
# Inside container
make test
make lint
make format
make demo
```

### Using Poe (Modern Python Task Runner)
```bash
# Inside container
poe test
poe lint
poe format
poe demo
poe precommit
```

### Using UV directly
```bash
# Inside container
uv run python experiments/demo.py
uv run pytest tests/
uv run ruff check .
```

## Development Workflow

### 1. Start Development Container
```bash
make docker-dev
# or
./docker-quickstart.sh dev
```

### 2. Inside the Container
```bash
# Install development dependencies
uv sync --all-extras --dev

# Run tests
poe test

# Format and lint code
poe precommit

# Run experiments
poe demo
poe benchmark
```

### 3. Code Changes
Your local code is mounted at `/home/app/workspace`, so changes are reflected immediately.

## Environment Variables

Configure the application using environment variables:

```bash
# Server configuration
SECURE_FL_SERVER_HOST=0.0.0.0
SECURE_FL_SERVER_PORT=8080

# Client configuration
SECURE_FL_CLIENT_ID=client-1
SECURE_FL_SERVER_URL=http://secure-fl-server:8080

# Environment mode
SECURE_FL_ENV=development  # or production, demo
```

## Volume Mounts

The Docker setup uses the following volume mounts:

- `./data:/home/app/data` - Dataset and model storage
- `./logs:/home/app/logs` - Application logs
- `./results:/home/app/results` - Experiment results
- `./:/home/app/workspace` - Source code (development only)

## ZKP Tools

All containers include Zero-Knowledge Proof tools:

- **Circom** - Circuit compiler for zk-SNARKs
- **SnarkJS** - JavaScript library for zk-SNARKs
- **Node.js** - Required for ZKP tools

Test ZKP availability:
```bash
docker run --rm secure-fl:base circom --version
docker run --rm secure-fl:base snarkjs --version
```

## Troubleshooting

### Build Issues

1. **Lock file out of date:**
   ```bash
   uv lock --upgrade
   ./docker-build.sh -c -t base
   ```

2. **Permission issues:**
   ```bash
   sudo chown -R $(id -u):$(id -g) data logs results
   ```

3. **Cache issues:**
   ```bash
   docker system prune -f
   make clean-docker
   ```

### Runtime Issues

1. **Module not found:**
   ```bash
   # Ensure you're using the correct Python path
   docker run --rm secure-fl:base uv run python -c "import secure_fl; print('OK')"
   ```

2. **ZKP tools not working:**
   ```bash
   # Check if ZKP tools are installed
   docker run --rm secure-fl:base which circom
   docker run --rm secure-fl:base which snarkjs
   ```

## Performance Optimization

### Multi-stage Builds
The Dockerfile uses multi-stage builds to minimize image size:
- Node.js stage for ZKP tools
- Base stage for core application
- Specialized stages for different use cases

### Caching
- Dependencies are cached using Docker layer caching
- UV cache is used for faster Python package installation
- Built images are reused when possible

### Resource Usage
```bash
# Monitor resource usage
docker stats

# Limit resources
docker run --memory=2g --cpus=1 secure-fl:base
```

## Security

### User Management
- Containers run as non-root user `app`
- Home directory: `/home/app`
- UID/GID: Created dynamically

### Network Security
- Internal Docker network for service communication
- Only necessary ports are exposed
- No hardcoded secrets in images

## Production Deployment

### Environment Configuration
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  secure-fl-server:
    image: secure-fl:production
    environment:
      - SECURE_FL_ENV=production
      - SECURE_FL_LOG_LEVEL=INFO
    ports:
      - "8080:8080"
    restart: unless-stopped
```

### Scaling
```bash
# Scale clients
docker-compose up -d --scale secure-fl-client=5
```

### Monitoring
```bash
# Health checks
docker-compose ps
docker inspect --format='{{.State.Health.Status}}' container_name
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Build and test Docker image
  run: |
    ./docker-build.sh -t base
    make docker-demo
```

### Testing in Docker
```bash
# Run tests in container
docker run --rm secure-fl:base poe test

# Run with coverage
docker run --rm secure-fl:base poe test --cov-report=xml
```

## Advanced Usage

### Custom Networks
```bash
# Create custom network
docker network create secure-fl-net

# Run with custom network
docker run --rm --network=secure-fl-net secure-fl:base
```

### GPU Support
```bash
# For CUDA support (requires nvidia-docker)
docker run --rm --gpus all secure-fl:base
```

### Debugging
```bash
# Debug container
docker run --rm -it --entrypoint=/bin/bash secure-fl:development

# Inspect layers
docker history secure-fl:base
```

## Best Practices

1. **Use specific tags** - Don't use `latest` in production
2. **Multi-stage builds** - Keep images small
3. **Non-root user** - Security best practice
4. **Health checks** - Monitor container health
5. **Proper logging** - Use structured logs
6. **Resource limits** - Prevent resource exhaustion
7. **Regular updates** - Keep base images updated

## Migration from PDM

If you're migrating from the old PDM-based setup:

1. **Update dependencies:**
   ```bash
   # Remove old PDM files
   rm pdm.lock .pdm-python
   
   # Use new uv setup
   ./docker-quickstart.sh
   ```

2. **Update scripts:**
   - Replace `pdm run` with `uv run`
   - Replace `pdm install` with `uv sync`
   - Use `poe` or `make` for task running

3. **Environment setup:**
   - No more PDM virtual environment management
   - UV handles everything automatically
   - Faster dependency resolution and installation

For more information, see the main [README.md](README.md) or visit the [project repository](https://github.com/krishantt/secure-fl).