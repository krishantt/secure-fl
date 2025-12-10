# Secure FL Scripts Directory

This directory contains various utility scripts for the Secure FL project. These scripts help with development, deployment, testing, and maintenance tasks.

## Quick Start

### Master Docker Script
The main entry point for all Docker operations:

```bash
# Interactive menu
./scripts/docker.sh

# Quick commands
./scripts/docker.sh demo        # Run demo
./scripts/docker.sh dev         # Development environment
./scripts/docker.sh build       # Build images
./scripts/docker.sh up          # Start FL environment
./scripts/docker.sh clean       # Cleanup resources
```

## Directory Structure

```
scripts/
├── README.md              # This file
├── docker.sh              # Master Docker interface
├── docker/                # Docker-specific scripts
│   ├── build.sh          # Build Docker images
│   ├── clean.sh          # Cleanup Docker resources
│   ├── compose.sh        # Docker Compose management
│   ├── dev.sh            # Development environment
│   └── quickstart.sh     # Quick demo and setup
├── version.py            # Version management
└── cleanup_redundancy.py # Code cleanup utilities
```

## Script Categories

### 🐳 Docker Scripts (`docker/`)

#### `docker.sh` - Master Docker Interface
Unified command-line interface for all Docker operations.

**Usage:**
```bash
./scripts/docker.sh [COMMAND] [OPTIONS]
```

**Commands:**
- `demo` - Run federated learning demo
- `dev` - Start development environment
- `build` - Build Docker images
- `up/down` - Start/stop services
- `clean` - Cleanup resources
- `info` - System information

#### `docker/build.sh` - Image Builder
Builds Docker images with various targets and options.

**Usage:**
```bash
./scripts/docker/build.sh [OPTIONS]
```

**Options:**
- `-t TARGET` - Build specific target (base, development, server, client)
- `--all` - Build all targets
- `-c, --clean` - Clean build (no cache)
- `--test` - Run tests after building

#### `docker/dev.sh` - Development Environment
Starts and manages development containers.

**Usage:**
```bash
./scripts/docker/dev.sh [OPTIONS]
```

**Modes:**
- `--interactive` - Interactive shell (default)
- `--server` - FL server
- `--client` - FL client
- `--demo` - Run demo and exit
- `--test` - Run tests and exit

#### `docker/compose.sh` - Compose Management
Advanced Docker Compose operations for multi-service deployments.

**Usage:**
```bash
./scripts/docker/compose.sh [COMMAND] [OPTIONS]
```

**Commands:**
- `up/down` - Start/stop services
- `logs` - View logs (`--follow` for live)
- `scale` - Scale services
- `health` - Check service health
- `shell` - Open shell in service

#### `docker/clean.sh` - Resource Cleanup
Comprehensive Docker cleanup with safety options.

**Usage:**
```bash
./scripts/docker/clean.sh [OPTIONS]
```

**Options:**
- `--all` - Clean everything
- `--containers` - Clean containers only
- `--images` - Clean images only
- `--volumes` - Clean volumes only
- `--dry-run` - Preview changes
- `--force` - Skip confirmation

#### `docker/quickstart.sh` - Quick Setup
One-command setup for common scenarios.

**Usage:**
```bash
./scripts/docker/quickstart.sh [MODE]
```

**Modes:**
- `demo` - Quick demo
- `dev` - Development setup
- `full` - Full FL environment

### 🔧 Utility Scripts

#### `version.py` - Version Management
Handles version numbering, incrementing, and build information.

**Usage:**
```bash
python scripts/version.py [COMMAND]
```

**Commands:**
- `show` - Display current version
- `increment` - Increment build number
- `build-info` - Show build information
- `publish` - Prepare for publishing

#### `cleanup_redundancy.py` - Code Cleanup
Removes redundant code, unused imports, and optimizes the codebase.

## Integration with Build Tools

### Make Integration
All Docker scripts are integrated with the Makefile:

```bash
make docker-demo          # Run demo
make docker-dev           # Development environment
make docker-build         # Build images
make up                   # Start services
make down                 # Stop services
make clean-docker         # Cleanup Docker resources
```

### Poethepoet (poe) Integration
Available as poe tasks:

```bash
poe docker-demo           # Run demo
poe docker-dev            # Development environment
poe docker-build          # Build images
poe docker-clean          # Cleanup resources
```

### UV Integration
Direct usage with uv:

```bash
uv run python scripts/version.py show
```

## Environment Setup

### Prerequisites
- Docker and Docker Compose
- UV package manager (for Python tasks)
- Bash shell (for script execution)

### Directory Structure
Scripts expect to be run from the project root and will create necessary directories:

```
secure-fl/
├── data/                 # Datasets and models
├── logs/                 # Application logs  
├── results/              # Experiment results
└── temp/                 # Temporary files
```

## Common Workflows

### Development Workflow
```bash
# 1. Start development environment
./scripts/docker.sh dev

# 2. Inside container, run tasks
poe test                  # Run tests
poe lint                  # Check code quality
poe demo                  # Run demo
make help                 # Show all commands

# 3. Exit when done
exit
```

### Testing Workflow
```bash
# Build and test
./scripts/docker.sh build --test

# Run specific tests
./scripts/docker.sh dev --test

# Clean testing environment
./scripts/docker.sh clean --containers
```

### Production Deployment
```bash
# Build production images
./scripts/docker.sh build --all

# Start production environment
./scripts/docker.sh up

# Monitor services
./scripts/docker.sh status
./scripts/docker.sh health

# View logs
./scripts/docker.sh logs --follow
```

### Maintenance
```bash
# Check system status
./scripts/docker.sh info

# Cleanup old resources
./scripts/docker.sh clean --all

# Update dependencies
uv lock --upgrade
./scripts/docker.sh build --clean
```

## Troubleshooting

### Common Issues

**Script not executable:**
```bash
chmod +x scripts/docker/*.sh scripts/docker.sh
```

**Docker not running:**
```bash
# Start Docker service
sudo systemctl start docker  # Linux
# or start Docker Desktop     # macOS/Windows
```

**Permission issues:**
```bash
# Fix directory permissions
sudo chown -R $(id -u):$(id -g) data logs results
```

**Build failures:**
```bash
# Clean build with latest dependencies
uv lock --upgrade
./scripts/docker.sh build --clean --all
```

### Getting Help

1. **Script help:** Most scripts have `--help` option
2. **Interactive menu:** Run `./scripts/docker.sh` without arguments
3. **Make help:** Run `make help` for available commands
4. **Documentation:** Check `DOCKER.md` for detailed Docker information

## Contributing

When adding new scripts:

1. **Location:** Place in appropriate subdirectory
2. **Executable:** Make scripts executable (`chmod +x`)
3. **Documentation:** Update this README
4. **Integration:** Add to Makefile and pyproject.toml if needed
5. **Error handling:** Use proper error handling and logging
6. **Help:** Include `--help` option for complex scripts

### Script Template
```bash
#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m' 
BLUE='\033[0;34m'
NC='\033[0m'

print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }

# Your script logic here
```

For more detailed Docker information, see [DOCKER.md](../DOCKER.md).