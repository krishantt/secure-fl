#!/bin/bash

# Docker development environment script for Secure FL
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║            Secure FL Development Environment         ║"
    echo "║         Zero-Knowledge Federated Learning            ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Default values
DEV_MODE="interactive"
CONTAINER_NAME="secure-fl-dev-$(date +%s)"
PORT_MAPPING="8080:8080"
VOLUME_MOUNTS=""
ENVIRONMENT_VARS=""
COMMAND=""
AUTO_REMOVE=true
DETACHED=false
BUILD_FIRST=false
USE_COMPOSE=false

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Docker development environment for Secure FL"
    echo ""
    echo "Modes:"
    echo "  --interactive     Start interactive shell (default)"
    echo "  --jupyter         Start Jupyter Lab server"
    echo "  --server          Start FL server"
    echo "  --client          Start FL client"
    echo "  --demo            Run demo and exit"
    echo "  --test            Run tests and exit"
    echo "  --shell           Start shell in existing container"
    echo ""
    echo "Options:"
    echo "  -n, --name NAME   Container name [default: secure-fl-dev-timestamp]"
    echo "  -p, --port PORT   Port mapping [default: 8080:8080]"
    echo "  -d, --detach      Run in detached mode"
    echo "  --build           Build image before starting"
    echo "  --compose         Use docker-compose instead"
    echo "  --gpu             Enable GPU support (requires nvidia-docker)"
    echo "  --no-remove       Don't auto-remove container on exit"
    echo "  -e VAR=VALUE      Set environment variable"
    echo "  -v HOST:CONTAINER Mount additional volume"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Start interactive development shell"
    echo "  $0 --jupyter -p 8888:8888      # Start Jupyter Lab on port 8888"
    echo "  $0 --server -d                  # Start FL server in background"
    echo "  $0 --demo                       # Run demo and exit"
    echo "  $0 --test                       # Run tests"
    echo "  $0 --shell -n my-container      # Connect to existing container"
    echo "  $0 -e DEBUG=1 --interactive     # Start with debug enabled"
    echo ""
    echo "Development workflow:"
    echo "  1. Start development environment: $0"
    echo "  2. Inside container, use: poe, make, or uv commands"
    echo "  3. Code changes are reflected immediately (volume mounted)"
    echo "  4. Exit container when done (Ctrl+D or 'exit')"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --interactive)
                DEV_MODE="interactive"
                COMMAND="bash"
                shift
                ;;
            --jupyter)
                DEV_MODE="jupyter"
                COMMAND="uv run jupyter lab --ip=0.0.0.0 --port=8080 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
                shift
                ;;
            --server)
                DEV_MODE="server"
                COMMAND="uv run python -m secure_fl.cli server"
                shift
                ;;
            --client)
                DEV_MODE="client"
                COMMAND="uv run python -m secure_fl.cli client"
                shift
                ;;
            --demo)
                DEV_MODE="demo"
                COMMAND="uv run python experiments/demo.py"
                AUTO_REMOVE=true
                shift
                ;;
            --test)
                DEV_MODE="test"
                COMMAND="poe test"
                AUTO_REMOVE=true
                shift
                ;;
            --shell)
                DEV_MODE="shell"
                shift
                ;;
            -n|--name)
                CONTAINER_NAME="$2"
                shift 2
                ;;
            -p|--port)
                PORT_MAPPING="$2"
                shift 2
                ;;
            -d|--detach)
                DETACHED=true
                shift
                ;;
            --build)
                BUILD_FIRST=true
                shift
                ;;
            --compose)
                USE_COMPOSE=true
                shift
                ;;
            --gpu)
                GPU_SUPPORT="--gpus all"
                shift
                ;;
            --no-remove)
                AUTO_REMOVE=false
                shift
                ;;
            -e)
                ENVIRONMENT_VARS="$ENVIRONMENT_VARS -e $2"
                shift 2
                ;;
            -v)
                VOLUME_MOUNTS="$VOLUME_MOUNTS -v $2"
                shift 2
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi

    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running"
        exit 1
    fi

    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" ]] || [[ ! -f "Dockerfile" ]]; then
        print_error "Please run this script from the secure-fl project root directory"
        exit 1
    fi
}

# Build image if requested
build_image() {
    if [[ "$BUILD_FIRST" == "true" ]]; then
        print_status "Building development image..."
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        "$SCRIPT_DIR/build.sh" -t development
    fi
}

# Setup volume mounts
setup_volumes() {
    # Standard mounts
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $(pwd):/home/app/workspace"
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $(pwd)/data:/home/app/data"
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $(pwd)/logs:/home/app/logs"
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $(pwd)/results:/home/app/results"

    # Create directories if they don't exist
    mkdir -p data logs results temp
}

# Setup environment variables
setup_environment() {
    # Standard environment variables
    ENVIRONMENT_VARS="$ENVIRONMENT_VARS -e SECURE_FL_ENV=development"
    ENVIRONMENT_VARS="$ENVIRONMENT_VARS -e PYTHONPATH=/home/app/workspace"

    # Add current user ID for file permissions
    ENVIRONMENT_VARS="$ENVIRONMENT_VARS -e HOST_UID=$(id -u)"
    ENVIRONMENT_VARS="$ENVIRONMENT_VARS -e HOST_GID=$(id -g)"
}

# Start with docker-compose
start_with_compose() {
    print_status "Starting development environment with docker-compose..."

    if [[ "$BUILD_FIRST" == "true" ]]; then
        docker-compose up --build -d secure-fl-dev
    else
        docker-compose up -d secure-fl-dev
    fi

    print_success "Development environment started"
    print_status "Container name: $(docker-compose ps -q secure-fl-dev)"
    print_status "To connect: docker exec -it $(docker-compose ps -q secure-fl-dev) bash"
}

# Connect to existing container
connect_to_existing() {
    print_status "Connecting to existing container: $CONTAINER_NAME"

    # Check if container exists and is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_error "Container '$CONTAINER_NAME' is not running"
        print_status "Available containers:"
        docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
        exit 1
    fi

    docker exec -it "$CONTAINER_NAME" bash
}

# Start new container
start_container() {
    print_status "Starting development container..."

    # Build docker run command
    local docker_cmd="docker run"

    # Add auto-remove flag
    if [[ "$AUTO_REMOVE" == "true" ]]; then
        docker_cmd="$docker_cmd --rm"
    fi

    # Add interactive flags for interactive modes
    if [[ "$DEV_MODE" == "interactive" || "$DEV_MODE" == "shell" ]]; then
        docker_cmd="$docker_cmd -it"
    fi

    # Add detached flag
    if [[ "$DETACHED" == "true" ]]; then
        docker_cmd="$docker_cmd -d"
    fi

    # Add name
    docker_cmd="$docker_cmd --name $CONTAINER_NAME"

    # Add port mapping
    if [[ -n "$PORT_MAPPING" ]]; then
        docker_cmd="$docker_cmd -p $PORT_MAPPING"
    fi

    # Add GPU support
    if [[ -n "$GPU_SUPPORT" ]]; then
        docker_cmd="$docker_cmd $GPU_SUPPORT"
    fi

    # Add volume mounts
    docker_cmd="$docker_cmd $VOLUME_MOUNTS"

    # Add environment variables
    docker_cmd="$docker_cmd $ENVIRONMENT_VARS"

    # Add image
    docker_cmd="$docker_cmd secure-fl:development"

    # Add command
    if [[ -n "$COMMAND" ]]; then
        docker_cmd="$docker_cmd $COMMAND"
    fi

    # Execute the command
    print_status "Executing: $docker_cmd"
    eval "$docker_cmd"

    if [[ "$?" -eq 0 ]]; then
        print_success "Container started successfully"

        if [[ "$DETACHED" == "true" ]]; then
            print_status "Container is running in background"
            print_status "To connect: docker exec -it $CONTAINER_NAME bash"
            print_status "To view logs: docker logs -f $CONTAINER_NAME"
            print_status "To stop: docker stop $CONTAINER_NAME"
        fi
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Show development tips
show_dev_tips() {
    echo ""
    print_status "Development Environment Ready! 🚀"
    echo ""
    echo -e "${CYAN}Available tools inside the container:${NC}"
    echo "  • uv          - Fast Python package manager"
    echo "  • poe         - Modern Python task runner"
    echo "  • make        - Traditional task runner"
    echo "  • circom      - ZK circuit compiler"
    echo "  • snarkjs     - ZK-SNARK JavaScript library"
    echo ""
    echo -e "${CYAN}Quick commands:${NC}"
    echo "  • poe demo    - Run federated learning demo"
    echo "  • poe test    - Run test suite"
    echo "  • poe lint    - Check code quality"
    echo "  • make help   - Show all available commands"
    echo ""
    echo -e "${CYAN}File locations:${NC}"
    echo "  • /home/app/workspace  - Your source code (live mounted)"
    echo "  • /home/app/data       - Datasets and models"
    echo "  • /home/app/logs       - Application logs"
    echo "  • /home/app/results    - Experiment results"
    echo ""
    echo -e "${CYAN}Networking:${NC}"
    echo "  • FL Server will be available at: http://localhost:8080"
    echo "  • Jupyter Lab (if started): http://localhost:8888"
    echo ""
    echo -e "${GREEN}Happy coding! 🎉${NC}"
}

# Show container status
show_status() {
    echo ""
    print_status "Container Status:"
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" --filter "name=secure-fl"
    echo ""

    print_status "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" --filter "name=secure-fl" 2>/dev/null || print_warning "No running containers found"
}

# Main execution function
main() {
    print_banner

    # Parse arguments
    parse_args "$@"

    # Check prerequisites
    check_prerequisites

    # Handle different modes
    case "$DEV_MODE" in
        shell)
            connect_to_existing
            ;;
        *)
            # Build if requested
            build_image

            # Setup volumes and environment
            setup_volumes
            setup_environment

            if [[ "$USE_COMPOSE" == "true" ]]; then
                start_with_compose
            else
                start_container
            fi

            # Show tips for interactive modes
            if [[ "$DEV_MODE" == "interactive" && "$DETACHED" == "false" ]]; then
                show_dev_tips
            fi
            ;;
    esac

    # Show status for detached containers
    if [[ "$DETACHED" == "true" ]]; then
        show_status
    fi
}

# Trap to cleanup on script exit
cleanup() {
    if [[ "$AUTO_REMOVE" == "false" && -n "$CONTAINER_NAME" ]]; then
        print_status "Container '$CONTAINER_NAME' is still running"
        print_status "To stop: docker stop $CONTAINER_NAME"
        print_status "To remove: docker rm $CONTAINER_NAME"
    fi
}

trap cleanup EXIT

# Run main function with all arguments
main "$@"
