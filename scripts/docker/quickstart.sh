#!/bin/bash

# Secure FL Docker Quickstart Script (using uv)
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║              Secure FL Docker Quickstart            ║"
    echo "║         Zero-Knowledge Federated Learning            ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi

    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi

    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" ]] || [[ ! -f "Dockerfile" ]] || [[ ! -f "uv.lock" ]]; then
        print_error "Please run this script from the secure-fl project root directory."
        print_error "Make sure pyproject.toml, Dockerfile, and uv.lock files exist."
        exit 1
    fi

    # Check if uv is available (optional, for local development)
    if command -v uv &> /dev/null; then
        print_success "uv package manager detected"
    else
        print_warning "uv not found locally (will use Docker's uv)"
    fi

    print_success "Prerequisites check passed!"
}

# Setup directories
setup_directories() {
    print_status "Setting up directories..."

    # Create required directories
    mkdir -p data logs results temp k8s infra
    mkdir -p data/{datasets,models,experiments}
    mkdir -p logs/{server,client,experiments}
    mkdir -p results/{benchmarks,training,evaluation}

    # Create gitkeep files to preserve directory structure
    find data logs results -type d -empty -exec touch {}/.gitkeep \;

    print_success "Directories created successfully!"
}

# Build Docker image
build_image() {
    print_status "Building Secure FL Docker image..."

    # Ensure uv.lock is up to date
    if command -v uv &> /dev/null; then
        print_status "Updating uv.lock..."
        uv lock
    fi

    # Build the base image first
    docker build --target base -t secure-fl:base . || {
        print_error "Failed to build Docker image"
        exit 1
    }

    print_success "Docker image built successfully!"
}

# Run demo
run_demo() {
    print_status "Running Secure FL demo..."

    docker run --rm \
        -v "$(pwd)/data:/home/app/data" \
        -v "$(pwd)/logs:/home/app/logs" \
        -v "$(pwd)/results:/home/app/results" \
        secure-fl:base \
        uv run python experiments/demo.py

    print_success "Demo completed successfully!"
}

# Setup development environment
setup_development() {
    print_status "Setting up development environment..."

    # Build development image
    docker build --target development -t secure-fl:dev . || {
        print_error "Failed to build development image"
        exit 1
    }

    print_success "Development environment ready!"
    echo ""
    echo "  # Start development environment:"
    echo "  docker run --rm -it -p 8080:8080 -v \$(pwd):/home/app/workspace secure-fl:development"
    echo ""
    echo "Available tools in development container:"
    echo "  - uv (package manager)"
    echo "  - poe (task runner)"
    echo "  - make (traditional task runner)"
    echo "  - All ZKP tools (circom, snarkjs)"
}

# Setup full federated learning environment
setup_full_fl() {
    print_status "Setting up full federated learning environment..."

    # Build all required images
    docker build --target server -t secure-fl:server . || {
        print_error "Failed to build server image"
        exit 1
    }

    docker build --target client -t secure-fl:client . || {
        print_error "Failed to build client image"
        exit 1
    }

    print_success "Full FL environment ready!"
    echo ""
    echo "To start the full environment:"
    echo "  docker-compose up -d"
    echo ""
    echo "To view logs:"
    echo "  docker-compose logs -f"
    echo ""
    echo "To stop the environment:"
    echo "  docker-compose down"
}

# Show usage menu
show_menu() {
    echo ""
    echo "What would you like to do?"
    echo ""
    echo "1) Quick Demo - Build and run a simple demo"
    echo "2) Development Setup - Setup development environment with Jupyter"
    echo "3) Full FL Setup - Setup complete federated learning environment"
    echo "4) Build Only - Just build the Docker images"
    echo "5) Exit"
    echo ""
    read -p "Enter your choice (1-5): " choice
}

# Main execution
main() {
    print_banner

    check_prerequisites
    setup_directories

    if [[ $# -eq 0 ]]; then
        # Interactive mode
        while true; do
            show_menu
            case $choice in
                1)
                    build_image
                    run_demo
                    break
                    ;;
                2)
                    setup_development
                    break
                    ;;
                3)
                    setup_full_fl
                    break
                    ;;
                4)
                    build_image
                    break
                    ;;
                5)
                    print_status "Exiting..."
                    exit 0
                    ;;
                *)
                    print_error "Invalid option. Please choose 1-5."
                    ;;
            esac
        done
    else
        # Command line mode
        case $1 in
            demo)
                build_image
                run_demo
                ;;
            dev|development)
                setup_development
                ;;
            full|fl)
                setup_full_fl
                ;;
            build)
                build_image
                ;;
            *)
                echo "Usage: $0 [demo|dev|full|build]"
                echo ""
                echo "Commands:"
                echo "  demo  - Build and run demo"
                echo "  dev   - Setup development environment"
                echo "  full  - Setup full FL environment"
                echo "  build - Build Docker images only"
                echo ""
                echo "Run without arguments for interactive mode."
                exit 1
                ;;
        esac
    fi

    echo ""
    print_success "Secure FL Docker setup completed!"
    echo ""
    echo "Next steps:"
    echo "• Check the logs/ directory for application logs"
    echo "• Check the results/ directory for training results"
    echo "• Use 'make help' to see available commands"
    echo "• Use 'poe' for task running inside containers"
    echo "• Read the documentation: README.md"
    echo "• Visit the project repository: https://github.com/krishantt/secure-fl"
    echo ""
    echo "Quick commands:"
    echo "• make docker-demo  # Run demo in Docker"
    echo "• make docker-dev   # Start development environment"
    echo "• make up          # Start full FL environment"
    echo "• make help        # Show all available commands"
}

# Run main function with all arguments
main "$@"
