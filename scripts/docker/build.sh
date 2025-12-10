#!/bin/bash

# Docker build script for Secure FL (using uv)
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
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

# Default values
BUILD_TARGET="base"
PUSH_IMAGES=false
CLEAN_BUILD=false
RUN_TESTS=false

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --target TARGET     Build target (base, production, server, client, development) [default: base]"
    echo "  -p, --push             Push images to registry after building"
    echo "  -c, --clean            Clean build (no cache)"
    echo "  --test                 Run tests after building"
    echo "  --all                  Build all targets"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Build base image"
    echo "  $0 -t server           # Build server image"
    echo "  $0 --all               # Build all images"
    echo "  $0 -c -t development   # Clean build of development image"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -p|--push)
            PUSH_IMAGES=true
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --all)
            BUILD_TARGET="all"
            shift
            ;;
        -h|--help)
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

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -f "Dockerfile" ]] || [[ ! -f "uv.lock" ]]; then
    print_error "Please run this script from the secure-fl project root directory."
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data logs results temp
mkdir -p data/{datasets,models,experiments}
mkdir -p logs/{server,client,experiments}
mkdir -p results/{benchmarks,training,evaluation}

# Ensure uv.lock is present and up to date
# Ensure uv.lock is up to date
if [[ ! -f "uv.lock" ]]; then
    print_warning "uv.lock not found, generating lock file..."
    uv lock
fi

# Ensure we can find other docker scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build function
build_image() {
    local target=$1
    local tag="secure-fl:$target"

    print_status "Building $tag..."

    local build_args=""
    if [[ "$CLEAN_BUILD" == "true" ]]; then
        build_args="$build_args --no-cache"
    fi

    if [[ "$target" == "base" ]]; then
        docker build $build_args --target base -t "$tag" .
    else
        docker build $build_args --target "$target" -t "$tag" .
    fi

    if [[ $? -eq 0 ]]; then
        print_success "Successfully built $tag"
    else
        print_error "Failed to build $tag"
        exit 1
    fi
}

# Main build logic
if [[ "$BUILD_TARGET" == "all" ]]; then
    print_status "Building all images..."
    targets=("base" "production" "server" "client" "development")
    for target in "${targets[@]}"; do
        build_image "$target"
    done
else
    build_image "$BUILD_TARGET"
fi

# Run tests if requested
if [[ "$RUN_TESTS" == "true" ]]; then
    print_status "Running tests..."
    docker run --rm secure-fl:$BUILD_TARGET uv run pytest tests/ -v
fi

# Push images if requested
if [[ "$PUSH_IMAGES" == "true" ]]; then
    print_warning "Image pushing not implemented yet. Set up your registry first."
fi

print_success "Docker build completed successfully!"

# Show available images
print_status "Available secure-fl images:"
docker images secure-fl --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

echo ""
print_status "Usage examples:"
echo "  # Run demo:"
echo "  docker run --rm -v \$(pwd)/data:/home/app/data secure-fl:base uv run python experiments/demo.py"
echo ""
echo "  # Start development environment:"
echo "  docker run --rm -it -p 8080:8080 -v \$(pwd):/home/app/workspace secure-fl:development"
echo ""
echo "  # Use script commands:"
echo "  ./scripts/docker/quickstart.sh demo"
echo "  ./scripts/docker/dev.sh"
echo ""
echo "  # Use make commands:"
echo "  make docker-demo"
echo "  make docker-dev"
echo ""
echo "  # Use docker-compose for full setup:"
echo "  docker-compose up -d"
