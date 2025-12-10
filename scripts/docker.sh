#!/bin/bash

# Master Docker script for Secure FL
# Provides unified access to all Docker functionality
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
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║                Secure FL Docker CLI                  ║"
    echo "║         Zero-Knowledge Federated Learning            ║"
    echo "║                                                      ║"
    echo "║  Unified interface for all Docker operations         ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR/docker"

# Usage function
usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Unified Docker interface for Secure FL development and deployment"
    echo ""
    echo -e "${CYAN}Quick Start Commands:${NC}"
    echo "  demo                    Run a quick demo"
    echo "  dev                     Start development environment"
    echo "  build                   Build all Docker images"
    echo "  up                      Start full FL environment"
    echo "  down                    Stop all services"
    echo "  clean                   Clean up Docker resources"
    echo ""
    echo -e "${CYAN}Build Commands:${NC}"
    echo "  build [TARGET]          Build specific image target"
    echo "    --all                 Build all images"
    echo "    --clean               Clean build (no cache)"
    echo "    --test                Run tests after building"
    echo ""
    echo -e "${CYAN}Development Commands:${NC}"
    echo "  dev                     Start interactive development shell"
    echo "    --jupyter             Start Jupyter Lab"
    echo "    --server              Start FL server"
    echo "    --client              Start FL client"
    echo "    --demo                Run demo and exit"
    echo "    --test                Run tests and exit"
    echo "    --build               Build before starting"
    echo ""
    echo -e "${CYAN}Compose Commands:${NC}"
    echo "  up                      Start all services"
    echo "  down                    Stop all services"
    echo "  restart                 Restart services"
    echo "  logs                    Show logs"
    echo "  status                  Show service status"
    echo "  scale SERVICE N         Scale service to N instances"
    echo ""
    echo -e "${CYAN}Maintenance Commands:${NC}"
    echo "  clean                   Interactive cleanup"
    echo "    --all                 Clean everything"
    echo "    --containers          Clean containers only"
    echo "    --images              Clean images only"
    echo "    --volumes             Clean volumes only"
    echo "    --force               Force cleanup without confirmation"
    echo ""
    echo -e "${CYAN}Testing Commands:${NC}"
    echo "  test                    Test Docker images"
    echo "    --image IMAGE         Test specific image"
    echo "    --all                 Test all images"
    echo "    --quick               Quick tests only"
    echo "    --full                Full test suite"
    echo ""
    echo -e "${CYAN}Utility Commands:${NC}"
    echo "  shell [CONTAINER]       Connect to running container"
    echo "  exec CONTAINER CMD      Execute command in container"
    echo "  health                  Check service health"
    echo "  info                    Show Docker system information"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo "  $0 demo                 # Quick demo"
    echo "  $0 dev                  # Development environment"
    echo "  $0 dev --jupyter        # Jupyter Lab environment"
    echo "  $0 build --all          # Build all images"
    echo "  $0 up                   # Start FL environment"
    echo "  $0 logs --follow        # Follow all logs"
    echo "  $0 scale client 5       # Scale to 5 clients"
    echo "  $0 test --image test    # Test specific image"
    echo "  $0 clean --containers   # Clean containers only"
    echo ""
    echo "For detailed help on specific commands, use:"
    echo "  $0 COMMAND --help"
}

# Check if Docker scripts exist
check_docker_scripts() {
    if [[ ! -d "$DOCKER_DIR" ]]; then
        print_error "Docker scripts directory not found: $DOCKER_DIR"
        exit 1
    fi

    local required_scripts=("build.sh" "quickstart.sh" "dev.sh" "compose.sh" "clean.sh" "test.sh")
    for script in "${required_scripts[@]}"; do
        if [[ ! -f "$DOCKER_DIR/$script" ]]; then
            print_error "Required Docker script not found: $DOCKER_DIR/$script"
            exit 1
        fi
        if [[ ! -x "$DOCKER_DIR/$script" ]]; then
            print_warning "Making $script executable..."
            chmod +x "$DOCKER_DIR/$script"
        fi
    done
}

# Show Docker system information
show_info() {
    echo -e "${CYAN}Docker System Information:${NC}"
    echo ""

    # Docker version
    echo -e "${BLUE}Docker Version:${NC}"
    docker --version
    echo ""

    # Docker system info
    echo -e "${BLUE}System Status:${NC}"
    if docker info >/dev/null 2>&1; then
        echo "✓ Docker is running"
    else
        echo "✗ Docker is not running"
        return 1
    fi

    # Check docker-compose
    if command -v docker-compose &> /dev/null; then
        echo "✓ Docker Compose available: $(docker-compose --version)"
    else
        echo "✗ Docker Compose not available"
    fi
    echo ""

    # Current project images
    echo -e "${BLUE}Secure FL Images:${NC}"
    docker images secure-fl --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" 2>/dev/null || echo "No Secure FL images found"
    echo ""

    # Running containers
    echo -e "${BLUE}Running Containers:${NC}"
    docker ps --filter "name=secure-fl" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No Secure FL containers running"
    echo ""

    # System resources
    echo -e "${BLUE}System Resources:${NC}"
    docker system df 2>/dev/null || echo "Unable to get system resource info"
}

# Route to appropriate script
route_command() {
    local command="$1"
    shift

    case "$command" in
        # Quick start commands
        demo)
            "$DOCKER_DIR/quickstart.sh" demo "$@"
            ;;

        # Build commands
        build)
            "$DOCKER_DIR/build.sh" "$@"
            ;;

        # Development commands
        dev|development)
            "$DOCKER_DIR/dev.sh" "$@"
            ;;

        # Compose commands
        up|start)
            "$DOCKER_DIR/compose.sh" up "$@"
            ;;
        down|stop)
            "$DOCKER_DIR/compose.sh" down "$@"
            ;;
        restart)
            "$DOCKER_DIR/compose.sh" restart "$@"
            ;;
        logs)
            "$DOCKER_DIR/compose.sh" logs "$@"
            ;;
        status|ps)
            "$DOCKER_DIR/compose.sh" status "$@"
            ;;
        scale)
            "$DOCKER_DIR/compose.sh" scale "$@"
            ;;
        health)
            "$DOCKER_DIR/compose.sh" health "$@"
            ;;

        # Maintenance commands
        clean|cleanup)
            "$DOCKER_DIR/clean.sh" "$@"
            ;;

        # Testing commands
        test)
            "$DOCKER_DIR/test.sh" "$@"
            ;;

        # Utility commands
        shell)
            if [[ -n "$1" ]]; then
                docker exec -it "$1" bash
            else
                "$DOCKER_DIR/dev.sh" --shell "$@"
            fi
            ;;
        exec)
            local container="$1"
            shift
            if [[ -z "$container" ]]; then
                print_error "Container name required for exec command"
                exit 1
            fi
            docker exec -it "$container" "$@"
            ;;
        info)
            show_info
            ;;

        # Help and unknown commands
        help|--help|-h)
            usage
            ;;
        "")
            print_warning "No command specified"
            usage
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            echo "Available commands: demo, dev, build, up, down, logs, test, clean, shell, info"
            echo "Use '$0 --help' for full command list"
            exit 1
            ;;
    esac
}

# Show quick menu for interactive use
show_quick_menu() {
    echo ""
    echo -e "${CYAN}What would you like to do?${NC}"
    echo ""
    echo "1) Demo - Run a quick federated learning demo"
    echo "2) Develop - Start development environment"
    echo "3) Build - Build Docker images"
    echo "4) Test - Test Docker images"
    echo "5) Deploy - Start full FL environment"
    echo "6) Status - Check running services"
    echo "7) Clean - Cleanup Docker resources"
    echo "8) Help - Show detailed help"
    echo "9) Quit"
    echo ""
    read -p "Enter your choice (1-9): " choice

    case $choice in
        1) route_command "demo" ;;
        2) route_command "dev" ;;
        3) route_command "build" ;;
        4) route_command "test" ;;
        5) route_command "up" ;;
        6) route_command "status" ;;
        7) route_command "clean" ;;
        8) usage ;;
        9) exit 0 ;;
        *)
            print_error "Invalid choice"
            show_quick_menu
            ;;
    esac
}

# Main execution
main() {
    # Show banner
    print_banner

    # Check prerequisites
    check_docker_scripts

    # Handle arguments
    if [[ $# -eq 0 ]]; then
        # No arguments - show interactive menu
        show_quick_menu
    else
        # Route to appropriate command
        route_command "$@"
    fi
}

# Cleanup function
cleanup() {
    # Any cleanup needed on script exit
    :
}

# Set trap for cleanup
trap cleanup EXIT

# Check if script is being run from the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -f "Dockerfile" ]]; then
    print_error "This script must be run from the secure-fl project root directory"
    print_error "Current directory: $(pwd)"
    exit 1
fi

# Run main function with all arguments
main "$@"
