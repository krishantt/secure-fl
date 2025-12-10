#!/bin/bash

# Docker Compose management script for Secure FL
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
    echo "║            Secure FL Docker Compose Manager         ║"
    echo "║         Zero-Knowledge Federated Learning            ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Default values
COMPOSE_FILE="docker-compose.yml"
SERVICE=""
FOLLOW_LOGS=false
BUILD_IMAGES=false
SCALE_COUNT=""

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] COMMAND [SERVICE]"
    echo ""
    echo "Docker Compose management for Secure FL"
    echo ""
    echo "Commands:"
    echo "  up          Start all services"
    echo "  down        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show service status"
    echo "  logs        Show logs"
    echo "  shell       Open shell in service"
    echo "  exec        Execute command in service"
    echo "  build       Build all images"
    echo "  pull        Pull latest images"
    echo "  scale       Scale services"
    echo "  clean       Clean up containers and volumes"
    echo "  health      Check service health"
    echo ""
    echo "Options:"
    echo "  -f FILE     Use specific compose file [default: docker-compose.yml]"
    echo "  -s SERVICE  Target specific service"
    echo "  --follow    Follow logs output"
    echo "  --build     Build images before starting"
    echo "  --scale N   Scale to N instances"
    echo "  -h, --help  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 up                          # Start all services"
    echo "  $0 up --build                  # Build and start services"
    echo "  $0 logs -s secure-fl-server    # Show server logs"
    echo "  $0 logs --follow               # Follow all logs"
    echo "  $0 shell -s secure-fl-dev      # Open shell in dev container"
    echo "  $0 scale -s secure-fl-client --scale 5  # Scale clients to 5"
    echo "  $0 exec -s secure-fl-server \"poe demo\"  # Run demo on server"
}

# Check prerequisites
check_prerequisites() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi

    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running"
        exit 1
    fi

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        print_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--file)
                COMPOSE_FILE="$2"
                shift 2
                ;;
            -s|--service)
                SERVICE="$2"
                shift 2
                ;;
            --follow)
                FOLLOW_LOGS=true
                shift
                ;;
            --build)
                BUILD_IMAGES=true
                shift
                ;;
            --scale)
                SCALE_COUNT="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                if [[ -z "$COMMAND" ]]; then
                    COMMAND="$1"
                else
                    ARGS="$ARGS $1"
                fi
                shift
                ;;
        esac
    done
}

# Docker Compose wrapper
compose_cmd() {
    local cmd="$1"
    shift
    docker-compose -f "$COMPOSE_FILE" "$cmd" "$@"
}

# Start services
cmd_up() {
    print_status "Starting Secure FL services..."

    local up_args=""
    if [[ "$BUILD_IMAGES" == "true" ]]; then
        up_args="$up_args --build"
    fi

    if [[ -n "$SERVICE" ]]; then
        compose_cmd up -d $up_args "$SERVICE"
        print_success "Service $SERVICE started"
    else
        compose_cmd up -d $up_args
        print_success "All services started"
    fi

    # Show status after starting
    cmd_status
}

# Stop services
cmd_down() {
    print_status "Stopping Secure FL services..."

    if [[ -n "$SERVICE" ]]; then
        compose_cmd stop "$SERVICE"
        print_success "Service $SERVICE stopped"
    else
        compose_cmd down
        print_success "All services stopped"
    fi
}

# Restart services
cmd_restart() {
    print_status "Restarting Secure FL services..."

    if [[ -n "$SERVICE" ]]; then
        compose_cmd restart "$SERVICE"
        print_success "Service $SERVICE restarted"
    else
        compose_cmd restart
        print_success "All services restarted"
    fi

    cmd_status
}

# Show service status
cmd_status() {
    print_status "Service Status:"
    compose_cmd ps
    echo ""

    # Show health status
    print_status "Health Status:"
    for container in $(docker-compose -f "$COMPOSE_FILE" ps -q); do
        if [[ -n "$container" ]]; then
            local name=$(docker inspect --format='{{.Name}}' "$container" | sed 's|^/||')
            local health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "no healthcheck")
            local status=$(docker inspect --format='{{.State.Status}}' "$container")
            echo "  $name: $status ($health)"
        fi
    done
}

# Show logs
cmd_logs() {
    local log_args=""
    if [[ "$FOLLOW_LOGS" == "true" ]]; then
        log_args="$log_args -f"
    fi

    if [[ -n "$SERVICE" ]]; then
        print_status "Showing logs for $SERVICE..."
        compose_cmd logs $log_args "$SERVICE"
    else
        print_status "Showing logs for all services..."
        compose_cmd logs $log_args
    fi
}

# Open shell in service
cmd_shell() {
    if [[ -z "$SERVICE" ]]; then
        SERVICE="secure-fl-dev"
        print_warning "No service specified, using $SERVICE"
    fi

    print_status "Opening shell in $SERVICE..."
    compose_cmd exec "$SERVICE" bash
}

# Execute command in service
cmd_exec() {
    if [[ -z "$SERVICE" ]]; then
        SERVICE="secure-fl-dev"
        print_warning "No service specified, using $SERVICE"
    fi

    if [[ -z "$ARGS" ]]; then
        print_error "No command specified for exec"
        exit 1
    fi

    print_status "Executing '$ARGS' in $SERVICE..."
    compose_cmd exec "$SERVICE" bash -c "$ARGS"
}

# Build images
cmd_build() {
    print_status "Building Secure FL images..."

    if [[ -n "$SERVICE" ]]; then
        compose_cmd build "$SERVICE"
        print_success "Service $SERVICE built"
    else
        compose_cmd build
        print_success "All services built"
    fi
}

# Pull images
cmd_pull() {
    print_status "Pulling latest images..."
    compose_cmd pull
    print_success "Images pulled"
}

# Scale services
cmd_scale() {
    if [[ -z "$SERVICE" ]]; then
        print_error "Service must be specified for scaling"
        exit 1
    fi

    if [[ -z "$SCALE_COUNT" ]]; then
        print_error "Scale count must be specified with --scale"
        exit 1
    fi

    print_status "Scaling $SERVICE to $SCALE_COUNT instances..."
    compose_cmd up -d --scale "$SERVICE=$SCALE_COUNT"
    print_success "Service $SERVICE scaled to $SCALE_COUNT instances"

    cmd_status
}

# Clean up
cmd_clean() {
    print_status "Cleaning up containers and volumes..."

    compose_cmd down -v --remove-orphans
    docker system prune -f

    print_success "Cleanup completed"
}

# Check health
cmd_health() {
    print_status "Checking service health..."

    local all_healthy=true
    for container in $(docker-compose -f "$COMPOSE_FILE" ps -q); do
        if [[ -n "$container" ]]; then
            local name=$(docker inspect --format='{{.Name}}' "$container" | sed 's|^/||')
            local health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null)
            local status=$(docker inspect --format='{{.State.Status}}' "$container")

            if [[ "$status" != "running" ]]; then
                print_error "$name is not running (status: $status)"
                all_healthy=false
            elif [[ "$health" == "unhealthy" ]]; then
                print_error "$name is unhealthy"
                all_healthy=false
            elif [[ "$health" == "healthy" ]]; then
                print_success "$name is healthy"
            else
                print_warning "$name has no health check (status: $status)"
            fi
        fi
    done

    if [[ "$all_healthy" == "true" ]]; then
        print_success "All services are healthy"
        return 0
    else
        print_error "Some services are not healthy"
        return 1
    fi
}

# Main execution
main() {
    print_banner

    # Parse arguments
    COMMAND=""
    ARGS=""
    parse_args "$@"

    if [[ -z "$COMMAND" ]]; then
        print_error "No command specified"
        usage
        exit 1
    fi

    # Check prerequisites
    check_prerequisites

    # Execute command
    case "$COMMAND" in
        up|start)
            cmd_up
            ;;
        down|stop)
            cmd_down
            ;;
        restart)
            cmd_restart
            ;;
        status|ps)
            cmd_status
            ;;
        logs)
            cmd_logs
            ;;
        shell)
            cmd_shell
            ;;
        exec)
            cmd_exec
            ;;
        build)
            cmd_build
            ;;
        pull)
            cmd_pull
            ;;
        scale)
            cmd_scale
            ;;
        clean)
            cmd_clean
            ;;
        health)
            cmd_health
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
