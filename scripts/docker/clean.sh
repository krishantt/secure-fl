#!/bin/bash

# Docker cleanup script for Secure FL
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
    echo "║              Secure FL Docker Cleanup               ║"
    echo "║         Clean containers, images, and volumes        ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Default values
CLEAN_ALL=false
CLEAN_CONTAINERS=false
CLEAN_IMAGES=false
CLEAN_VOLUMES=false
CLEAN_NETWORKS=false
CLEAN_CACHE=false
FORCE=false
DRY_RUN=false

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Docker cleanup script for Secure FL project"
    echo ""
    echo "Options:"
    echo "  --all               Clean everything (containers, images, volumes, networks, cache)"
    echo "  --containers        Clean stopped containers"
    echo "  --images            Clean unused images"
    echo "  --volumes           Clean unused volumes"
    echo "  --networks          Clean unused networks"
    echo "  --cache             Clean build cache"
    echo "  --secure-fl-only    Only clean Secure FL related resources"
    echo "  -f, --force         Force cleanup without confirmation"
    echo "  --dry-run           Show what would be cleaned without actually doing it"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all            # Clean everything"
    echo "  $0 --containers     # Clean only stopped containers"
    echo "  $0 --images --force # Clean images without confirmation"
    echo "  $0 --dry-run --all  # Show what would be cleaned"
    echo ""
    echo "Safe defaults (if no options specified):"
    echo "  - Clean stopped containers"
    echo "  - Clean dangling images"
    echo "  - Clean unused volumes"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            CLEAN_ALL=true
            shift
            ;;
        --containers)
            CLEAN_CONTAINERS=true
            shift
            ;;
        --images)
            CLEAN_IMAGES=true
            shift
            ;;
        --volumes)
            CLEAN_VOLUMES=true
            shift
            ;;
        --networks)
            CLEAN_NETWORKS=true
            shift
            ;;
        --cache)
            CLEAN_CACHE=true
            shift
            ;;
        --secure-fl-only)
            SECURE_FL_ONLY=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
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

# Set defaults if no specific options were provided
if [[ "$CLEAN_ALL" == "false" && "$CLEAN_CONTAINERS" == "false" && "$CLEAN_IMAGES" == "false" && "$CLEAN_VOLUMES" == "false" && "$CLEAN_NETWORKS" == "false" && "$CLEAN_CACHE" == "false" ]]; then
    CLEAN_CONTAINERS=true
    CLEAN_IMAGES=true
    CLEAN_VOLUMES=true
    print_warning "No specific cleanup options specified, using safe defaults"
fi

# If --all is specified, enable everything
if [[ "$CLEAN_ALL" == "true" ]]; then
    CLEAN_CONTAINERS=true
    CLEAN_IMAGES=true
    CLEAN_VOLUMES=true
    CLEAN_NETWORKS=true
    CLEAN_CACHE=true
fi

# Check if Docker is running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi

    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running"
        exit 1
    fi
}

# Get confirmation from user
confirm_action() {
    if [[ "$FORCE" == "true" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi

    echo ""
    echo "This will clean up the following Docker resources:"
    [[ "$CLEAN_CONTAINERS" == "true" ]] && echo "  - Stopped containers"
    [[ "$CLEAN_IMAGES" == "true" ]] && echo "  - Unused images"
    [[ "$CLEAN_VOLUMES" == "true" ]] && echo "  - Unused volumes"
    [[ "$CLEAN_NETWORKS" == "true" ]] && echo "  - Unused networks"
    [[ "$CLEAN_CACHE" == "true" ]] && echo "  - Build cache"
    echo ""

    read -p "Are you sure you want to continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Cleanup cancelled"
        exit 0
    fi
}

# Execute command with dry-run support
execute_cmd() {
    local cmd="$1"
    local description="$2"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "[DRY RUN] Would execute: $cmd"
        print_status "[DRY RUN] Purpose: $description"
    else
        print_status "$description"
        eval "$cmd"
    fi
}

# Stop and remove Secure FL containers
stop_secure_fl_containers() {
    print_status "Stopping Secure FL containers..."

    # Stop containers from docker-compose
    if [[ -f "docker-compose.yml" ]]; then
        execute_cmd "docker-compose down" "Stop docker-compose services"
    fi

    # Stop any running secure-fl containers
    local containers=$(docker ps -q --filter "name=secure-fl" 2>/dev/null || true)
    if [[ -n "$containers" ]]; then
        execute_cmd "docker stop $containers" "Stop running Secure FL containers"
    fi
}

# Clean containers
clean_containers() {
    if [[ "$CLEAN_CONTAINERS" != "true" ]]; then
        return
    fi

    print_status "Cleaning containers..."

    if [[ "$SECURE_FL_ONLY" == "true" ]]; then
        # Remove only Secure FL containers
        local containers=$(docker ps -aq --filter "name=secure-fl" 2>/dev/null || true)
        if [[ -n "$containers" ]]; then
            execute_cmd "docker rm $containers" "Remove Secure FL containers"
        else
            print_status "No Secure FL containers found"
        fi
    else
        # Remove all stopped containers
        local containers=$(docker ps -aq --filter "status=exited" 2>/dev/null || true)
        if [[ -n "$containers" ]]; then
            execute_cmd "docker rm $containers" "Remove stopped containers"
        else
            print_status "No stopped containers found"
        fi
    fi
}

# Clean images
clean_images() {
    if [[ "$CLEAN_IMAGES" != "true" ]]; then
        return
    fi

    print_status "Cleaning images..."

    if [[ "$SECURE_FL_ONLY" == "true" ]]; then
        # Remove only Secure FL images
        local images=$(docker images -q "secure-fl" 2>/dev/null || true)
        if [[ -n "$images" ]]; then
            execute_cmd "docker rmi $images" "Remove Secure FL images"
        else
            print_status "No Secure FL images found"
        fi
    else
        # Remove dangling images
        local dangling=$(docker images -q --filter "dangling=true" 2>/dev/null || true)
        if [[ -n "$dangling" ]]; then
            execute_cmd "docker rmi $dangling" "Remove dangling images"
        else
            print_status "No dangling images found"
        fi

        # Remove unused images (be more aggressive if --all is specified)
        if [[ "$CLEAN_ALL" == "true" ]]; then
            execute_cmd "docker image prune -a -f" "Remove all unused images"
        else
            execute_cmd "docker image prune -f" "Remove unused images"
        fi
    fi
}

# Clean volumes
clean_volumes() {
    if [[ "$CLEAN_VOLUMES" != "true" ]]; then
        return
    fi

    print_status "Cleaning volumes..."

    if [[ "$SECURE_FL_ONLY" == "true" ]]; then
        # Remove only Secure FL volumes
        local volumes=$(docker volume ls -q --filter "name=secure-fl" 2>/dev/null || true)
        if [[ -n "$volumes" ]]; then
            execute_cmd "docker volume rm $volumes" "Remove Secure FL volumes"
        else
            print_status "No Secure FL volumes found"
        fi
    else
        # Remove unused volumes
        execute_cmd "docker volume prune -f" "Remove unused volumes"
    fi
}

# Clean networks
clean_networks() {
    if [[ "$CLEAN_NETWORKS" != "true" ]]; then
        return
    fi

    print_status "Cleaning networks..."

    if [[ "$SECURE_FL_ONLY" == "true" ]]; then
        # Remove only Secure FL networks
        local networks=$(docker network ls -q --filter "name=secure-fl" 2>/dev/null || true)
        if [[ -n "$networks" ]]; then
            execute_cmd "docker network rm $networks" "Remove Secure FL networks"
        else
            print_status "No Secure FL networks found"
        fi
    else
        # Remove unused networks
        execute_cmd "docker network prune -f" "Remove unused networks"
    fi
}

# Clean build cache
clean_cache() {
    if [[ "$CLEAN_CACHE" != "true" ]]; then
        return
    fi

    print_status "Cleaning build cache..."
    execute_cmd "docker builder prune -f" "Remove build cache"
}

# Show disk space before and after
show_disk_usage() {
    local when="$1"
    print_status "Docker disk usage ($when cleanup):"
    if [[ "$DRY_RUN" != "true" ]]; then
        docker system df
    else
        print_status "[DRY RUN] Would show: docker system df"
    fi
    echo ""
}

# Main cleanup function
main() {
    print_banner

    check_docker

    # Show disk usage before
    show_disk_usage "before"

    # Get confirmation
    confirm_action

    # Stop Secure FL containers first
    stop_secure_fl_containers

    # Perform cleanup operations
    clean_containers
    clean_images
    clean_volumes
    clean_networks
    clean_cache

    # Show results
    if [[ "$DRY_RUN" == "true" ]]; then
        print_success "Dry run completed - no changes were made"
    else
        print_success "Docker cleanup completed!"
        echo ""
        show_disk_usage "after"
    fi

    # Provide some helpful tips
    echo ""
    print_status "Tips for maintaining clean Docker environment:"
    echo "  • Run this script periodically to free up disk space"
    echo "  • Use 'docker system df' to monitor disk usage"
    echo "  • Use '--dry-run' to preview changes before executing"
    echo "  • Use '--secure-fl-only' to clean only project-related resources"
    echo "  • Consider setting up automated cleanup with cron jobs"
}

# Run main function
main "$@"
