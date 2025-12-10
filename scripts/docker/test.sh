#!/bin/bash

# Docker test verification script for Secure FL
# Verifies that Docker images are working correctly
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

# Test configuration
IMAGES_TO_TEST=("base" "development" "test")
FAILED_TESTS=()
PASSED_TESTS=()

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Test Docker images for Secure FL to verify they work correctly"
    echo ""
    echo "Options:"
    echo "  -i, --image IMAGE      Test specific image (base, development, test, server, client)"
    echo "  -a, --all              Test all available images"
    echo "  -q, --quick            Run quick tests only"
    echo "  -f, --full             Run full test suite"
    echo "  -v, --verbose          Verbose output"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Test default images"
    echo "  $0 -i test             # Test only the test image"
    echo "  $0 -a                  # Test all available images"
    echo "  $0 -f -i development   # Full test of development image"
}

# Check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Check if image exists
check_image_exists() {
    local image=$1
    if ! docker images secure-fl:$image --format "{{.Repository}}" | grep -q secure-fl; then
        print_error "Image secure-fl:$image not found. Build it first with:"
        print_error "  ./scripts/docker/build.sh -t $image"
        return 1
    fi
    return 0
}

# Test basic functionality
test_basic_functionality() {
    local image=$1
    print_status "Testing basic functionality of secure-fl:$image"

    # Test Python import
    if docker run --rm secure-fl:$image python -c "import secure_fl; print('✓ Package import successful')" >/dev/null 2>&1; then
        print_success "Python import test passed"
    else
        print_error "Python import test failed"
        return 1
    fi

    # Test CLI help
    if docker run --rm secure-fl:$image uv run secure-fl --help >/dev/null 2>&1; then
        print_success "CLI help test passed"
    else
        print_error "CLI help test failed"
        return 1
    fi

    # Test version display
    if docker run --rm secure-fl:$image uv run python scripts/version.py show >/dev/null 2>&1; then
        print_success "Version display test passed"
    else
        print_warning "Version display test failed (non-critical)"
    fi

    return 0
}

# Test development image specific features
test_development_image() {
    local image=$1
    print_status "Testing development features of secure-fl:$image"

    # Test that tests directory exists
    if docker run --rm secure-fl:$image test -d tests; then
        print_success "Tests directory exists"
    else
        print_error "Tests directory not found"
        return 1
    fi

    # Test pytest is available
    if docker run --rm secure-fl:$image uv run pytest --version >/dev/null 2>&1; then
        print_success "Pytest is available"
    else
        print_error "Pytest not available"
        return 1
    fi

    # Test development dependencies
    if docker run --rm secure-fl:$image uv run python -c "import mypy, ruff" >/dev/null 2>&1; then
        print_success "Development dependencies available"
    else
        print_error "Development dependencies missing"
        return 1
    fi

    return 0
}

# Test that can run actual tests
test_run_tests() {
    local image=$1
    print_status "Running actual test suite in secure-fl:$image"

    # Run a quick subset of tests
    if docker run --rm secure-fl:$image uv run pytest tests/unit/ -x -q --tb=no >/dev/null 2>&1; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed"
        return 1
    fi

    return 0
}

# Test specific image configurations
test_server_image() {
    print_status "Testing server image configuration"

    # Test server help
    if docker run --rm secure-fl:server uv run secure-fl server --help >/dev/null 2>&1; then
        print_success "Server command available"
    else
        print_error "Server command failed"
        return 1
    fi

    return 0
}

test_client_image() {
    print_status "Testing client image configuration"

    # Test client help
    if docker run --rm secure-fl:client uv run secure-fl client --help >/dev/null 2>&1; then
        print_success "Client command available"
    else
        print_error "Client command failed"
        return 1
    fi

    return 0
}

# Test ZKP tools availability
test_zkp_tools() {
    local image=$1
    print_status "Testing ZKP tools availability in secure-fl:$image"

    # Test Node.js
    if docker run --rm secure-fl:$image node --version >/dev/null 2>&1; then
        print_success "Node.js available"
    else
        print_warning "Node.js not available (may be expected for some images)"
    fi

    # Test Circom
    if docker run --rm secure-fl:$image circom --version >/dev/null 2>&1; then
        print_success "Circom available"
    else
        print_warning "Circom not available (may be expected for some images)"
    fi

    # Test SnarkJS
    if docker run --rm secure-fl:$image snarkjs help >/dev/null 2>&1; then
        print_success "SnarkJS available"
    else
        print_warning "SnarkJS not available (may be expected for some images)"
    fi

    return 0
}

# Test system check
test_system_check() {
    local image=$1
    print_status "Testing system check in secure-fl:$image"

    if docker run --rm secure-fl:$image uv run secure-fl setup check >/dev/null 2>&1; then
        print_success "System check passed"
    else
        print_warning "System check failed (may have warnings)"
    fi

    return 0
}

# Run comprehensive test for an image
test_image() {
    local image=$1
    local quick_mode=${2:-false}

    print_status "Testing secure-fl:$image image..."

    # Check if image exists
    if ! check_image_exists $image; then
        FAILED_TESTS+=("$image")
        return 1
    fi

    local test_failed=false

    # Basic functionality tests (always run)
    if ! test_basic_functionality $image; then
        test_failed=true
    fi

    # Image-specific tests
    case $image in
        development|test)
            if ! test_development_image $image; then
                test_failed=true
            fi

            if [[ "$quick_mode" != "true" ]]; then
                if ! test_run_tests $image; then
                    test_failed=true
                fi
            fi
            ;;
        server)
            if ! test_server_image; then
                test_failed=true
            fi
            ;;
        client)
            if ! test_client_image; then
                test_failed=true
            fi
            ;;
    esac

    # ZKP tools test (for development and base images)
    if [[ "$image" == "development" || "$image" == "base" ]]; then
        test_zkp_tools $image
    fi

    # System check (non-critical)
    test_system_check $image

    if [[ "$test_failed" == "true" ]]; then
        FAILED_TESTS+=("$image")
        print_error "Tests failed for secure-fl:$image"
        return 1
    else
        PASSED_TESTS+=("$image")
        print_success "All tests passed for secure-fl:$image"
        return 0
    fi
}

# Main function
main() {
    local target_image=""
    local test_all=false
    local quick_mode=false
    local full_mode=false
    local verbose=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--image)
                target_image="$2"
                shift 2
                ;;
            -a|--all)
                test_all=true
                shift
                ;;
            -q|--quick)
                quick_mode=true
                shift
                ;;
            -f|--full)
                full_mode=true
                shift
                ;;
            -v|--verbose)
                verbose=true
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

    # Check Docker
    check_docker

    # Determine which images to test
    local images_to_test=()

    if [[ "$test_all" == "true" ]]; then
        # Get all available secure-fl images
        mapfile -t images_to_test < <(docker images secure-fl --format "{{.Tag}}" | grep -v '<none>' | sort)
        if [[ ${#images_to_test[@]} -eq 0 ]]; then
            print_error "No secure-fl images found. Build some first."
            exit 1
        fi
    elif [[ -n "$target_image" ]]; then
        images_to_test=("$target_image")
    else
        images_to_test=("${IMAGES_TO_TEST[@]}")
    fi

    print_status "Starting Docker image verification..."
    print_status "Images to test: ${images_to_test[*]}"
    echo ""

    # Test each image
    for image in "${images_to_test[@]}"; do
        echo "----------------------------------------"
        if test_image "$image" "$quick_mode"; then
            echo ""
        else
            echo ""
        fi
    done

    # Summary
    echo "========================================"
    print_status "TEST SUMMARY"
    echo ""

    if [[ ${#PASSED_TESTS[@]} -gt 0 ]]; then
        print_success "Passed: ${PASSED_TESTS[*]}"
    fi

    if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
        print_error "Failed: ${FAILED_TESTS[*]}"
        echo ""
        print_error "Some tests failed. Check the output above for details."
        print_error "You may need to rebuild images or check dependencies."
        exit 1
    else
        echo ""
        print_success "All Docker images are working correctly!"
        print_success "Your Docker setup is ready for Secure FL development."

        echo ""
        print_status "Next steps:"
        echo "  # Run a demo:"
        echo "  docker run --rm -v \$(pwd)/data:/home/app/data secure-fl:base uv run python experiments/demo.py"
        echo ""
        echo "  # Start development:"
        echo "  docker run --rm -it -p 8080:8080 -v \$(pwd):/home/app/workspace secure-fl:development"
        echo ""
        echo "  # Run full test suite:"
        echo "  docker run --rm secure-fl:test"
        exit 0
    fi
}

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -f "Dockerfile" ]]; then
    print_error "This script must be run from the secure-fl project root directory"
    exit 1
fi

# Run main function
main "$@"
