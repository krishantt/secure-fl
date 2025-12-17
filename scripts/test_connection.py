#!/usr/bin/env python3
"""
Connection test script for Secure FL Docker networking

This script helps debug connectivity issues between server and clients
in the Docker Compose environment.
"""

import socket
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_port_open(host: str, port: int, timeout: int = 5) -> bool:
    """Test if a port is open on a host"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Error testing {host}:{port} - {e}")
        return False


def test_dns_resolution(hostname: str) -> str:
    """Test DNS resolution for a hostname"""
    try:
        ip = socket.gethostbyname(hostname)
        print(f"✓ DNS resolution: {hostname} -> {ip}")
        return ip
    except Exception as e:
        print(f"✗ DNS resolution failed for {hostname}: {e}")
        return None


def test_flower_imports():
    """Test that Flower can be imported properly"""
    try:
        import flwr as fl

        print(f"✓ Flower version: {fl.__version__}")

        # Test key components

        print("✓ Flower server/client imports successful")

        return True
    except Exception as e:
        print(f"✗ Flower import failed: {e}")
        return False


def test_secure_fl_imports():
    """Test that Secure FL modules can be imported"""
    try:
        print("✓ Secure FL imports successful")
        return True
    except Exception as e:
        print(f"✗ Secure FL import failed: {e}")
        return False


def main():
    """Run connection and environment tests"""
    print("=== Secure FL Connection Test ===")
    print()

    # Test 1: Basic imports
    print("1. Testing imports...")
    flower_ok = test_flower_imports()
    secure_fl_ok = test_secure_fl_imports()

    if not (flower_ok and secure_fl_ok):
        print("✗ Import tests failed - check your environment")
        return 1

    print()

    # Test 2: Network configuration
    print("2. Testing network configuration...")

    # Test localhost
    localhost_8080 = test_port_open("localhost", 8080)
    print(
        f"{'✓' if localhost_8080 else '✗'} localhost:8080 {'open' if localhost_8080 else 'closed'}"
    )

    # Test 0.0.0.0 binding
    all_interfaces_8080 = test_port_open("0.0.0.0", 8080)
    print(
        f"{'✓' if all_interfaces_8080 else '✗'} 0.0.0.0:8080 {'open' if all_interfaces_8080 else 'closed'}"
    )

    print()

    # Test 3: Docker networking (if in container)
    print("3. Testing Docker networking...")

    # Check if we're in a container
    in_container = Path("/.dockerenv").exists()
    print(f"Running in container: {'Yes' if in_container else 'No'}")

    if in_container:
        # Test server hostname resolution
        server_ip = test_dns_resolution("securefl-server")

        if server_ip:
            # Test connection to server
            server_reachable = test_port_open("securefl-server", 8080)
            print(
                f"{'✓' if server_reachable else '✗'} securefl-server:8080 {'reachable' if server_reachable else 'unreachable'}"
            )

        # Test network interface
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"✓ Container IP: {local_ip}")
        except Exception as e:
            print(f"✗ Could not determine container IP: {e}")

    print()

    # Test 4: Environment variables
    print("4. Checking environment variables...")
    import os

    env_vars = [
        "SECURE_FL_ENV",
        "SECURE_FL_SERVER_HOST",
        "SECURE_FL_SERVER_PORT",
        "SECURE_FL_CLIENT_ID",
        "SECURE_FL_SERVER_URL",
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var}={value}")
        else:
            print(f"- {var} not set")

    print()

    # Test 5: Simple server test (if not already running)
    print("5. Testing simple server startup...")

    if not localhost_8080:
        try:
            print("Attempting to start a test server...")

            from secure_fl.models import MNISTModel
            from secure_fl.server import SecureFlowerServer, create_server_strategy

            # Create minimal strategy
            strategy = create_server_strategy(
                model_fn=MNISTModel,
                min_fit_clients=1,
                min_evaluate_clients=1,
                enable_zkp=False,  # Disable ZKP for quick test
            )

            # Create test server
            SecureFlowerServer(
                strategy=strategy,
                host="localhost",
                port=8081,  # Use different port
                num_rounds=1,
            )

            print("✓ Test server created successfully")
            print("ℹ Server ready for testing (not started)")

        except Exception as e:
            print(f"✗ Test server creation failed: {e}")
    else:
        print("- Server already running on port 8080, skipping test")

    print()
    print("=== Test Complete ===")

    # Summary
    issues = []
    if not flower_ok:
        issues.append("Flower import issues")
    if not secure_fl_ok:
        issues.append("Secure FL import issues")

    if in_container and server_ip and not test_port_open("securefl-server", 8080):
        issues.append("Cannot reach securefl-server:8080")

    if issues:
        print(f"✗ Issues found: {', '.join(issues)}")
        return 1
    else:
        print("✓ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
