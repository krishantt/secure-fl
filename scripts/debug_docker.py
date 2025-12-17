#!/usr/bin/env python3
"""
Docker Debug Script for Secure FL

This script helps debug Docker networking and connectivity issues
for the Secure FL framework.
"""

import os
import socket
import subprocess
import time
from pathlib import Path


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


def run_command(cmd: str) -> str:
    """Run a shell command and return output"""
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Command timed out"
    except Exception as e:
        return f"Exception: {e}"


def check_network_info():
    """Check network configuration"""
    print_header("Network Information")

    # Hostname and IP
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"Hostname: {hostname}")
        print(f"Local IP: {local_ip}")
    except Exception as e:
        print(f"Error getting hostname/IP: {e}")

    # Network interfaces
    print("\nNetwork interfaces:")
    interfaces = run_command("ip addr show")
    for line in interfaces.split("\n")[:20]:  # Show first 20 lines
        if "inet" in line or "eth" in line or "lo" in line:
            print(f"  {line.strip()}")

    # Routing table
    print("\nRouting table:")
    routes = run_command("ip route")
    for line in routes.split("\n")[:10]:  # Show first 10 lines
        print(f"  {line}")


def check_dns_resolution():
    """Check DNS resolution"""
    print_header("DNS Resolution")

    hosts_to_test = [
        "localhost",
        "securefl-server",
        "securefl-client-1",
        "securefl-client-2",
        "google.com",
    ]

    for host in hosts_to_test:
        try:
            ip = socket.gethostbyname(host)
            print(f"✓ {host} -> {ip}")
        except Exception as e:
            print(f"✗ {host} -> {e}")


def check_port_connectivity():
    """Check port connectivity"""
    print_header("Port Connectivity")

    # Ports to test
    test_cases = [
        ("localhost", 8080, "Local server port"),
        ("0.0.0.0", 8080, "All interfaces"),
        ("securefl-server", 8080, "Server container"),
        ("google.com", 80, "External connectivity"),
    ]

    for host, port, description in test_cases:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                print(f"✓ {description}: {host}:{port} - OPEN")
            else:
                print(f"✗ {description}: {host}:{port} - CLOSED")
        except Exception as e:
            print(f"✗ {description}: {host}:{port} - ERROR: {e}")


def check_processes():
    """Check running processes"""
    print_header("Running Processes")

    # Check for Python processes
    processes = run_command("ps aux")
    python_processes = []
    for line in processes.split("\n"):
        if "python" in line.lower() and "ps aux" not in line:
            python_processes.append(line)

    if python_processes:
        print("Python processes:")
        for proc in python_processes[:10]:  # Show first 10
            print(f"  {proc}")
    else:
        print("No Python processes found")

    # Check for listening ports
    print("\nListening ports:")
    netstat = run_command("netstat -tlnp")
    for line in netstat.split("\n"):
        if ":8080" in line or ":8081" in line:
            print(f"  {line}")


def check_environment():
    """Check environment variables"""
    print_header("Environment Variables")

    env_vars = [
        "SECURE_FL_ENV",
        "SECURE_FL_SERVER_HOST",
        "SECURE_FL_SERVER_PORT",
        "SECURE_FL_CLIENT_ID",
        "SECURE_FL_SERVER_URL",
        "HOSTNAME",
        "PATH",
        "PYTHONPATH",
    ]

    for var in env_vars:
        value = os.getenv(var, "NOT SET")
        print(f"{var}: {value}")


def check_docker_info():
    """Check Docker-specific information"""
    print_header("Docker Information")

    # Check if we're in a container
    in_container = Path("/.dockerenv").exists()
    print(f"In Docker container: {in_container}")

    if in_container:
        # Docker ENV file
        dockerenv_content = ""
        try:
            with open("/.dockerenv") as f:
                dockerenv_content = f.read()
        except Exception:
            dockerenv_content = "Could not read /.dockerenv"

        print(f"/.dockerenv exists: {bool(dockerenv_content)}")

        # Check /etc/hosts
        print("\n/etc/hosts entries:")
        try:
            with open("/etc/hosts") as f:
                hosts_content = f.read()
            for line in hosts_content.split("\n")[:20]:  # Show first 20 lines
                if line.strip() and not line.startswith("#"):
                    print(f"  {line}")
        except Exception as e:
            print(f"Could not read /etc/hosts: {e}")


def test_flower_connection():
    """Test Flower client-server connection"""
    print_header("Flower Connection Test")

    try:
        import flwr as fl

        print(f"✓ Flower version: {fl.__version__}")

        # Test imports
        from flwr.client import start_numpy_client  # noqa:F401
        from flwr.server import start_server  # noqa:F401

        print("✓ Flower imports successful")

    except ImportError as e:
        print(f"✗ Flower import failed: {e}")
        return

    # Test basic connection (without actually starting client)
    server_addresses = ["localhost:8080", "securefl-server:8080"]

    for addr in server_addresses:
        try:
            host, port = addr.split(":")
            port = int(port)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                print(f"✓ Can connect to {addr}")
            else:
                print(f"✗ Cannot connect to {addr}")
        except Exception as e:
            print(f"✗ Error testing {addr}: {e}")


def test_secure_fl():
    """Test Secure FL imports"""
    print_header("Secure FL Test")

    try:
        from secure_fl import __version__

        print(f"✓ Secure FL version: {__version__}")

        from secure_fl.models import MNISTModel

        print("✓ Secure FL imports successful")

        # Test model creation
        model = MNISTModel()
        print(f"✓ Model created: {type(model).__name__}")

    except Exception as e:
        print(f"✗ Secure FL test failed: {e}")


def main():
    """Run all debug checks"""
    print("Secure FL Docker Debug Script")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all checks
    check_environment()
    check_docker_info()
    check_network_info()
    check_dns_resolution()
    check_port_connectivity()
    check_processes()
    test_flower_connection()
    test_secure_fl()

    print_header("Debug Complete")
    print("If you're experiencing connectivity issues, check:")
    print("1. Server is running and listening on 0.0.0.0:8080")
    print("2. DNS resolution for 'securefl-server' works")
    print("3. No firewall blocking connections")
    print("4. Docker network configuration is correct")
    print("5. All containers are on the same network")


if __name__ == "__main__":
    main()
