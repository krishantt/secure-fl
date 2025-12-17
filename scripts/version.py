#!/usr/bin/env python3
"""
Version management script for Secure FL

This script helps manage versions during development.
It provides commands to display, increment, and set version information.

Usage:
    python scripts/version.py show          # Show current version
    python scripts/version.py increment     # Increment build increment (for publishing)
    python scripts/version.py set <number>  # Set specific build increment
    python scripts/version.py publish       # Increment and prepare for publishing
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from secure_fl._version import (  # noqa: E402
    get_build_info,
    get_version,
    increment_build_number,
    print_version_info,
    set_build_number,
)


def show_version(format_type=None):
    """Display current version information."""
    if format_type == "version":
        # Just output the version string for CI/scripting
        print(get_version())
        return

    print("=== Secure FL Version Information ===")
    print_version_info()

    info = get_build_info()
    print("\nDetailed Information:")
    print(f"  Full Version: {info['version']}")
    print("  Build Environment: Development")
    print(f"  Build Timestamp: {info['build_timestamp']}")
    print(f"  Build Increment: {info['build_increment']}")


def increment_version():
    """Increment the build increment for publishing."""
    old_version = get_version()
    new_increment = increment_build_number()

    new_version = get_version()

    print("Version incremented for publishing:")
    print(f"  Old: {old_version}")
    print(f"  New: {new_version}")
    print(f"  Build Increment: {new_increment}")
    print("\nThis increment should only be used when publishing to PyPI!")


def set_version(increment_number):
    """Set a specific build increment."""
    try:
        increment_num = int(increment_number)
        if increment_num < 1:
            raise ValueError("Build increment must be positive")
    except ValueError as e:
        print(f"Error: Invalid build increment '{increment_number}': {e}")
        return False

    old_version = get_version()
    set_build_number(increment_num)

    new_version = get_version()

    print("Version updated:")
    print(f"  Old: {old_version}")
    print(f"  New: {new_version}")
    print(f"  Build Increment: {increment_num}")

    return True


def prepare_publish():
    """Prepare version for publishing to PyPI."""
    print("=== Preparing for PyPI Publishing ===")

    # Show current version
    current_version = get_version()
    print(f"Current version: {current_version}")

    # Check if we're in a clean state
    build_file = Path(project_root) / "secure_fl" / ".build_number"
    if build_file.exists():
        content = build_file.read_text().strip()
        print(f"Build file exists: {content}")
    else:
        print("No build file found - will start from increment 1")

    # Increment for publishing
    increment_version()

    publish_version = get_version()
    print(f"\nVersion ready for publishing: {publish_version}")

    print("\nTo publish to PyPI:")
    print("1. Build package: pdm build")
    print("2. Publish: pdm publish")
    print("3. The increment will be preserved for this date")


def show_build_info():
    """Show build file information."""
    print("=== Build Information ===")

    build_file = Path(project_root) / "secure_fl" / ".build_number"

    print(f"Build file path: {build_file}")

    if build_file.exists():
        content = build_file.read_text().strip()
        print(f"Build file content: {content}")

        try:
            date_part, increment_part = content.split(":", 1)
            print(f"  Date: {date_part}")
            print(f"  Increment: {increment_part}")
        except ValueError:
            print("  Invalid format in build file")
    else:
        print("Build file does not exist")

    print(f"\nCurrent version: {get_version()}")


def build_package():
    """Build the package and show version."""
    print("=== Building Package ===")

    # Import PDM programmatically if available
    try:
        import subprocess

        print("Running PDM build...")
        result = subprocess.run(
            ["pdm", "build"], cwd=project_root, capture_output=True, text=True
        )

        if result.returncode == 0:
            print("Build successful!")
            print(f"Version: {get_version()}")

            # List built files
            dist_dir = project_root / "dist"
            if dist_dir.exists():
                print("\nBuilt files:")
                for file in dist_dir.glob("*"):
                    print(f"  {file.name}")
        else:
            print(f"Build failed: {result.stderr}")

    except FileNotFoundError:
        print("PDM not found. Please install PDM first.")
    except Exception as e:
        print(f"Build error: {e}")


def main():
    """Main command line interface."""
    parser = argparse.ArgumentParser(
        description="Secure FL Version Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show current version information")
    show_parser.add_argument(
        "--format",
        choices=["version"],
        help="Output format (version: just the version string)",
    )

    # Increment command
    subparsers.add_parser("increment", help="Increment build increment for publishing")

    # Set command
    set_parser = subparsers.add_parser("set", help="Set specific build increment")
    set_parser.add_argument("number", help="Build increment to set")

    # Publish command
    subparsers.add_parser("publish", help="Prepare for PyPI publishing")

    # Build info command
    subparsers.add_parser("build-info", help="Show build file information")

    # Build command
    subparsers.add_parser("build", help="Build package with version info")

    args = parser.parse_args()

    if args.command == "show":
        show_version(getattr(args, "format", None))
    elif args.command == "increment":
        increment_version()
    elif args.command == "set":
        set_version(args.number)
    elif args.command == "publish":
        prepare_publish()
    elif args.command == "build-info":
        show_build_info()
    elif args.command == "build":
        build_package()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
