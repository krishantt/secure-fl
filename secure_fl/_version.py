"""
Simple development versioning for Secure FL

This module provides development version generation with build numbers.
Format: YYYY.MM.DD.dev.INCREMENT
"""

import os
from datetime import datetime
from pathlib import Path

# Static version for hatchling - will be overridden by get_version()
__version__ = "2024.12.20.dev.1"


def get_version() -> str:
    """
    Generate development version string.

    Format: YYYY.MM.DD.dev.INCREMENT

    Examples:
        - 2024.12.03.dev.1
        - 2024.12.03.dev.5

    Returns:
        Version string with date and build increment
    """
    # Current date
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day

    # Get build increment from file or default to 1
    increment = _get_build_increment()

    # Always use dev format
    version = f"{year}.{month}.{day}.dev.{increment}"

    return version


def _get_build_increment():
    """
    Get current build increment from file.

    Returns:
        Current build increment number
    """
    build_file = Path(__file__).parent / ".build_number"
    current_date = datetime.now().strftime("%Y.%m.%d")

    if build_file.exists():
        try:
            content = build_file.read_text().strip()
            if content:
                stored_date, stored_increment = content.split(":", 1)
                # Handle both old format (2025.12.03) and new format (2025.12.3)
                stored_date_normalized = stored_date.replace(".0", ".")
                current_date_normalized = current_date.replace(".0", ".")
                if stored_date_normalized == current_date_normalized:
                    return int(stored_increment)
        except (ValueError, FileNotFoundError):
            pass

    # Default to 1 for new dates or errors
    return 1


def get_build_info() -> dict:
    """
    Get build information.

    Returns:
        Dictionary with build details
    """
    now = datetime.now()

    return {
        "version": get_version(),
        "build_date": now.strftime("%Y-%m-%d"),
        "build_time": now.strftime("%H:%M:%S"),
        "build_timestamp": now.isoformat(),
        "build_increment": _get_build_increment(),
        "is_dev": True,  # Always dev
        "git_ref": os.getenv("GITHUB_REF", "unknown"),
        "git_sha": os.getenv("GITHUB_SHA", "unknown"),
    }


def print_version_info():
    """Print version information."""
    info = get_build_info()

    print(f"Secure FL Version: {info['version']}")
    print(f"Build Date: {info['build_date']} {info['build_time']}")
    print(f"Build Increment: {info['build_increment']}")
    print("Build Type: Development")


def increment_build_number():
    """
    Increment build increment for publishing.

    This function is called only when publishing to PyPI.
    It reads the current increment from a file and increments it.
    """
    build_file = Path(__file__).parent / ".build_number"
    current_date = datetime.now().strftime("%Y.%m.%d")

    increment = 1

    if build_file.exists():
        try:
            content = build_file.read_text().strip()
            if content:
                stored_date, stored_increment = content.split(":", 1)
                # Handle both old format (2025.12.03) and new format (2025.12.3)
                stored_date_normalized = stored_date.replace(".0", ".")
                current_date_normalized = current_date.replace(".0", ".")
                if stored_date_normalized == current_date_normalized:
                    increment = int(stored_increment) + 1
        except (ValueError, FileNotFoundError):
            increment = 1

    # Save new increment with normalized format
    build_file.write_text(f"{current_date}:{increment}")

    return str(increment)


def set_build_number(number: int):
    """
    Set specific build increment for current date.

    Args:
        number: Build increment to set
    """
    build_file = Path(__file__).parent / ".build_number"
    current_date = datetime.now().strftime("%Y.%m.%d")

    build_file.write_text(f"{current_date}:{number}")


# Override with dynamic version at import time
__version__ = get_version()

# Expose main functions
__all__ = [
    "__version__",
    "get_version",
    "get_build_info",
    "print_version_info",
    "increment_build_number",
    "set_build_number",
]
