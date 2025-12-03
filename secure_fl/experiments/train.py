"""
Training Entry Point for Secure FL

This module provides a simplified entry point for training experiments
that matches the PDM script expectations.
"""

import argparse
import sys
from pathlib import Path

# Import the main training function
from .train_secure_fl import main as train_secure_fl_main


def main():
    """Main entry point for training experiments"""

    # Parse arguments to determine dataset and pass through
    parser = argparse.ArgumentParser(description="Secure FL Training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "mnist", "cifar10", "medmnist"],
        help="Dataset to use for training",
    )

    # Parse known args and pass the rest to the main training script
    args, unknown_args = parser.parse_known_args()

    # Reconstruct sys.argv for the main training script
    new_argv = [sys.argv[0]]  # Keep script name

    # Add dataset argument
    new_argv.extend(["--dataset", args.dataset])

    # Add any other arguments passed through
    new_argv.extend(unknown_args)

    # Temporarily replace sys.argv
    original_argv = sys.argv
    sys.argv = new_argv

    try:
        # Call the main training function
        train_secure_fl_main()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()
