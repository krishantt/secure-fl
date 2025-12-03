"""
Main entry point for Secure FL CLI

This allows the package to be executed as a module:
python -m secure_fl <command>
"""

from .cli import main

if __name__ == "__main__":
    main()
