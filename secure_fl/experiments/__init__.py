"""
Experiments Module for Secure FL

This module contains experiment scripts, demos, and benchmarking utilities
for the Secure FL framework.
"""

from .demo import run_demo, run_quick_demo
from .train_secure_fl import main as train_main

__all__ = [
    "run_demo",
    "run_quick_demo",
    "train_main",
]
