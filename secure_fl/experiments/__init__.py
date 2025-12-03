"""
Experiments Module for Secure FL

This module contains experiment scripts, demos, and benchmarking utilities
for the Secure FL framework.
"""

from .benchmark import main as benchmark_main
from .demo import run_demo, run_quick_demo
from .train import main as train_main

__all__ = [
    "run_demo",
    "run_quick_demo",
    "train_main",
    "benchmark_main",
]
