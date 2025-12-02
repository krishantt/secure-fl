"""
Secure Federated Learning Training Experiment

This script demonstrates the complete secure FL framework with dual ZKP verification.
It sets up multiple clients, runs federated training with FedJSCM aggregation,
and includes zk-STARK (client) and zk-SNARK (server) proof generation.

Usage:
    python train_secure_fl.py --config config.yaml
    python train_secure_fl.py --num-clients 5 --rounds 10 --dataset medmnist
"""

import argparse
import logging
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import List, Dict, Any, Tuple
import multiprocessing as mp
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Import our secure FL components
import sys

sys.path.append(str(Path(__file__).parent.parent))

from secure_fl.server import SecureFlowerServer, create_server_strategy
from secure_fl.client import create_client, start_client
from secure_fl.utils import get_parameter_stats
from secure_fl.quantization import compute_quantization_error

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleNN(nn.Module):
    """Simple neural network for federated learning experiments"""

    def __init__(
        self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SecureFL_Experiment:
    """Main experiment class for secure federated learning"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {
            "training_history": [],
            "client_metrics": [],
            "proof_metrics": [],
            "stability_metrics": [],
            "final_accuracy": 0.0,
            "total_time": 0.0,
        }

        # Setup experiment parameters
        self.num_clients = config.get("num_clients", 5)
        self.num_rounds = config.get("num_rounds", 10)
        self.dataset_name = config.get("dataset", "synthetic")
        self.enable_zkp = config.get("enable_zkp", True)
        self.proof_rigor = config.get("proof_rigor", "high")

        logger.info(
            f"Initialized SecureFL experiment with {self.num_clients} clients, {self.num_rounds} rounds"
        )
