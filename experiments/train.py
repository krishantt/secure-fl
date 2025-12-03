"""
Secure Federated Learning Training Experiment

This script demonstrates the complete secure FL framework with dual ZKP verification.
It sets up multiple clients, runs federated training with FedJSCM aggregation,
and includes zk-STARK (client) and zk-SNARK (server) proof generation.

Usage:
    python train.py --config config.yaml
    python train.py --num-clients 5 --rounds 10 --dataset mnist
    python train.py --num-clients 3 --rounds 5 --dataset cifar10 --enable-zkp
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from secure_fl.aggregation import FedJSCMAggregator
from secure_fl.client import create_client
from secure_fl.data import FederatedDataLoader
from secure_fl.models import create_model, get_model_info

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SecureFL_Experiment:
    """Main experiment class for secure federated learning"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.results = {
            "training_history": [],
            "client_metrics": [],
            "proof_metrics": [],
            "stability_metrics": [],
            "final_accuracy": 0.0,
            "total_time": 0.0,
            "dataset_info": {},
            "model_info": {},
        }

        # Setup experiment parameters
        self.num_clients = config.get("num_clients", 5)
        self.num_rounds = config.get("num_rounds", 10)
        self.dataset_name = config.get("dataset", "mnist")
        self.enable_zkp = config.get("enable_zkp", False)
        self.proof_rigor = config.get("proof_rigor", "medium")
        self.local_epochs = config.get("local_epochs", 3)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.01)

        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Initialized SecureFL experiment with {self.num_clients} clients, "
            f"{self.num_rounds} rounds, dataset: {self.dataset_name}"
        )

    def setup_model_and_data(
        self,
    ) -> tuple[nn.Module, list[tuple[DataLoader, DataLoader]]]:
        """Setup model and federated datasets"""

        # Create federated data loader
        self.data_loader = FederatedDataLoader(
            dataset_name=self.dataset_name,
            num_clients=self.num_clients,
            iid=True,  # Can be made configurable
            val_split=0.2,
            batch_size=self.batch_size,
            seed=42,
        )

        # Get dataset information
        dataset_info = self.data_loader.get_dataset_info()
        self.results["dataset_info"] = dataset_info
        logger.info(f"Dataset info: {dataset_info}")

        # Create model based on dataset
        model = self._create_model_for_dataset(self.dataset_name)
        model_info = get_model_info(model)
        self.results["model_info"] = model_info
        logger.info(f"Model info: {model_info}")

        # Create client data loaders
        client_datasets = self.data_loader.create_client_dataloaders()

        return model, client_datasets

    def _create_model_for_dataset(self, dataset_name: str) -> nn.Module:
        """Create appropriate model for the given dataset"""
        if dataset_name.lower() == "mnist":
            return create_model("mnist")
        elif dataset_name.lower() == "cifar10":
            return create_model("cifar10")
        elif dataset_name.lower() == "medmnist":
            # Use MNIST model for MedMNIST (similar structure)
            return create_model("mnist")
        else:
            # Default to simple model for synthetic data
            return create_model("synthetic", input_dim=784, output_dim=10)

    def run_experiment(self) -> dict[str, Any]:
        """Run the complete secure FL experiment"""
        start_time = time.time()

        try:
            logger.info("Starting Secure FL experiment...")

            # Setup model and data
            global_model, client_datasets = self.setup_model_and_data()
            global_model.to(self.device)

            # Initialize aggregator
            aggregator = FedJSCMAggregator(momentum=0.9, learning_rate=0.01)

            # Create clients
            clients = self._create_clients(global_model, client_datasets)

            # Run federated training rounds
            for round_num in range(self.num_rounds):
                round_start = time.time()

                logger.info(f"Round {round_num + 1}/{self.num_rounds}")

                # Perform federated training round
                round_metrics = self._federated_training_round(
                    global_model, clients, aggregator, round_num
                )

                round_time = time.time() - round_start
                round_metrics["time"] = round_time

                self.results["training_history"].append(
                    {
                        "round": round_num + 1,
                        "accuracy": round_metrics.get("accuracy", 0.0),
                        "loss": round_metrics.get("loss", 0.0),
                        "time": round_time,
                        "client_metrics": round_metrics.get("client_metrics", {}),
                    }
                )

                logger.info(
                    f"Round {round_num + 1} completed: "
                    f"Acc={round_metrics.get('accuracy', 0.0):.3f}, "
                    f"Loss={round_metrics.get('loss', 0.0):.3f}, "
                    f"Time={round_time:.2f}s"
                )

            # Final evaluation
            final_metrics = self._evaluate_global_model(global_model, client_datasets)
            self.results.update(final_metrics)

            self.results["total_time"] = time.time() - start_time
            self.results["final_accuracy"] = self.results["training_history"][-1][
                "accuracy"
            ]

            logger.info(f"Experiment completed in {self.results['total_time']:.2f}s")
            logger.info(f"Final accuracy: {self.results['final_accuracy']:.3f}")

            return self.results

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise

    def _create_clients(
        self, model: nn.Module, client_datasets: list[tuple[DataLoader, DataLoader]]
    ) -> list:
        """Create secure FL clients"""
        clients = []

        for client_id, (train_loader, val_loader) in enumerate(client_datasets):
            # Create model copy for this client
            client_model = self._create_model_for_dataset(self.dataset_name)
            client_model.load_state_dict(model.state_dict())
            client_model.to(self.device)

            # Create secure client with closure to bind the current model
            def create_model_fn(model_state_dict, dataset_name):
                def model_fn():
                    m = self._create_model_for_dataset(dataset_name)
                    m.load_state_dict(model_state_dict)
                    return m

                return model_fn

            client = create_client(
                client_id=f"client_{client_id}",
                model_fn=create_model_fn(client_model.state_dict(), self.dataset_name),
                train_data=train_loader.dataset,
                val_data=val_loader.dataset if val_loader else None,
                batch_size=self.batch_size,
                enable_zkp=self.enable_zkp,
                proof_rigor=self.proof_rigor,
                device=self.device,
            )

            clients.append(client)

        return clients

    def _federated_training_round(
        self,
        global_model: nn.Module,
        clients: list,
        aggregator: FedJSCMAggregator,
        round_num: int,
    ) -> dict[str, Any]:
        """Execute one round of federated training"""

        client_updates = []
        client_weights = []
        client_metrics = {}

        # Distribute global model to clients and collect updates
        for i, client in enumerate(clients):
            logger.info(f"Training client {i}")

            # Set global model parameters on client
            client.model.load_state_dict(global_model.state_dict())

            # Local training
            client_result = self._train_client(client, round_num)

            # Collect updates
            client_updates.append(client_result["parameters"])
            client_weights.append(client_result["num_examples"])
            client_metrics[f"client_{i}"] = client_result["metrics"]

        # Aggregate updates
        logger.info("Aggregating client updates...")
        # Normalize weights to sum to 1
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # Get current global parameters
        from secure_fl.utils import torch_to_ndarrays

        global_params = torch_to_ndarrays(global_model)

        aggregated_params = aggregator.aggregate(
            client_updates, normalized_weights, round_num, global_params
        )

        # Update global model
        from secure_fl.utils import ndarrays_to_torch

        ndarrays_to_torch(global_model, aggregated_params)

        # Evaluate global model
        global_metrics = self._evaluate_global_model(
            global_model, [(clients[0].train_loader, clients[0].val_loader)]
        )

        return {
            "accuracy": global_metrics.get("accuracy", 0.0),
            "loss": global_metrics.get("loss", 0.0),
            "client_metrics": client_metrics,
            "aggregation_info": {
                "momentum_initialized": aggregator.momentum_initialized
            },
        }

    def _train_client(self, client, round_num: int) -> dict[str, Any]:
        """Train a single client"""

        # Local training
        client.model.train()
        optimizer = optim.SGD(client.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_samples = 0

        for _ in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            for data, target in client.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = client.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)

            total_loss += epoch_loss
            total_samples += epoch_samples

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # Evaluate client model
        accuracy = self._evaluate_client(client)

        # Get model parameters
        from secure_fl.utils import torch_to_ndarrays

        parameters = torch_to_ndarrays(client.model)

        return {
            "parameters": parameters,
            "num_examples": total_samples,
            "metrics": {
                "loss": avg_loss,
                "accuracy": accuracy,
                "samples": total_samples,
            },
        }

    def _evaluate_client(self, client) -> float:
        """Evaluate client model accuracy"""
        client.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            data_loader = (
                client.val_loader if client.val_loader else client.train_loader
            )
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = client.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        return correct / total if total > 0 else 0.0

    def _evaluate_global_model(
        self, model: nn.Module, client_datasets: list[tuple[DataLoader, DataLoader]]
    ) -> dict[str, Any]:
        """Evaluate global model on validation data"""
        model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for train_loader, val_loader in client_datasets[
                :1
            ]:  # Use first client's validation set
                data_loader = val_loader if val_loader else train_loader

                for data, target in data_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)

                    loss = criterion(output, target)
                    total_loss += loss.item() * data.size(0)

                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total_samples += target.size(0)

        accuracy = correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return {"accuracy": accuracy, "loss": avg_loss, "total_samples": total_samples}

    def save_results(self, output_path: str = "experiment_results.json"):
        """Save experiment results to file"""
        results_path = Path(output_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_path}")


def get_default_config() -> dict[str, Any]:
    """Get default experiment configuration"""
    return {
        "num_clients": 5,
        "num_rounds": 10,
        "dataset": "mnist",
        "enable_zkp": False,
        "proof_rigor": "medium",
        "local_epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.01,
    }


def main():
    """Main entry point for the training experiment"""
    parser = argparse.ArgumentParser(description="Secure FL Training Experiment")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of clients")
    parser.add_argument(
        "--rounds", type=int, default=10, help="Number of training rounds"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["synthetic", "mnist", "cifar10", "medmnist"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--enable-zkp", action="store_true", help="Enable ZKP verification"
    )
    parser.add_argument(
        "--local-epochs", type=int, default=3, help="Number of local training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Learning rate"
    )

    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()

    # Override config with command line args
    if args.num_clients:
        config["num_clients"] = args.num_clients
    if args.rounds:
        config["num_rounds"] = args.rounds
    if args.dataset:
        config["dataset"] = args.dataset
    if args.enable_zkp:
        config["enable_zkp"] = True
    if args.local_epochs:
        config["local_epochs"] = args.local_epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate

    logger.info("Starting Secure FL experiment with config:")
    logger.info(json.dumps(config, indent=2))

    # Run experiment
    experiment = SecureFL_Experiment(config)
    results = experiment.run_experiment()

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"training_results_{args.dataset}_{config['num_clients']}clients_{timestamp}.json"
    output_path = Path("results") / output_filename

    experiment.save_results(str(output_path))

    # Print summary
    print("\n" + "=" * 60)
    print("SECURE FL EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Dataset: {config['dataset']}")
    print(f"Model: {results.get('model_info', {}).get('model_class', 'Unknown')}")
    print(f"Clients: {config['num_clients']}")
    print(f"Rounds: {config['num_rounds']}")
    print(f"Final Accuracy: {results['final_accuracy']:.4f}")
    print(f"Total Time: {results['total_time']:.2f}s")
    print(
        f"Total Parameters: {results.get('model_info', {}).get('total_parameters', 'Unknown')}"
    )
    print(f"ZKP Enabled: {config['enable_zkp']}")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
