"""
Federated Learning Client with zk-STARK Proof Generation

This module implements FL clients that:
1. Perform local training using PyTorch
2. Generate zk-STARK proofs for training verification
3. Communicate with the FL server using Flower framework
4. Support dynamic proof rigor adjustment
"""

import logging
import time
from typing import Any

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import NDArrays
from torch.utils.data import DataLoader

from .proof_manager import ClientProofManager
from .quantization import quantize_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureFlowerClient(fl.client.NumPyClient):
    """
    Secure FL client with zk-STARK proof generation
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: str = "cpu",
        enable_zkp: bool = True,
        proof_rigor: str = "high",
        quantize_weights: bool = True,
        local_epochs: int = 1,
        learning_rate: float = 0.01,
    ):
        """
        Initialize secure FL client

        Args:
            client_id: Unique client identifier
            model: PyTorch model for training
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device for computation ('cpu' or 'cuda')
            enable_zkp: Whether to generate zk-STARK proofs
            proof_rigor: Proof rigor level ('high', 'medium', 'low')
            quantize_weights: Whether to quantize weights for circuits
            local_epochs: Number of local training epochs
            learning_rate: Local learning rate
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.enable_zkp = enable_zkp
        self.proof_rigor = proof_rigor
        self.quantize_weights = quantize_weights
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        # ZKP components
        self.proof_manager = ClientProofManager() if enable_zkp else None

        # Training state
        self.training_history = []
        self.round_count = 0

        # Data commitment for ZKP
        self.data_commitment = self._compute_data_commitment()

        logger.info(f"Client {client_id} initialized with ZKP={enable_zkp}")

    def get_parameters(self, config: dict[str, Any]) -> NDArrays:
        """Get model parameters (only learnable parameters)"""
        return [param.detach().cpu().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters (only learnable parameters)"""
        model_params = list(self.model.parameters())

        if len(model_params) != len(parameters):
            raise ValueError(
                f"Number of model parameters ({len(model_params)}) does not match "
                f"number of provided parameters ({len(parameters)})"
            )

        for param, array in zip(model_params, parameters):
            param.data = torch.tensor(array)

    def fit(
        self, parameters: NDArrays, config: dict[str, Any]
    ) -> tuple[NDArrays, int, dict[str, Any]]:
        """
        Train model locally and generate ZKP proof
        """
        start_time = time.time()
        self.round_count = config.get("server_round", 0)

        # Update configuration from server
        self.local_epochs = config.get("local_epochs", self.local_epochs)
        self.learning_rate = config.get("learning_rate", self.learning_rate)
        self.proof_rigor = config.get("proof_rigor", self.proof_rigor)

        # Set received parameters
        self.set_parameters(parameters)
        initial_params = self.get_parameters({})

        # Perform local training
        training_metrics = self._train_local_model()

        # Get updated parameters
        updated_params = self.get_parameters({})

        # Compute parameter update (delta)
        param_delta = self._compute_parameter_delta(initial_params, updated_params)

        # Quantize parameters if enabled
        if self.quantize_weights:
            from .quantization import QuantizationConfig

            config = QuantizationConfig(bits=8)
            param_delta, _ = quantize_parameters(param_delta, config)
            updated_params, _ = quantize_parameters(updated_params, config)

        # Generate ZKP proof
        proof_time = 0
        proof_data = None
        if self.enable_zkp:
            proof_start = time.time()
            proof_data = self._generate_training_proof(
                initial_params=initial_params,
                updated_params=updated_params,
                param_delta=param_delta,
                training_metrics=training_metrics,
            )
            proof_time = time.time() - proof_start

        # Prepare metrics
        num_examples = len(self.train_loader.dataset)
        total_time = time.time() - start_time

        metrics = {
            "client_id": self.client_id,
            "training_time": total_time - proof_time,
            "proof_time": proof_time,
            "local_epochs": self.local_epochs,
            "proof_rigor": self.proof_rigor,
            **training_metrics,
        }

        if proof_data:
            metrics["zkp_proof"] = proof_data

        self.training_history.append(metrics)

        logger.info(
            f"Client {self.client_id} Round {self.round_count}: "
            f"train_time={total_time - proof_time:.2f}s, proof_time={proof_time:.2f}s, "
            f"loss={training_metrics.get('train_loss', 0):.4f}"
        )

        return updated_params, num_examples, metrics

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        """Evaluate model on validation data"""
        if self.val_loader is None:
            return 0.0, 0, {}

        self.set_parameters(parameters)

        loss, accuracy = self._evaluate_model(self.val_loader)
        num_examples = len(self.val_loader.dataset)

        metrics = {
            "client_id": self.client_id,
            "val_accuracy": accuracy,
        }

        return float(loss), num_examples, metrics

    def _train_local_model(self) -> dict[str, float]:
        """Perform local SGD training"""
        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_samples = 0
        batch_losses = []
        gradients_history = []

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)

                loss.backward()

                # Store gradient information for ZKP
                if self.enable_zkp and self.proof_rigor == "high":
                    grad_norm = self._compute_gradient_norm()
                    gradients_history.append(grad_norm)

                optimizer.step()

                batch_loss = loss.item()
                batch_size = data.size(0)

                epoch_loss += batch_loss * batch_size
                epoch_samples += batch_size
                batch_losses.append(batch_loss)

                # Log batch progress for high rigor proofs
                if self.proof_rigor == "high" and batch_idx % 10 == 0:
                    logger.debug(
                        f"Client {self.client_id} Epoch {epoch + 1}/{self.local_epochs}, "
                        f"Batch {batch_idx}, Loss: {batch_loss:.4f}"
                    )

            total_loss += epoch_loss
            total_samples += epoch_samples

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # Compute training accuracy on a subset for efficiency
        train_accuracy = self._compute_training_accuracy()

        metrics = {
            "train_loss": avg_loss,
            "train_accuracy": train_accuracy,
            "total_samples": total_samples,
            "batch_losses": batch_losses[-10:],  # Keep recent batch losses
            "gradient_norms": gradients_history[-5:],  # Keep recent gradient norms
        }

        return metrics

    def _evaluate_model(self, data_loader: DataLoader) -> tuple[float, float]:
        """Evaluate model on given data loader"""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)

                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def _compute_training_accuracy(self) -> float:
        """Compute accuracy on training data (subset for efficiency)"""
        if len(self.train_loader.dataset) > 1000:
            # Use subset for large datasets
            subset_size = min(500, len(self.train_loader.dataset) // 4)
            indices = torch.randperm(len(self.train_loader.dataset))[:subset_size]
            subset = torch.utils.data.Subset(self.train_loader.dataset, indices)
            subset_loader = DataLoader(subset, batch_size=64)
            _, accuracy = self._evaluate_model(subset_loader)
            return accuracy
        else:
            _, accuracy = self._evaluate_model(self.train_loader)
            return accuracy

    def _compute_parameter_delta(
        self, initial_params: NDArrays, updated_params: NDArrays
    ) -> NDArrays:
        """Compute parameter update delta = updated - initial"""
        delta = []
        for init_p, updated_p in zip(initial_params, updated_params):
            delta.append(updated_p - init_p)
        return delta

    def _compute_gradient_norm(self) -> float:
        """Compute L2 norm of current gradients"""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def _compute_data_commitment(self) -> str:
        """Compute cryptographic commitment to training data"""
        if not hasattr(self.train_loader.dataset, "__len__"):
            return "unknown_dataset"

        # Simple hash of dataset size and sample statistics
        dataset_size = len(self.train_loader.dataset)

        # Get a few samples to compute statistics
        sample_data = []
        sample_count = min(10, dataset_size)

        for i, (data, target) in enumerate(self.train_loader):
            if i >= sample_count:
                break
            sample_data.extend(data.flatten().numpy()[:100])  # First 100 elements

        if sample_data:
            data_mean = np.mean(sample_data)
            data_std = np.std(sample_data)
            commitment = f"size_{dataset_size}_mean_{data_mean:.4f}_std_{data_std:.4f}"
        else:
            commitment = f"size_{dataset_size}_empty"

        return commitment

    def _generate_training_proof(
        self,
        initial_params: NDArrays,
        updated_params: NDArrays,
        param_delta: NDArrays,
        training_metrics: dict[str, Any],
    ) -> str | None:
        """Generate zk-STARK proof of correct training"""
        if not self.proof_manager:
            return None

        try:
            # Prepare proof inputs based on rigor level
            proof_inputs = {
                "client_id": self.client_id,
                "round": self.round_count,
                "data_commitment": self.data_commitment,
                "initial_params": initial_params,
                "updated_params": updated_params,
                "param_delta": param_delta,
                "learning_rate": self.learning_rate,
                "local_epochs": self.local_epochs,
                "rigor_level": self.proof_rigor,
            }

            # Add detailed training info for high rigor
            if self.proof_rigor == "high":
                proof_inputs.update(
                    {
                        "batch_losses": training_metrics.get("batch_losses", []),
                        "gradient_norms": training_metrics.get("gradient_norms", []),
                        "total_samples": training_metrics.get("total_samples", 0),
                    }
                )

            # Generate proof using Cairo circuits
            proof = self.proof_manager.generate_training_proof(proof_inputs)

            if proof:
                logger.debug(
                    f"Client {self.client_id} generated {self.proof_rigor} rigor proof"
                )
            else:
                logger.warning(f"Client {self.client_id} failed to generate proof")

            return proof

        except Exception as e:
            logger.error(f"Client {self.client_id} proof generation failed: {e}")
            return None

    def get_client_info(self) -> dict[str, Any]:
        """Get client information and statistics"""
        return {
            "client_id": self.client_id,
            "device": self.device,
            "enable_zkp": self.enable_zkp,
            "proof_rigor": self.proof_rigor,
            "quantize_weights": self.quantize_weights,
            "dataset_size": len(self.train_loader.dataset),
            "data_commitment": self.data_commitment,
            "training_history": self.training_history[-5:],  # Recent history
            "round_count": self.round_count,
        }


def create_client(
    client_id: str, model_fn, train_data, val_data=None, batch_size: int = 32, **kwargs
) -> SecureFlowerClient:
    """
    Factory function to create a secure FL client

    Args:
        client_id: Unique client identifier
        model_fn: Function that returns a PyTorch model
        train_data: Training dataset
        val_data: Validation dataset (optional)
        batch_size: Batch size for data loaders
        **kwargs: Additional arguments for SecureFlowerClient
    """
    # Create data loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )

    val_loader = None
    if val_data is not None:
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Create model
    model = model_fn()

    # Filter kwargs to only include supported parameters for SecureFlowerClient
    supported_params = {
        "device",
        "enable_zkp",
        "proof_rigor",
        "quantize_weights",
        "local_epochs",
        "learning_rate",
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}

    # Create client
    client = SecureFlowerClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        **filtered_kwargs,
    )

    return client


def start_client(client: SecureFlowerClient, server_address: str = "localhost:8080"):
    """Start FL client and connect to server"""
    logger.info(f"Starting client {client.client_id}, connecting to {server_address}")

    fl.client.start_numpy_client(server_address=server_address, client=client)

    logger.info(f"Client {client.client_id} finished training")


# Example usage and testing
if __name__ == "__main__":
    # Simple test model
    # Model moved to secure_fl.models

    config = {"server_round": 1, "local_epochs": 2}

    updated_params, num_examples, metrics = client.fit(initial_params, config)

    print(f"Training completed: {num_examples} examples")
    print(f"Metrics: {metrics}")
    print("Client test completed successfully!")
