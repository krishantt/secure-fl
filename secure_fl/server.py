"""
Federated Learning Server with FedJSCM Aggregation and ZKP Integration

This module implements the main FL server that:
1. Manages client connections using Flower framework
2. Implements FedJSCM (Federated Joint Server-Client Momentum) aggregation
3. Integrates zk-SNARK proof generation for server-side verification
4. Manages dynamic proof rigor adjustment based on training stability
"""

import logging
import time
from typing import override

import flwr as fl
import numpy as np
import torch
from flwr.common import EvaluateRes, FitRes, NDArrays, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from .aggregation import FedJSCMAggregator
from .proof_manager import ServerProofManager
from .stability_monitor import StabilityMonitor
from .utils import ndarrays_to_parameters, parameters_to_ndarrays

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureFlowerStrategy(Strategy):
    """
    Custom Flower strategy implementing dual ZKP verification for federated learning
    """

    def __init__(
        self,
        initial_parameters: Parameters | None = None,
        momentum: float = 0.9,
        learning_rate: float = 0.01,
        min_available_clients: int = 2,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        enable_zkp: bool = True,
        proof_rigor: str = "high",  # "high", "medium", "low"
        blockchain_verification: bool = False,
    ):
        super().__init__()

        self.initial_parameters = initial_parameters
        self.min_available_clients = min_available_clients
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate

        # FedJSCM components
        self.aggregator = FedJSCMAggregator(
            momentum=momentum, learning_rate=learning_rate
        )

        # ZKP components
        self.enable_zkp = enable_zkp
        self.proof_manager = ServerProofManager() if enable_zkp else None
        self.blockchain_verification = blockchain_verification

        # Dynamic proof adjustment
        self.stability_monitor = StabilityMonitor()
        self.proof_rigor = proof_rigor

        # Training state
        self.current_round = 0
        self.current_global_params = None
        self.training_metrics = []

        logger.info(f"Initialized SecureFlowerStrategy with ZKP={enable_zkp}")

    def initialize_parameters(self, client_manager) -> Parameters | None:
        """Initialize global model parameters"""
        if self.initial_parameters:
            self.current_global_params = parameters_to_ndarrays(self.initial_parameters)
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> list[tuple[ClientProxy, dict]]:
        """Configure clients for training round"""
        self.current_round = server_round

        # Sample clients
        sample_size = max(
            self.min_fit_clients, int(len(client_manager.all()) * self.fraction_fit)
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_available_clients
        )

        # Configure client instructions
        config = {
            "server_round": server_round,
            "local_epochs": self._get_local_epochs(),
            "learning_rate": self.aggregator.learning_rate,
            "proof_rigor": self.proof_rigor,
            "enable_zkp": self.enable_zkp,
        }

        return [(client, config) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate client updates with ZKP verification"""

        if not results:
            logger.warning(f"No client results received in round {server_round}")
            return None, {}

        start_time = time.time()

        # Step 1: Verify client ZKPs if enabled
        verified_results = []
        if self.enable_zkp:
            for client, fit_res in results:
                if self._verify_client_proof(client, fit_res):
                    verified_results.append((client, fit_res))
                else:
                    logger.warning(f"Client {client.cid} failed ZKP verification")

            logger.info(
                f"Verified {len(verified_results)}/{len(results)} client proofs"
            )
        else:
            verified_results = results

        if not verified_results:
            logger.error("No verified client updates available")
            return None, {}

        # Step 2: Extract parameters and weights
        client_updates = []
        client_weights = []

        for client, fit_res in verified_results:
            # Convert parameters to numpy arrays
            client_params = parameters_to_ndarrays(fit_res.parameters)
            client_updates.append(client_params)

            # Use number of examples as weight (can be modified)
            num_examples = fit_res.num_examples
            client_weights.append(num_examples)

        # Normalize weights
        total_examples = sum(client_weights)
        client_weights = [w / total_examples for w in client_weights]

        # Step 3: FedJSCM Aggregation
        aggregated_params = self.aggregator.aggregate(
            client_updates=client_updates,
            client_weights=client_weights,
            server_round=server_round,
            global_params=self.current_global_params,
        )

        # Update current global parameters
        self.current_global_params = aggregated_params

        # Step 4: Generate server ZKP if enabled
        server_proof_time = 0
        if self.enable_zkp and self._should_generate_server_proof():
            proof_start = time.time()
            server_proof = self._generate_server_proof(
                client_updates=client_updates,
                client_weights=client_weights,
                aggregated_params=aggregated_params,
            )
            server_proof_time = time.time() - proof_start

            if self.blockchain_verification and server_proof:
                self._submit_to_blockchain(server_proof)

        # Step 5: Update stability monitoring and proof rigor
        self._update_stability_metrics(aggregated_params)
        self._adjust_proof_rigor()

        # Prepare metrics
        aggregation_time = time.time() - start_time
        metrics = {
            "aggregation_time": aggregation_time,
            "server_proof_time": server_proof_time,
            "verified_clients": len(verified_results),
            "total_clients": len(results),
            "momentum_norm": float(
                np.sqrt(sum(np.sum(p**2) for p in self.aggregator.server_momentum))
            ),
            "proof_rigor": self.proof_rigor,
        }

        self.training_metrics.append(metrics)

        logger.info(
            f"Round {server_round}: Aggregated {len(verified_results)} clients, "
            f"time={aggregation_time:.2f}s, proof_time={server_proof_time:.2f}s"
        )

        return ndarrays_to_parameters(aggregated_params), metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> list[tuple[ClientProxy, dict]]:
        """Configure clients for evaluation"""
        if not self.fraction_evaluate:
            return []

        # Sample clients for evaluation
        sample_size = max(
            self.min_evaluate_clients,
            int(len(client_manager.all()) * self.fraction_evaluate),
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_available_clients
        )

        config = {"server_round": server_round}
        return [(client, config) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate evaluation results"""

        if not results:
            return None, {}

        # Aggregate metrics
        total_examples = sum([res.num_examples for _, res in results])

        # Weighted average of losses
        weighted_loss = (
            sum([res.loss * res.num_examples for _, res in results]) / total_examples
        )

        # Aggregate other metrics
        metrics = {}
        metric_keys = set()
        for _, res in results:
            if res.metrics:
                metric_keys.update(res.metrics.keys())

        for key in metric_keys:
            values = [
                (res.metrics.get(key, 0) * res.num_examples, res.num_examples)
                for _, res in results
                if res.metrics and key in res.metrics
            ]
            if values:
                weighted_value = sum([v * w for v, w in values]) / sum(
                    [w for _, w in values]
                )
                metrics[f"avg_{key}"] = weighted_value

        metrics["total_examples"] = total_examples

        return weighted_loss, metrics

    @override
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, Scalar]] | None:
        return super().evaluate(server_round, parameters)

    def _verify_client_proof(self, client: ClientProxy, fit_res: FitRes) -> bool:
        """Verify client's zk-STARK proof"""
        if not self.proof_manager:
            return True

        # Extract proof from fit_res metrics
        client_proof = fit_res.metrics.get("zkp_proof")
        if not client_proof:
            logger.warning(f"No ZKP proof from client {client.cid}")
            return not self.enable_zkp  # Allow if ZKP disabled

        # Verify zk-STARK proof
        return self.proof_manager.verify_client_proof(client_proof, fit_res.parameters)

    def _generate_server_proof(
        self,
        client_updates: list[NDArrays],
        client_weights: list[float],
        aggregated_params: NDArrays,
    ) -> str | None:
        """Generate zk-SNARK proof for server aggregation"""
        if not self.proof_manager:
            return None

        try:
            return self.proof_manager.generate_server_proof(
                client_updates=client_updates,
                client_weights=client_weights,
                aggregated_params=aggregated_params,
                momentum=self.aggregator.momentum,
                momentum_coeff=self.aggregator.momentum,
            )
        except Exception as e:
            logger.error(f"Failed to generate server proof: {e}")
            return None

    def _should_generate_server_proof(self) -> bool:
        """Determine if server proof should be generated based on rigor level"""
        if self.proof_rigor == "high":
            return True
        elif self.proof_rigor == "medium":
            return self.current_round % 2 == 0
        elif self.proof_rigor == "low":
            return self.current_round % 5 == 0
        return False

    def _submit_to_blockchain(self, proof: str) -> bool:
        """Submit proof to blockchain for verification"""
        # TODO: Implement blockchain submission
        logger.info("Submitting proof to blockchain (not implemented)")
        return True

    def _update_stability_metrics(self, params: NDArrays) -> None:
        """Update training stability metrics"""
        self.stability_monitor.update(params, self.current_round)

    def _adjust_proof_rigor(self) -> None:
        """Dynamically adjust proof rigor based on stability"""
        stability_score = self.stability_monitor.get_stability_score()

        if stability_score > 0.9:  # Very stable
            self.proof_rigor = "low"
        elif stability_score > 0.7:  # Moderately stable
            self.proof_rigor = "medium"
        else:  # Unstable
            self.proof_rigor = "high"

    def _get_local_epochs(self) -> int:
        """Get number of local epochs based on current rigor"""
        if self.proof_rigor == "high":
            return 1  # More frequent updates
        elif self.proof_rigor == "medium":
            return 2
        else:
            return 3  # Fewer updates when stable


class SecureFlowerServer:
    """Main server class that orchestrates the entire FL process"""

    def __init__(
        self,
        strategy: SecureFlowerStrategy,
        host: str = "localhost",
        port: int = 8080,
        num_rounds: int = 10,
        config: dict | None = None,
    ):
        self.strategy = strategy
        self.host = host
        self.port = port
        self.num_rounds = num_rounds
        self.config = config or {}

        logger.info(f"SecureFlowerServer initialized on {host}:{port}")

    def start(self):
        """Start the federated learning server"""
        logger.info(f"Starting FL server for {self.num_rounds} rounds")

        # Start Flower server
        fl.server.start_server(
            server_address=f"{self.host}:{self.port}",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=self.strategy,
        )

        logger.info("FL training completed")

    def get_training_history(self) -> list[dict]:
        """Get training history and metrics"""
        return self.strategy.training_metrics


def create_server_strategy(
    model_fn,
    momentum: float = 0.9,
    learning_rate: float = 0.01,
    enable_zkp: bool = True,
    proof_rigor: str = "high",
    **kwargs,
) -> SecureFlowerStrategy:
    """Factory function to create server strategy with initial model"""

    # Initialize model to get initial parameters (only learnable parameters)
    model = model_fn()
    initial_params = [param.detach().cpu().numpy() for param in model.parameters()]
    initial_parameters = ndarrays_to_parameters(initial_params)

    # Filter kwargs to only include valid SecureFlowerStrategy parameters
    valid_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        in {
            "min_available_clients",
            "min_fit_clients",
            "min_evaluate_clients",
            "fraction_fit",
            "fraction_evaluate",
            "blockchain_verification",
        }
    }

    return SecureFlowerStrategy(
        initial_parameters=initial_parameters,
        momentum=momentum,
        learning_rate=learning_rate,
        enable_zkp=enable_zkp,
        proof_rigor=proof_rigor,
        **valid_kwargs,
    )


if __name__ == "__main__":
    # Example usage
    def simple_model():
        return torch.nn.Sequential(
            torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
        )

    strategy = create_server_strategy(
        model_fn=simple_model,
        momentum=0.9,
        enable_zkp=False,  # Disable ZKP for initial testing
    )

    server = SecureFlowerServer(strategy=strategy, num_rounds=5)
    server.start()
