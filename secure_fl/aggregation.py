"""
FedJSCM (Federated Joint Server-Client Momentum) Aggregation Algorithm

This module implements the momentum-based aggregation technique that stabilizes
federated learning under non-IID conditions as described in the project proposal.

The FedJSCM algorithm uses server-side momentum to accelerate convergence and
avoid oscillations common in heterogeneous federated learning setups.
"""

import logging
from typing import Any

import numpy as np
from flwr.common import NDArrays

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FedJSCMAggregator:
    """
    Federated Joint Server-Client Momentum Aggregator

    Implements momentum-based aggregation with the update rules:
    m^{(t+1)} = γ * m^{(t)} + Σ(p_i * Δ_i)
    w^{(t+1)} = w^{(t)} + m^{(t+1)}

    where:
    - m is the server momentum
    - γ is the momentum coefficient
    - p_i are client weights
    - Δ_i are client updates
    """

    def __init__(
        self,
        momentum: float = 0.9,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        adaptive_momentum: bool = False,
        momentum_decay: float = 0.99,
    ):
        """
        Initialize FedJSCM aggregator

        Args:
            momentum: Momentum coefficient (γ) for server-side momentum
            learning_rate: Global learning rate
            weight_decay: L2 regularization coefficient
            adaptive_momentum: Whether to adaptively adjust momentum
            momentum_decay: Decay rate for adaptive momentum
        """
        # Validation
        if not (0 <= momentum <= 1):
            raise ValueError("Momentum must be between 0 and 1")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")

        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adaptive_momentum = adaptive_momentum
        self.momentum_decay = momentum_decay

        # Server state
        self.server_momentum: NDArrays | None = None
        self.global_params: NDArrays | None = None
        self.round_count = 0

        # Statistics for adaptive momentum
        self.gradient_variance_history = []
        self.convergence_history = []

        logger.info(f"FedJSCM initialized with momentum={momentum}, lr={learning_rate}")

    @property
    def momentum_initialized(self) -> bool:
        """Check if momentum has been initialized"""
        return self.server_momentum is not None

    def aggregate(
        self,
        client_updates: list[NDArrays],
        client_weights: list[float],
        server_round: int,
        global_params: NDArrays | None = None,
        stability_score: float | None = None,
    ) -> NDArrays:
        """
        Aggregate client updates using FedJSCM algorithm

        Args:
            client_updates: List of client parameters (w_i)
            client_weights: List of client weights (p_i), should sum to 1
            server_round: Current server round number
            global_params: Current global parameters (w^{(t)})
            stability_score: Optional stability score for adaptive momentum

        Returns:
            Updated global parameters (w^{(t+1)})
        """
        if not client_updates:
            raise ValueError("At least one client update is required")

        if len(client_updates) != len(client_weights):
            raise ValueError("Number of client updates must match number of weights")

        # Check for negative weights
        for _i, weight in enumerate(client_weights):
            if weight < 0:
                raise ValueError("All client weights must be non-negative")

        # Check for NaN/Inf values in client updates
        for i, update in enumerate(client_updates):
            for j, param in enumerate(update):
                if np.isnan(param).any() or np.isinf(param).any():
                    raise ValueError(
                        f"Client update {i} contains NaN or Inf values in parameter {j}"
                    )

        self.round_count = server_round

        # Normalize weights to ensure they sum to 1
        total_weight = sum(client_weights)
        if total_weight == 0:
            raise ValueError("Total client weights must be positive")
        client_weights = [w / total_weight for w in client_weights]

        # Check if momentum shapes match current client updates and reinitialize if needed
        if self.server_momentum is not None:
            if len(self.server_momentum) != len(client_updates[0]):
                logger.info(
                    f"Model architecture changed: {len(self.server_momentum)} -> {len(client_updates[0])} layers. "
                    "Reinitializing momentum."
                )
                self.server_momentum = None
            else:
                # Check individual layer shapes
                for i, (momentum_layer, update_layer) in enumerate(
                    zip(self.server_momentum, client_updates[0], strict=False)
                ):
                    if momentum_layer.shape != update_layer.shape:
                        logger.info(
                            f"Model architecture changed at layer {i}: {momentum_layer.shape} -> {update_layer.shape}. "
                            "Reinitializing momentum."
                        )
                        self.server_momentum = None
                        break

        # Initialize global parameters if not provided
        if global_params is None:
            if self.global_params is None:
                raise ValueError("Global parameters must be provided for first round")
            global_params = self.global_params
        else:
            self.global_params = global_params

        # Step 1: Compute weighted average of client parameters
        aggregated_params = self._weighted_average(client_updates, client_weights)

        # Special case: if momentum is 0, return standard FedAvg
        current_momentum = self._get_current_momentum_coefficient()
        if current_momentum == 0.0:
            # Initialize momentum for reporting purposes, even though it won't be used
            if self.server_momentum is None:
                self.server_momentum = [
                    np.zeros_like(param) for param in aggregated_params
                ]
            self.global_params = aggregated_params
            return aggregated_params

        # Special case: single client should return client parameters directly
        if len(client_updates) == 1 and client_weights[0] == 1.0:
            # Initialize momentum for reporting purposes
            if self.server_momentum is None:
                self.server_momentum = [param.copy() for param in aggregated_params]
            self.global_params = aggregated_params
            return aggregated_params

        # Step 2: Apply weight decay if specified
        if self.weight_decay > 0:
            aggregated_params = self._apply_weight_decay(
                aggregated_params, global_params, self.weight_decay
            )

        # Step 3: Update server momentum
        # Initialize momentum if first round
        if self.server_momentum is None:
            # First round: momentum = weighted average
            self.server_momentum = [param.copy() for param in aggregated_params]
        else:
            # Subsequent rounds: m^{(t+1)} = γ * m^{(t)} + (1-γ) * weighted_avg
            new_momentum = []
            for old_m, new_avg in zip(
                self.server_momentum, aggregated_params, strict=False
            ):
                updated_m = current_momentum * old_m + (1 - current_momentum) * new_avg
                new_momentum.append(updated_m)
            self.server_momentum = new_momentum

        # Step 4: Update global parameters
        # w^{(t+1)} = w^{(t)} + η * m^{(t+1)}
        if global_params is None:
            raise ValueError(
                "Global parameters required for momentum-based aggregation"
            )

        new_global_params = self._apply_momentum_update(
            global_params, self.server_momentum, self.learning_rate
        )

        # Step 5: Update statistics for adaptive momentum
        if self.adaptive_momentum:
            self._update_adaptive_statistics(client_updates, aggregated_params)

        self.global_params = new_global_params

        # Log aggregation statistics
        momentum_norm = self._compute_norm(self.server_momentum)
        update_norm = self._compute_norm(aggregated_params)

        logger.debug(
            f"Round {server_round}: momentum_norm={momentum_norm:.6f}, "
            f"update_norm={update_norm:.6f}"
        )

        return new_global_params

    def _initialize_momentum(self, reference_params: NDArrays) -> NDArrays:
        """Initialize momentum with zeros matching parameter structure"""
        return [np.zeros_like(param) for param in reference_params]

    def _weighted_average(
        self, client_updates: list[NDArrays], client_weights: list[float]
    ) -> NDArrays:
        """Compute weighted average of client parameters"""
        if not client_updates:
            raise ValueError("Empty client updates")

        # Validate parameter shapes are consistent across clients
        reference_shapes = [param.shape for param in client_updates[0]]
        for _client_idx, client_params in enumerate(client_updates[1:], 1):
            if len(client_params) != len(reference_shapes):
                raise ValueError("Parameter shapes must match")
            for layer_idx, param in enumerate(client_params):
                if param.shape != reference_shapes[layer_idx]:
                    raise ValueError("Parameter shapes must match")

        # Initialize result with zeros
        num_layers = len(client_updates[0])
        aggregated = [np.zeros_like(client_updates[0][i]) for i in range(num_layers)]

        # Weighted sum
        for client_params, weight in zip(client_updates, client_weights, strict=False):
            for i, layer_params in enumerate(client_params):
                aggregated[i] += weight * layer_params

        return aggregated

    def _apply_weight_decay(
        self, update: NDArrays, params: NDArrays, decay: float
    ) -> NDArrays:
        """Apply L2 weight decay regularization"""
        regularized_update = []
        for update_layer, param_layer in zip(update, params, strict=False):
            # Add weight decay term: update - λ * params
            regularized_layer = update_layer - decay * param_layer
            regularized_update.append(regularized_layer)
        return regularized_update

    def _update_momentum(
        self, momentum: NDArrays, update: NDArrays, momentum_coeff: float
    ) -> NDArrays:
        """Update server momentum: m^{(t+1)} = γ * m^{(t)} + update"""
        new_momentum = []
        for momentum_layer, update_layer in zip(momentum, update, strict=False):
            new_layer = momentum_coeff * momentum_layer + update_layer
            new_momentum.append(new_layer)
        return new_momentum

    def _apply_momentum_update(
        self, params: NDArrays, momentum: NDArrays, lr: float
    ) -> NDArrays:
        """Apply momentum update: w^{(t+1)} = w^{(t)} + η * m^{(t+1)}"""
        updated_params = []
        for param_layer, momentum_layer in zip(params, momentum, strict=False):
            new_layer = param_layer + lr * momentum_layer
            updated_params.append(new_layer)
        return updated_params

    def _get_current_momentum_coefficient(self) -> float:
        """Get current momentum coefficient (adaptive if enabled)"""
        if not self.adaptive_momentum:
            return self.momentum

        # Adaptive momentum based on gradient variance
        if len(self.gradient_variance_history) < 3:
            return self.momentum

        # Reduce momentum when gradients are highly variable
        recent_variance = np.mean(self.gradient_variance_history[-3:])
        variance_threshold = 0.1  # Tunable hyperparameter

        if recent_variance > variance_threshold:
            adaptive_coeff = self.momentum * self.momentum_decay
        else:
            adaptive_coeff = min(
                self.momentum * (1 + 0.01),
                0.99,  # Max momentum cap
            )

        return adaptive_coeff

    def _update_adaptive_statistics(
        self, client_updates: list[NDArrays], aggregated_update: NDArrays
    ):
        """Update statistics for adaptive momentum"""
        # Compute gradient variance across clients
        variances = []
        for layer_idx in range(len(aggregated_update)):
            layer_updates = [
                client_update[layer_idx] for client_update in client_updates
            ]
            layer_variance = np.var([np.mean(update) for update in layer_updates])
            variances.append(layer_variance)

        avg_variance = np.mean(variances)
        self.gradient_variance_history.append(avg_variance)

        # Keep only recent history
        if len(self.gradient_variance_history) > 10:
            self.gradient_variance_history.pop(0)

    def _compute_norm(self, params: NDArrays) -> float:
        """Compute L2 norm of parameters"""
        total_norm = 0.0
        for layer_params in params:
            total_norm += np.sum(layer_params**2)
        return np.sqrt(total_norm)

    def get_momentum_state(self) -> dict[str, Any]:
        """Get current momentum state for debugging/monitoring"""
        if self.server_momentum is None:
            return {
                "momentum": self.momentum,
                "learning_rate": self.learning_rate,
                "initialized": False,
            }

        momentum_norms = [self._compute_norm([layer]) for layer in self.server_momentum]

        return {
            "momentum": self.server_momentum,
            "server_momentum": self.server_momentum,  # For backward compatibility
            "momentum_norms": momentum_norms,
            "momentum_coefficient": self._get_current_momentum_coefficient(),
            "round_count": self.round_count,
            "initialized": True,
            "gradient_variance_history": self.gradient_variance_history[
                -5:
            ],  # Recent history
        }

    def reset_momentum(self):
        """Reset momentum state (useful for experiments)"""
        self.server_momentum = None
        self.global_params = None
        self.round_count = 0
        self.gradient_variance_history = []
        self.convergence_history = []
        logger.info("Momentum state reset")

    def save_state(self, filepath: str):
        """Save aggregator state to disk"""
        import pickle

        state = {
            "momentum": self.server_momentum,
            "global_params": self.global_params,
            "round_count": self.round_count,
            "momentum_coeff": self.momentum,
            "learning_rate": self.learning_rate,
            "gradient_variance_history": self.gradient_variance_history,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Aggregator state saved to {filepath}")

    def load_state(self, filepath: str):
        """Load aggregator state from disk"""
        import pickle

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.server_momentum = state.get("momentum")
        self.global_params = state.get("global_params")
        self.round_count = state.get("round_count", 0)
        self.momentum = state.get("momentum_coeff", self.momentum)
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.gradient_variance_history = state.get("gradient_variance_history", [])

        logger.info(f"Aggregator state loaded from {filepath}")


def test_fedjscm_aggregator():
    """Test function for FedJSCM aggregator"""
    print("Testing FedJSCM Aggregator...")

    # Create test data
    aggregator = FedJSCMAggregator(momentum=0.9, learning_rate=0.01)

    # Simulate client updates (2 clients, 2 layers each)
    client1_update = [np.random.randn(10, 5), np.random.randn(5)]
    client2_update = [np.random.randn(10, 5), np.random.randn(5)]
    client_updates = [client1_update, client2_update]

    # Client weights (equal weight)
    client_weights = [0.5, 0.5]

    # Initial global parameters
    global_params = [np.random.randn(10, 5), np.random.randn(5)]

    # Run aggregation
    updated_params = aggregator.aggregate(
        client_updates=client_updates,
        client_weights=client_weights,
        server_round=1,
        global_params=global_params,
    )

    print(f"Original param shapes: {[p.shape for p in global_params]}")
    print(f"Updated param shapes: {[p.shape for p in updated_params]}")
    print(f"Momentum state: {aggregator.get_momentum_state()['initialized']}")

    # Test multiple rounds
    for round_num in range(2, 5):
        client_updates = [
            [np.random.randn(10, 5), np.random.randn(5)],
            [np.random.randn(10, 5), np.random.randn(5)],
        ]
        updated_params = aggregator.aggregate(
            client_updates=client_updates,
            client_weights=client_weights,
            server_round=round_num,
            global_params=updated_params,
        )

    momentum_state = aggregator.get_momentum_state()
    print(f"Final momentum norms: {momentum_state['momentum_norms']}")
    print("FedJSCM test completed successfully!")


if __name__ == "__main__":
    test_fedjscm_aggregator()
