"""
Stability Monitor for Dynamic Proof Rigor Adjustment

This module implements a training stability monitor that tracks various metrics
to dynamically adjust the rigor of zero-knowledge proofs during federated learning.

The monitor tracks:
1. Model parameter stability (gradient norms, parameter changes)
2. Training convergence indicators (loss variance, accuracy trends)
3. Communication overhead metrics
4. Historical stability patterns

Based on these metrics, it provides recommendations for proof rigor levels:
- High rigor: During unstable training phases
- Medium rigor: During moderately stable phases
- Low rigor: During highly stable/converged phases
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass
from flwr.common import NDArrays

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass

class StabilityMetrics:
    """Container for stability metrics"""

    parameter_stability: float = 0.0
    gradient_stability: float = 0.0
    loss_stability: float = 0.0
    convergence_score: float = 0.0
    communication_efficiency: float = 0.0
    overall_stability: float = 0.0


class StabilityMonitor:
    """
    Monitor for tracking training stability and adjusting proof rigor

    The stability monitor uses multiple indicators to assess the current
    state of federated training and recommend appropriate proof rigor levels.
    """

    def __init__(
        self,
        window_size: int = 10,
        stability_threshold_high: float = 0.9,
        stability_threshold_medium: float = 0.7,
        convergence_patience: int = 5,
        min_rounds_for_adjustment: int = 3,
    ):
        """
        Initialize stability monitor

        Args:
            window_size: Size of sliding window for metrics
            stability_threshold_high: Threshold for high stability (low rigor)
            stability_threshold_medium: Threshold for medium stability
            convergence_patience: Rounds to wait before declaring convergence
            min_rounds_for_adjustment: Minimum rounds before rigor adjustment
        """
        self.window_size = window_size
        self.stability_threshold_high = stability_threshold_high
        self.stability_threshold_medium = stability_threshold_medium
        self.convergence_patience = convergence_patience
        self.min_rounds_for_adjustment = min_rounds_for_adjustment

        # Sliding window storage
        self.parameter_norms = deque(maxlen=window_size)
        self.parameter_changes = deque(maxlen=window_size)
        self.gradient_norms = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        self.accuracy_history = deque(maxlen=window_size)
        self.proof_times = deque(maxlen=window_size)
        self.communication_times = deque(maxlen=window_size)

        # State tracking
        self.round_count = 0
        self.previous_parameters: Optional[NDArrays] = None
        self.convergence_counter = 0
        self.last_stability_score = 0.0
        self.rigor_history = deque(maxlen=20)

        # Adaptive thresholds
        self.adaptive_thresholds = {
            "param_change_threshold": 1e-3,
            "gradient_threshold": 1e-2,
            "loss_variance_threshold": 1e-4,
        }

        logger.info(f"StabilityMonitor initialized with window_size={window_size}")

    def update(
        self,
        parameters: NDArrays,
        round_num: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> StabilityMetrics:
        """
        Update stability monitor with new round data

        Args:
            parameters: Current model parameters
            round_num: Current training round number
            metrics: Optional training metrics (loss, accuracy, etc.)

        Returns:
            Current stability metrics
        """
        self.round_count = round_num

        # Compute parameter statistics
        param_norm = self._compute_parameter_norm(parameters)
        self.parameter_norms.append(param_norm)

        # Compute parameter change if we have previous parameters
        param_change = 0.0
        if self.previous_parameters is not None:
            param_change = self._compute_parameter_change(
                self.previous_parameters, parameters
            )
        self.parameter_changes.append(param_change)

        # Update gradient information if available
        if metrics and "gradient_norm" in metrics:
            self.gradient_norms.append(metrics["gradient_norm"])
        elif metrics and "momentum_norm" in metrics:
            # Use momentum norm as proxy for gradient norm
            self.gradient_norms.append(metrics["momentum_norm"])

        # Update loss and accuracy history
        if metrics:
            if "train_loss" in metrics:
                self.loss_history.append(metrics["train_loss"])
            if "train_accuracy" in metrics:
                self.accuracy_history.append(metrics["train_accuracy"])
            if "proof_time" in metrics:
                self.proof_times.append(metrics["proof_time"])
            if "communication_time" in metrics:
                self.communication_times.append(metrics["communication_time"])

        # Store current parameters for next round
        self.previous_parameters = [p.copy() for p in parameters]

        # Compute stability metrics
        stability_metrics = self._compute_stability_metrics()
        self.last_stability_score = stability_metrics.overall_stability

        # Update adaptive thresholds
        self._update_adaptive_thresholds()

        logger.debug(
            f"Round {round_num}: Stability={stability_metrics.overall_stability:.3f}, "
            f"Param_change={param_change:.6f}, Param_norm={param_norm:.6f}"
        )

        return stability_metrics

    def get_recommended_rigor(self) -> str:
        """
        Get recommended proof rigor level based on current stability

        Returns:
            Recommended rigor level: 'high', 'medium', or 'low'
        """
        if self.round_count < self.min_rounds_for_adjustment:
            return "high"  # Start with high rigor

        stability_score = self.last_stability_score

        # Determine rigor level based on stability
        if stability_score >= self.stability_threshold_high:
            recommended_rigor = "low"
            self.convergence_counter += 1
        elif stability_score >= self.stability_threshold_medium:
            recommended_rigor = "medium"
            self.convergence_counter = 0
        else:
            recommended_rigor = "high"
            self.convergence_counter = 0

        # Check for recent rigor changes (avoid oscillation)
        if len(self.rigor_history) >= 3:
            recent_rigors = list(self.rigor_history)[-3:]
            if len(set(recent_rigors)) > 2:  # Too much oscillation
                # Stick with medium rigor to stabilize
                recommended_rigor = "medium"

        self.rigor_history.append(recommended_rigor)

        return recommended_rigor

    def is_converged(self) -> bool:
        """Check if training has converged based on stability metrics"""
        return (
            self.convergence_counter >= self.convergence_patience
            and self.last_stability_score >= self.stability_threshold_high
        )

    def get_stability_score(self) -> float:
        """Get current overall stability score"""
        return self.last_stability_score

    def _compute_parameter_norm(self, parameters: NDArrays) -> float:
        """Compute L2 norm of all parameters"""
        total_norm_squared = 0.0
        for param in parameters:
            total_norm_squared += np.sum(param**2)
        return np.sqrt(total_norm_squared)

    def _compute_parameter_change(
        self, prev_params: NDArrays, curr_params: NDArrays
    ) -> float:
        """Compute relative change in parameters"""
        if len(prev_params) != len(curr_params):
            return float("inf")

        total_change_squared = 0.0
        total_norm_squared = 0.0

        for prev_p, curr_p in zip(prev_params, curr_params):
            if prev_p.shape != curr_p.shape:
                return float("inf")

            diff = curr_p - prev_p
            total_change_squared += np.sum(diff**2)
            total_norm_squared += np.sum(prev_p**2)

        if total_norm_squared == 0:
            return 0.0

        return np.sqrt(total_change_squared / total_norm_squared)

    def _compute_stability_metrics(self) -> StabilityMetrics:
        """Compute comprehensive stability metrics"""
        metrics = StabilityMetrics()

        # Parameter stability (low variance in parameter changes)
        if len(self.parameter_changes) >= 2:
            param_changes = np.array(list(self.parameter_changes))
            param_changes = param_changes[param_changes != 0]  # Remove zeros

            if len(param_changes) > 0:
                param_mean = np.mean(param_changes)
                param_variance = np.var(param_changes)

                # Stability is high when variance is low relative to mean
                if param_mean > 0:
                    param_cv = (
                        np.sqrt(param_variance) / param_mean
                    )  # Coefficient of variation
                    metrics.parameter_stability = np.exp(-param_cv)  # Exponential decay
                else:
                    metrics.parameter_stability = 1.0  # No changes = stable
            else:
                metrics.parameter_stability = 1.0

        # Gradient stability (consistent gradient magnitudes)
        if len(self.gradient_norms) >= 2:
            grad_norms = np.array(list(self.gradient_norms))
            grad_variance = np.var(grad_norms)
            grad_mean = np.mean(grad_norms)

            if grad_mean > 0:
                grad_stability = 1.0 / (1.0 + grad_variance / (grad_mean**2))
                metrics.gradient_stability = grad_stability
            else:
                metrics.gradient_stability = 1.0

        # Loss stability (decreasing loss with low variance)
        if len(self.loss_history) >= 3:
            losses = np.array(list(self.loss_history))

            # Check if loss is generally decreasing
            loss_trend = self._compute_trend(losses)
            loss_variance = np.var(losses[-min(5, len(losses)) :])  # Recent variance

            # Stability is high when loss is decreasing and has low variance
            trend_stability = max(0.0, -loss_trend)  # Negative trend is good
            variance_stability = np.exp(-loss_variance * 1000)  # Scale factor

            metrics.loss_stability = 0.7 * trend_stability + 0.3 * variance_stability

        # Convergence score (based on recent parameter changes)
        if len(self.parameter_changes) >= 3:
            recent_changes = list(self.parameter_changes)[-3:]
            avg_recent_change = np.mean(recent_changes)

            # High convergence when recent changes are small
            convergence_score = np.exp(-avg_recent_change * 1000)  # Scale factor
            metrics.convergence_score = min(1.0, convergence_score)

        # Communication efficiency (based on proof and communication times)
        if len(self.proof_times) >= 2 and len(self.communication_times) >= 2:
            avg_proof_time = np.mean(list(self.proof_times))
            avg_comm_time = np.mean(list(self.communication_times))

            # Efficiency is high when times are low and stable
            if avg_proof_time > 0 and avg_comm_time > 0:
                total_time = avg_proof_time + avg_comm_time
                efficiency = 1.0 / (
                    1.0 + total_time / 10.0
                )  # Normalize by expected time
                metrics.communication_efficiency = efficiency
            else:
                metrics.communication_efficiency = 1.0
        else:
            metrics.communication_efficiency = 0.5  # Neutral

        # Overall stability (weighted combination)
        weights = {
            "parameter": 0.3,
            "gradient": 0.2,
            "loss": 0.3,
            "convergence": 0.15,
            "communication": 0.05,
        }

        overall_stability = (
            weights["parameter"] * metrics.parameter_stability
            + weights["gradient"] * metrics.gradient_stability
            + weights["loss"] * metrics.loss_stability
            + weights["convergence"] * metrics.convergence_score
            + weights["communication"] * metrics.communication_efficiency
        )

        metrics.overall_stability = max(0.0, min(1.0, overall_stability))

        return metrics

    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute linear trend of values (slope)"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        # Simple linear regression
        slope = np.polyfit(x, values, 1)[0]
        return slope

    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on observed statistics"""
        if len(self.parameter_changes) >= 5:
            param_changes = np.array(list(self.parameter_changes))
            param_changes = param_changes[param_changes > 0]  # Remove zeros

            if len(param_changes) > 0:
                # Set threshold as median of observed changes
                self.adaptive_thresholds["param_change_threshold"] = np.median(
                    param_changes
                )

        if len(self.gradient_norms) >= 5:
            grad_norms = np.array(list(self.gradient_norms))
            if len(grad_norms) > 0:
                self.adaptive_thresholds["gradient_threshold"] = np.median(grad_norms)

        if len(self.loss_history) >= 5:
            recent_losses = list(self.loss_history)[-5:]
            loss_variance = np.var(recent_losses)
            self.adaptive_thresholds["loss_variance_threshold"] = loss_variance

    def get_monitor_state(self) -> Dict[str, Any]:
        """Get current state of the monitor for debugging/logging"""
        return {
            "round_count": self.round_count,
            "stability_score": self.last_stability_score,
            "convergence_counter": self.convergence_counter,
            "parameter_norms": list(self.parameter_norms),
            "parameter_changes": list(self.parameter_changes),
            "gradient_norms": list(self.gradient_norms),
            "loss_history": list(self.loss_history),
            "accuracy_history": list(self.accuracy_history),
            "rigor_history": list(self.rigor_history),
            "adaptive_thresholds": self.adaptive_thresholds,
            "is_converged": self.is_converged(),
        }

    def reset(self):
        """Reset monitor state for new training session"""
        self.parameter_norms.clear()
        self.parameter_changes.clear()
        self.gradient_norms.clear()
        self.loss_history.clear()
        self.accuracy_history.clear()
        self.proof_times.clear()
        self.communication_times.clear()

        self.round_count = 0
        self.previous_parameters = None
        self.convergence_counter = 0
        self.last_stability_score = 0.0
        self.rigor_history.clear()

        logger.info("StabilityMonitor reset")


def test_stability_monitor():
    """Test stability monitor functionality"""
    print("Testing StabilityMonitor...")

    monitor = StabilityMonitor(window_size=5)

    # Simulate training rounds with decreasing parameter changes (converging)
    for round_num in range(1, 12):
        # Create mock parameters with decreasing changes
        params = [
            np.random.randn(10, 5) * (1.0 / round_num),  # Decreasing variance
            np.random.randn(5) * (1.0 / round_num),
        ]

        # Mock metrics
        metrics = {
            "train_loss": 1.0 / round_num,  # Decreasing loss
            "train_accuracy": 1.0 - 1.0 / round_num,  # Increasing accuracy
            "gradient_norm": 1.0 / round_num,  # Decreasing gradients
            "proof_time": 0.5 + np.random.normal(0, 0.1),  # Stable proof time
        }

        stability_metrics = monitor.update(params, round_num, metrics)
        recommended_rigor = monitor.get_recommended_rigor()

        print(
            f"Round {round_num}: Stability={stability_metrics.overall_stability:.3f}, "
            f"Rigor={recommended_rigor}"
        )

    # Check final state
    final_state = monitor.get_monitor_state()
    print(f"Final stability score: {final_state['stability_score']:.3f}")
    print(f"Converged: {final_state['is_converged']}")
    print(f"Rigor history: {final_state['rigor_history']}")

    print("StabilityMonitor test completed!")

if __name__ == "__main__":
    test_stability_monitor()
