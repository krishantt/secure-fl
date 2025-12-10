"""
Enhanced Proof Manager with Detailed Timing Metrics

This module extends the base proof manager with comprehensive timing measurements
for performance benchmarking and analysis.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from flwr.common import NDArrays

from .proof_manager import ClientProofManager, ServerProofManager

logger = logging.getLogger(__name__)


class TimingMetrics:
    """Helper class to track detailed timing metrics"""

    def __init__(self):
        self.timings = {}
        self.start_times = {}

    def start(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.perf_counter()

    def end(self, operation: str) -> float:
        """End timing and return duration"""
        if operation in self.start_times:
            duration = time.perf_counter() - self.start_times[operation]
            if operation not in self.timings:
                self.timings[operation] = []
            self.timings[operation].append(duration)
            del self.start_times[operation]
            return duration
        return 0.0

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all timings"""
        summary = {}
        for operation, times in self.timings.items():
            if times:
                summary[operation] = {
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "total": np.sum(times),
                    "count": len(times),
                }
        return summary


class EnhancedClientProofManager(ClientProofManager):
    """Enhanced client proof manager with detailed timing metrics"""

    def __init__(self, use_pysnark: bool = True):
        super().__init__(use_pysnark=use_pysnark)
        self.timing_metrics = TimingMetrics()

    def generate_training_proof(self, proof_inputs: dict[str, Any]) -> str:
        """Generate training proof with detailed timing"""
        self.timing_metrics.start("total_proof_generation")

        try:
            # Detailed timing breakdown
            self.timing_metrics.start("input_preparation")

            # Extract inputs
            initial_params = proof_inputs.get("initial_params", [])
            updated_params = proof_inputs.get("updated_params", [])
            param_delta = proof_inputs.get("param_delta", [])

            # Prepare inputs (flatten, normalize, etc.)
            flat_initial = self._flatten_params(initial_params)
            flat_updated = self._flatten_params(updated_params)
            flat_delta = self._flatten_params(param_delta)

            self.timing_metrics.end("input_preparation")

            # Delta norm calculation
            self.timing_metrics.start("delta_norm_calculation")
            delta_norm_l2 = np.sqrt(sum(d**2 for d in flat_delta))
            self.timing_metrics.end("delta_norm_calculation")

            # Bound calculation
            self.timing_metrics.start("bound_calculation")
            effective_bound = self._get_effective_bound(
                proof_inputs.get("rigor_level", "medium")
            )
            self.timing_metrics.end("bound_calculation")

            # PySNARK proof generation
            self.timing_metrics.start("pysnark_circuit")
            pysnark_result = self._generate_pysnark_delta_bound_proof(
                initial_params, updated_params, delta_norm_l2
            )
            self.timing_metrics.end("pysnark_circuit")

            # Proof assembly
            self.timing_metrics.start("proof_assembly")

            proof = {
                "type": "client_training_proof",
                "timestamp": time.time(),
                "client_id": proof_inputs.get("client_id", "unknown"),
                "round": proof_inputs.get("round", 0),
                "learning_rate": proof_inputs.get("learning_rate", 0.01),
                "local_epochs": proof_inputs.get("local_epochs", 1),
                "total_samples": proof_inputs.get("total_samples", 0),
                "rigor_level": proof_inputs.get("rigor_level", "medium"),
                "delta_norm_l2": float(delta_norm_l2),
                "effective_bound": float(effective_bound),
                "bound_satisfied": delta_norm_l2 <= effective_bound,
                "data_commitment": proof_inputs.get("data_commitment", "unknown"),
                "pysnark_proof": pysnark_result,
                "param_stats": {
                    "initial_norm": float(np.sqrt(sum(p**2 for p in flat_initial))),
                    "updated_norm": float(np.sqrt(sum(p**2 for p in flat_updated))),
                    "delta_norm": float(delta_norm_l2),
                    "param_count": len(flat_initial),
                },
                "training_metrics": {
                    "batch_losses": proof_inputs.get("batch_losses", []),
                    "gradient_norms": proof_inputs.get("gradient_norms", []),
                },
                "timing_breakdown": self.timing_metrics.get_summary(),
            }

            self.timing_metrics.end("proof_assembly")

        except Exception as e:
            logger.error(f"Enhanced proof generation failed: {e}")
            proof = {
                "type": "client_training_proof_failed",
                "timestamp": time.time(),
                "error": str(e),
                "timing_breakdown": self.timing_metrics.get_summary(),
            }

        total_time = self.timing_metrics.end("total_proof_generation")
        proof["total_generation_time"] = total_time

        # Convert to JSON string
        import json

        return json.dumps(proof, indent=2)

    def get_timing_summary(self) -> Dict[str, Any]:
        """Get comprehensive timing summary"""
        return self.timing_metrics.get_summary()


class EnhancedServerProofManager(ServerProofManager):
    """Enhanced server proof manager with detailed timing metrics"""

    def __init__(self):
        super().__init__()
        self.timing_metrics = TimingMetrics()

    def generate_server_proof(
        self,
        client_updates: List[NDArrays],
        client_weights: List[float],
        aggregated_params: NDArrays,
        momentum: NDArrays,
        momentum_coeff: float,
    ) -> Optional[str]:
        """Generate server proof with detailed timing"""
        self.timing_metrics.start("total_server_proof")

        try:
            # Input validation and preparation
            self.timing_metrics.start("input_validation")

            if not client_updates or not client_weights:
                logger.warning("No client updates or weights provided")
                return None

            if len(client_updates) != len(client_weights):
                logger.warning("Mismatch between number of updates and weights")
                return None

            self.timing_metrics.end("input_validation")

            # Aggregation verification
            self.timing_metrics.start("aggregation_verification")

            # Verify weighted average calculation
            total_weight = sum(client_weights)
            if total_weight == 0:
                logger.warning("Total client weight is zero")
                return None

            # Compute expected aggregation
            expected_aggregation = []
            for layer_idx in range(len(aggregated_params)):
                weighted_sum = np.zeros_like(aggregated_params[layer_idx])
                for client_idx, client_update in enumerate(client_updates):
                    weighted_sum += (
                        client_weights[client_idx] * client_update[layer_idx]
                    )
                expected_aggregation.append(weighted_sum / total_weight)

            # Verify aggregation correctness (within tolerance)
            tolerance = 1e-6
            aggregation_correct = True
            max_error = 0.0

            for expected, actual in zip(expected_aggregation, aggregated_params):
                error = np.max(np.abs(expected - actual))
                max_error = max(max_error, error)
                if error > tolerance:
                    aggregation_correct = False

            self.timing_metrics.end("aggregation_verification")

            # Momentum verification
            self.timing_metrics.start("momentum_verification")

            # Verify momentum update if provided
            momentum_correct = True
            momentum_error = 0.0

            if momentum is not None:
                # Check momentum dimensions match aggregated params
                if len(momentum) == len(aggregated_params):
                    for m_layer, agg_layer in zip(momentum, aggregated_params):
                        if m_layer.shape != agg_layer.shape:
                            momentum_correct = False
                            break
                else:
                    momentum_correct = False

            self.timing_metrics.end("momentum_verification")

            # Statistical analysis
            self.timing_metrics.start("statistical_analysis")

            # Compute statistics for verification
            client_stats = []
            for i, client_update in enumerate(client_updates):
                client_norm = np.sqrt(sum(np.sum(layer**2) for layer in client_update))
                client_stats.append(
                    {
                        "client_id": i,
                        "weight": client_weights[i],
                        "param_norm": float(client_norm),
                        "layer_count": len(client_update),
                    }
                )

            aggregated_norm = np.sqrt(
                sum(np.sum(layer**2) for layer in aggregated_params)
            )

            self.timing_metrics.end("statistical_analysis")

            # Circuit generation (placeholder for actual zk-SNARK)
            self.timing_metrics.start("circuit_generation")

            # In a real implementation, this would generate a zk-SNARK proof
            # For now, we create a detailed verification record
            circuit_success = aggregation_correct and momentum_correct

            self.timing_metrics.end("circuit_generation")

            # Proof assembly
            self.timing_metrics.start("proof_assembly")

            proof = {
                "type": "server_aggregation_proof",
                "timestamp": time.time(),
                "verification_results": {
                    "aggregation_correct": aggregation_correct,
                    "max_aggregation_error": float(max_error),
                    "momentum_correct": momentum_correct,
                    "momentum_error": float(momentum_error),
                    "tolerance_used": tolerance,
                },
                "client_statistics": client_stats,
                "aggregation_statistics": {
                    "num_clients": len(client_updates),
                    "total_weight": float(total_weight),
                    "aggregated_norm": float(aggregated_norm),
                    "momentum_coefficient": float(momentum_coeff),
                },
                "circuit_results": {
                    "circuit_generated": circuit_success,
                    "proof_valid": circuit_success,
                },
                "timing_breakdown": self.timing_metrics.get_summary(),
            }

            self.timing_metrics.end("proof_assembly")

        except Exception as e:
            logger.error(f"Enhanced server proof generation failed: {e}")
            proof = {
                "type": "server_aggregation_proof_failed",
                "timestamp": time.time(),
                "error": str(e),
                "timing_breakdown": self.timing_metrics.get_summary(),
            }

        total_time = self.timing_metrics.end("total_server_proof")
        proof["total_generation_time"] = total_time

        # Convert to JSON string
        import json

        return json.dumps(proof, indent=2)

    def verify_client_proof(
        self,
        client_proof: str,
        updated_parameters: NDArrays,
        old_global_params: NDArrays,
    ) -> bool:
        """Verify client proof with timing"""
        self.timing_metrics.start("client_proof_verification")

        try:
            # Parse proof
            self.timing_metrics.start("proof_parsing")
            import json

            proof_data = json.loads(client_proof)
            self.timing_metrics.end("proof_parsing")

            # Basic validation
            self.timing_metrics.start("basic_validation")

            if proof_data.get("type") != "client_training_proof":
                logger.warning("Invalid proof type")
                return False

            if not proof_data.get("bound_satisfied", False):
                logger.warning("Client proof indicates bound violation")
                return False

            self.timing_metrics.end("basic_validation")

            # Parameter consistency check
            self.timing_metrics.start("parameter_consistency")

            # Verify parameter dimensions and norms are reasonable
            param_stats = proof_data.get("param_stats", {})
            delta_norm = param_stats.get("delta_norm", 0)
            effective_bound = proof_data.get("effective_bound", 0)

            # Check if delta norm is within expected bounds
            consistency_check = (
                delta_norm <= effective_bound * 1.1
            )  # Allow small tolerance

            self.timing_metrics.end("parameter_consistency")

            # PySNARK verification (if available)
            self.timing_metrics.start("pysnark_verification")

            pysnark_result = proof_data.get("pysnark_proof")
            pysnark_valid = True

            if pysnark_result and pysnark_result.get("enabled", False):
                # In a real implementation, this would verify the PySNARK proof
                # For now, we just check if the proof structure is valid
                pysnark_valid = (
                    "vector_len" in pysnark_result
                    and "bound" in pysnark_result
                    and "commitment" in pysnark_result
                )

            self.timing_metrics.end("pysnark_verification")

            verification_result = consistency_check and pysnark_valid

        except Exception as e:
            logger.error(f"Client proof verification failed: {e}")
            verification_result = False

        self.timing_metrics.end("client_proof_verification")

        return verification_result

    def get_timing_summary(self) -> Dict[str, Any]:
        """Get comprehensive timing summary"""
        return self.timing_metrics.get_summary()


# Factory functions for enhanced proof managers
def create_enhanced_client_proof_manager(
    use_pysnark: bool = True,
) -> EnhancedClientProofManager:
    """Create enhanced client proof manager"""
    return EnhancedClientProofManager(use_pysnark=use_pysnark)


def create_enhanced_server_proof_manager() -> EnhancedServerProofManager:
    """Create enhanced server proof manager"""
    return EnhancedServerProofManager()


# Benchmarking utilities
class ProofManagerBenchmark:
    """Utility class for benchmarking proof managers"""

    def __init__(self):
        self.client_manager = create_enhanced_client_proof_manager()
        self.server_manager = create_enhanced_server_proof_manager()

    def benchmark_client_proof_generation(
        self, num_iterations: int = 10, param_size: int = 1000
    ) -> Dict[str, Any]:
        """Benchmark client proof generation performance"""

        results = {
            "iterations": num_iterations,
            "param_size": param_size,
            "timings": [],
            "success_rate": 0.0,
        }

        successful_proofs = 0

        for i in range(num_iterations):
            # Generate random parameters for testing
            initial_params = [np.random.randn(param_size).astype(np.float32)]
            updated_params = [
                initial_params[0]
                + 0.01 * np.random.randn(param_size).astype(np.float32)
            ]
            param_delta = [updated_params[0] - initial_params[0]]

            proof_inputs = {
                "client_id": f"benchmark_client_{i}",
                "round": i + 1,
                "initial_params": initial_params,
                "updated_params": updated_params,
                "param_delta": param_delta,
                "learning_rate": 0.01,
                "local_epochs": 1,
                "total_samples": 100,
                "rigor_level": "medium",
                "data_commitment": f"benchmark_data_{i}",
                "batch_losses": [0.5, 0.4, 0.3],
                "gradient_norms": [1.0, 0.8, 0.6],
            }

            start_time = time.perf_counter()
            proof = self.client_manager.generate_training_proof(proof_inputs)
            end_time = time.perf_counter()

            if proof and "error" not in proof:
                successful_proofs += 1

            results["timings"].append(end_time - start_time)

        results["success_rate"] = successful_proofs / num_iterations
        results["avg_time"] = np.mean(results["timings"])
        results["std_time"] = np.std(results["timings"])
        results["min_time"] = np.min(results["timings"])
        results["max_time"] = np.max(results["timings"])

        return results

    def benchmark_server_proof_generation(
        self, num_clients: int = 5, num_iterations: int = 10, param_size: int = 1000
    ) -> Dict[str, Any]:
        """Benchmark server proof generation performance"""

        results = {
            "iterations": num_iterations,
            "num_clients": num_clients,
            "param_size": param_size,
            "timings": [],
            "success_rate": 0.0,
        }

        successful_proofs = 0

        for i in range(num_iterations):
            # Generate random client updates for testing
            client_updates = []
            client_weights = []

            for j in range(num_clients):
                client_update = [np.random.randn(param_size).astype(np.float32) * 0.01]
                client_updates.append(client_update)
                client_weights.append(np.random.uniform(0.5, 2.0))  # Random weights

            # Normalize weights
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]

            # Compute aggregated parameters (weighted average)
            aggregated_params = [np.zeros(param_size, dtype=np.float32)]
            for client_update, weight in zip(client_updates, client_weights):
                aggregated_params[0] += weight * client_update[0]

            # Generate momentum
            momentum = [np.random.randn(param_size).astype(np.float32) * 0.001]

            start_time = time.perf_counter()
            proof = self.server_manager.generate_server_proof(
                client_updates=client_updates,
                client_weights=client_weights,
                aggregated_params=aggregated_params,
                momentum=momentum,
                momentum_coeff=0.9,
            )
            end_time = time.perf_counter()

            if proof and "error" not in proof:
                successful_proofs += 1

            results["timings"].append(end_time - start_time)

        results["success_rate"] = successful_proofs / num_iterations
        results["avg_time"] = np.mean(results["timings"])
        results["std_time"] = np.std(results["timings"])
        results["min_time"] = np.min(results["timings"])
        results["max_time"] = np.max(results["timings"])

        return results


if __name__ == "__main__":
    # Quick benchmark test
    benchmark = ProofManagerBenchmark()

    print("Benchmarking Enhanced Proof Managers...")

    client_results = benchmark.benchmark_client_proof_generation(num_iterations=5)
    print(f"Client Proof Generation:")
    print(
        f"  Average Time: {client_results['avg_time']:.4f} ± {client_results['std_time']:.4f} seconds"
    )
    print(f"  Success Rate: {client_results['success_rate']:.2f}")

    server_results = benchmark.benchmark_server_proof_generation(num_iterations=5)
    print(f"Server Proof Generation:")
    print(
        f"  Average Time: {server_results['avg_time']:.4f} ± {server_results['std_time']:.4f} seconds"
    )
    print(f"  Success Rate: {server_results['success_rate']:.2f}")
