"""
Proof Manager for ZKP Generation and Verification

This module implements the proof management system for both client-side zk-STARK
proofs and server-side zk-SNARK proofs in the secure federated learning framework.

Client-side: zk-STARK proofs for verifying correct local training
Server-side: zk-SNARK (Groth16) proofs for verifying correct aggregation

The proof managers handle:
1. Circuit compilation and setup
2. Proof generation with parameter inputs
3. Proof verification """

from .utils import compute_hash, compute_parameter_norm, parameters_to_ndarrays
from flwr.common import Parameters, NDArrays


import hashlib
import json
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProofManagerBase(ABC):
    """Base class for proof managers"""

    def __init__(self):
        self.circuit_cache = {}
        self.proof_cache = {}
        self.setup_complete = False

    @abstractmethod
    def setup(self) -> bool:
        """Setup circuits and proving keys"""
        pass

    @abstractmethod
    def generate_proof(self, inputs: dict[str, Any]) -> str | None:
        """Generate a proof for given inputs"""
        pass

    @abstractmethod
    def verify_proof(self, proof: str, public_inputs: dict[str, Any]) -> bool:
        """Verify a proof"""
        pass

    def _hash_inputs(self, inputs: dict[str, Any]) -> str:
        """Create hash of inputs for caching"""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        return hashlib.sha256(input_str.encode()).hexdigest()

class ClientProofManager(ProofManagerBase):

    def __init__(self, max_update_norm: float | None = None):
        super().__init__()
        # Optional: global bound on ||Δw||_2 per round
        self.max_update_norm = max_update_norm

        logger.info(
            f"ClientProofManager initialized (max_update_norm={self.max_update_norm})"
        )

    def setup(self) -> bool:
        """Nothing to setup for this phase."""
        self.setup_complete = True
        return True

    def generate_training_proof(self, proof_inputs: dict[str, Any]) -> str | None:
        """
        Entry point used by SecureFlowerClient.

        Expected keys in proof_inputs:
          - client_id (str)
          - round (int)
          - data_commitment (str)
          - initial_params (NDArrays)
          - updated_params (NDArrays)
          - param_delta (NDArrays)
          - learning_rate (float)
          - local_epochs (int)
          - rigor_level (str)
          - OPTIONAL: batch_losses, gradient_norms, total_samples
        """
        return self.generate_proof(proof_inputs)

    def generate_proof(self, inputs: dict[str, Any]) -> str | None:
        """Build a JSON proof object we can verify server-side."""
        try:
            initial_params = inputs["initial_params"]
            updated_params = inputs["updated_params"]
            param_delta = inputs["param_delta"]

            # Basic hashes (use same serializer as the rest of the code)
            initial_hash = compute_hash(initial_params)
            updated_hash = compute_hash(updated_params)
            delta_hash = compute_hash(param_delta)

            # Norms
            delta_norm_l2 = compute_parameter_norm(param_delta, norm_type="l2")

            # If user configured a max norm, include it
            max_norm = self.max_update_norm

            proof_obj = {
                "type": "client_training_proof",
                "version": 1,
                "client_id": inputs.get("client_id"),
                "round": int(inputs.get("round", 0)),
                "data_commitment": inputs.get("data_commitment"),
                "initial_hash": initial_hash,
                "updated_hash": updated_hash,
                "delta_hash": delta_hash,
                "delta_norm_l2": delta_norm_l2,
                "max_delta_norm_l2": max_norm,
                "learning_rate": float(inputs.get("learning_rate", 0.0)),
                "local_epochs": int(inputs.get("local_epochs", 0)),
                "rigor_level": inputs.get("rigor_level", "medium"),
                # Optional detailed metrics
                "total_samples": int(inputs.get("total_samples", 0)),
                "batch_losses": inputs.get("batch_losses", []),
                "gradient_norms": inputs.get("gradient_norms", []),
            }

            # Cache by hash if you want (not required now)
            proof_key = self._hash_inputs(proof_obj)
            self.proof_cache[proof_key] = proof_obj

            return json.dumps(proof_obj)

        except Exception as e:
            logger.error(f"Client proof generation failed: {e}")
            return None

    def verify_proof(self, proof: str, public_inputs: dict[str, Any]) -> bool:
        """
        Not used on the client side; verification is done on the server.

        Implemented only to satisfy the base class API.
        """
        logger.warning("ClientProofManager.verify_proof called unexpectedly")
        return False


class ServerProofManager(ProofManagerBase):
    """
    Server-side proof manager using zk-SNARKs (Groth16)

    Generates proofs for:
    1. Correct FedJSCM aggregation
    2. Valid client update integration
    3. Proper momentum updates
    """

    def __init__(self, circom_path: str = "circom", snarkjs_path: str = "snarkjs"):
        super().__init__()
        self.circom_path = circom_path
        self.snarkjs_path = snarkjs_path

        # Circuit directory
        self.circuit_dir = os.path.join(
            os.path.dirname(__file__), "..", "proofs", "server"
        )

        # Proving and verification keys
        self.proving_key = None
        self.verification_key = None

        logger.info("ServerProofManager initialized")

    def setup(self) -> bool:
        """Setup Circom circuits and generate proving keys"""
        try:
            # Check if tools are available
            result = subprocess.run(
                [self.circom_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.error("Circom not found. Please install Circom.")
                return False

            # Compile circuits and setup keys
            success = self._setup_snark_circuit()
            if success:
                self.setup_complete = True
                logger.info("Server proof setup completed")
            else:
                logger.error("Failed to setup server circuits")

            return success

        except Exception as e:
            logger.error(f"Server proof setup failed: {e}")
            return False

    def generate_server_proof(
        self,
        client_updates: list[NDArrays],
        client_weights: list[float],
        aggregated_params: NDArrays,
        momentum: NDArrays,
        momentum_coeff: float,
    ) -> str | None:
        """Generate zk-SNARK proof for server aggregation"""

        inputs = {
            "client_updates": client_updates,
            "client_weights": client_weights,
            "aggregated_params": aggregated_params,
            "momentum": momentum,
            "momentum_coeff": momentum_coeff,
        }

        return self.generate_proof(inputs)

    def generate_proof(self, inputs: dict[str, Any]) -> str | None:
        """Generate zk-SNARK proof using Circom/SnarkJS"""
        if not self.setup_complete:
            logger.warning("Proof setup not complete, attempting setup...")
            if not self.setup():
                return None

        try:
            # Prepare circuit inputs
            circuit_inputs = self._prepare_snark_inputs(inputs)

            # Generate proof
            proof = self._generate_snark_proof(circuit_inputs)

            if proof:
                logger.debug("Generated server aggregation proof")

            return proof

        except Exception as e:
            logger.error(f"Server proof generation failed: {e}")
            return None

    def verify_proof(self, proof: str, public_inputs: dict[str, Any]) -> bool:
        """Verify zk-SNARK proof"""
        if not self.setup_complete:
            return False

        try:
            # Parse proof
            proof_data = json.loads(proof)

            # Prepare public inputs
            formatted_inputs = self._format_public_inputs(public_inputs)

            # Verify proof
            return self._verify_snark_proof(proof_data, formatted_inputs)

        except Exception as e:
            logger.error(f"Server proof verification failed: {e}")
            return False

    def _setup_snark_circuit(self) -> bool:
        """Setup Circom circuit and generate keys"""
        try:
            os.makedirs(self.circuit_dir, exist_ok=True)

            # Create aggregation circuit
            circuit_file = os.path.join(self.circuit_dir, "aggregation.circom")
            if not os.path.exists(circuit_file):
                self._create_aggregation_circuit(circuit_file)

            # Mock setup - real implementation would compile and setup keys
            # This would involve:
            # 1. circom circuit.circom --r1cs --wasm --sym
            # 2. snarkjs groth16 setup circuit.r1cs pot12_final.ptau circuit.zkey
            # 3. snarkjs zkey export verificationkey circuit.zkey verification_key.json

            self.proving_key = "mock_proving_key"
            self.verification_key = "mock_verification_key"

            logger.debug("SNARK circuit setup completed")
            return True

        except Exception as e:
            logger.error(f"SNARK circuit setup failed: {e}")
            return False

    def verify_client_proof(
        self,
        proof: str,
        updated_parameters: Parameters | NDArrays,
        old_global_params: NDArrays,
    ) -> bool:
        try:
            # Parse proof JSON
                proof_obj = json.loads(proof) if isinstance(proof, str) else proof

                if proof_obj.get("type") != "client_training_proof":
                    logger.warning("Unknown proof type")
                    return False

                # Convert Parameters -> NDArrays if needed
                if isinstance(updated_parameters, list):
                    new_params = updated_parameters
                else:
                    new_params = parameters_to_ndarrays(updated_parameters)

                # Recompute hashes
                recomputed_updated_hash = compute_hash(new_params)

                if recomputed_updated_hash != proof_obj.get("updated_hash"):
                    logger.warning("Updated hash mismatch in client proof")
                    return False

                # Compute delta from server's view: Δw = w_new - w_old
                if len(old_global_params) != len(new_params):
                    logger.warning("Parameter length mismatch for client proof")
                    return False

                delta = []
                for old_layer, new_layer in zip(old_global_params, new_params):
                    if old_layer.shape != new_layer.shape:
                        logger.warning("Parameter shape mismatch for client proof")
                        return False
                    delta.append(new_layer - old_layer)

                # Check delta hash and norm
                recomputed_delta_hash = compute_hash(delta)
                recomputed_delta_norm = compute_parameter_norm(delta, norm_type="l2")

                if recomputed_delta_hash != proof_obj.get("delta_hash"):
                    logger.warning("Delta hash mismatch in client proof")
                    return False

                claimed_delta_norm = float(proof_obj.get("delta_norm_l2", -1.0))
                if claimed_delta_norm < 0:
                    logger.warning("Client proof missing delta_norm_l2")
                    return False

                # They must agree within a small numeric tolerance
                if abs(claimed_delta_norm - recomputed_delta_norm) > 1e-4:
                    logger.warning(
                        f"Delta norm mismatch: claimed={claimed_delta_norm}, "
                        f"recomputed={recomputed_delta_norm}"
                    )
                    return False

                # Optional bound check (can be None)
                max_norm = proof_obj.get("max_delta_norm_l2", None)
                if max_norm is not None:
                    max_norm = float(max_norm)
                    if recomputed_delta_norm > max_norm:
                        logger.warning(
                            f"Client delta norm {recomputed_delta_norm:.6f} "
                            f"exceeds bound {max_norm:.6f}"
                        )
                        return False

                return True

        except Exception as e:
            logger.error(f"Client proof verification failed: {e}")
            return False
    
    
    
    def _create_aggregation_circuit(self, circuit_path: str):
        """Create Circom aggregation circuit"""
        circuit_code = """
pragma circom 2.0.0;

// FedJSCM Aggregation Circuit
// Verifies: w_new = w_old + momentum_coeff * momentum + sum(weight_i * delta_i)

template FedJSCMAggregation(n_clients, param_size) {
    // Private inputs
    signal private input client_deltas[n_clients][param_size];
    signal private input old_momentum[param_size];

    // Public inputs
    signal input client_weights[n_clients];
    signal input momentum_coeff;
    signal input old_params[param_size];

    // Outputs
    signal output new_params[param_size];
    signal output new_momentum[param_size];

    // Intermediate signals
    signal weighted_sum[param_size];

    component weighted_aggregation[param_size];
    component momentum_update[param_size];
    component param_update[param_size];

    // Compute weighted sum of client updates
    for (var i = 0; i < param_size; i++) {
        weighted_sum[i] <== 0;
        for (var j = 0; j < n_clients; j++) {
            weighted_sum[i] += client_weights[j] * client_deltas[j][i];
        }
    }

    // Update momentum and parameters
    for (var i = 0; i < param_size; i++) {
        // new_momentum = momentum_coeff * old_momentum + weighted_sum
        new_momentum[i] <== momentum_coeff * old_momentum[i] + weighted_sum[i];

        // new_params = old_params + new_momentum
        new_params[i] <== old_params[i] + new_momentum[i];
    }
}

component main = FedJSCMAggregation(5, 10);  // 5 clients, 10 parameters
"""

        with open(circuit_path, "w") as f:
            f.write(circuit_code)

        logger.debug(f"Created aggregation circuit: {circuit_path}")

    def _prepare_snark_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Prepare inputs for Circom circuit"""
        circuit_inputs = {}

        # Extract components
        client_updates = inputs.get("client_updates", [])
        client_weights = inputs.get("client_weights", [])
        aggregated_params = inputs.get("aggregated_params", [])
        momentum = inputs.get("momentum", [])
        momentum_coeff = inputs.get("momentum_coeff", 0.9)

        # Flatten and quantize parameters
        circuit_inputs["client_deltas"] = []
        for update in client_updates:
            flattened = np.concatenate([arr.flatten() for arr in update])
            quantized = (flattened * 1000).astype(int).tolist()  # Simple quantization
            circuit_inputs["client_deltas"].append(quantized[:10])  # Limit size

        circuit_inputs["client_weights"] = [int(w * 1000) for w in client_weights]

        if momentum:
            momentum_flat = np.concatenate([arr.flatten() for arr in momentum])
            circuit_inputs["old_momentum"] = (
                (momentum_flat * 1000).astype(int).tolist()[:10]
            )
        else:
            circuit_inputs["old_momentum"] = [0] * 10

        circuit_inputs["momentum_coeff"] = int(momentum_coeff * 1000)

        # Mock old parameters for circuit
        circuit_inputs["old_params"] = [0] * 10

        return circuit_inputs

    def _format_public_inputs(self, public_inputs: dict[str, Any]) -> list[str]:
        """Format public inputs for verification"""
        # Extract public inputs and convert to string format expected by snarkjs
        formatted = []

        if "client_weights" in public_inputs:
            formatted.extend([str(w) for w in public_inputs["client_weights"]])

        if "momentum_coeff" in public_inputs:
            formatted.append(str(public_inputs["momentum_coeff"]))

        return formatted

    def _generate_snark_proof(self, inputs: dict[str, Any]) -> str | None:
        """Generate SNARK proof"""
        try:
            # Mock proof generation - real implementation would use snarkjs
            mock_proof = {
                "proof": {
                    "pi_a": ["0x123", "0x456", "0x1"],
                    "pi_b": [["0x789", "0xabc"], ["0xdef", "0x123"], ["0x1", "0x0"]],
                    "pi_c": ["0x456", "0x789", "0x1"],
                },
                "publicSignals": self._format_public_inputs(inputs),
            }

            return json.dumps(mock_proof, indent=2)

        except Exception as e:
            logger.error(f"SNARK proof generation failed: {e}")
            return None

    def _verify_snark_proof(
        self, proof_data: dict[str, Any], public_inputs: list[str]
    ) -> bool:
        """Verify SNARK proof"""
        try:
            # Mock verification - real implementation would use snarkjs verify
            required_fields = ["proof", "publicSignals"]
            for field in required_fields:
                if field not in proof_data:
                    return False

            # Check proof structure
            proof = proof_data["proof"]
            if not all(key in proof for key in ["pi_a", "pi_b", "pi_c"]):
                return False

            logger.debug("Verified SNARK aggregation proof")
            return True

        except Exception as e:
            logger.error(f"SNARK proof verification failed: {e}")
            return False


def test_proof_managers():
    """Test proof manager functionality"""
    print("Testing ProofManagers...")

    # Test client proof manager
    print("Testing ClientProofManager...")
    client_pm = ClientProofManager()

    # Mock setup (would normally require Cairo installation)
    client_pm.setup_complete = True

    # Test proof generation
    mock_inputs = {
        "client_id": "test_client_1",
        "round": 1,
        "data_commitment": "test_data",
        "initial_params": [np.random.randn(5, 3), np.random.randn(3)],
        "updated_params": [np.random.randn(5, 3), np.random.randn(3)],
        "learning_rate": 0.01,
        "rigor_level": "medium",
    }

    # Mock proof generation
    mock_proof = json.dumps(
        {
            "circuit": "sgd_single_step",
            "proof": {
                "commitment": "abc123",
                "trace_length": 1024,
                "merkle_root": "def456",
            },
        }
    )

    # Test verification
    verification_result = client_pm.verify_proof(mock_proof, mock_inputs)
    print(f"Client proof verification: {verification_result}")

    # Test server proof manager
    print("Testing ServerProofManager...")
    server_pm = ServerProofManager()
    server_pm.setup_complete = True

    mock_server_inputs = {
        "client_updates": [[np.random.randn(3, 2)]],
        "client_weights": [1.0],
        "aggregated_params": [np.random.randn(3, 2)],
        "momentum": [np.random.randn(3, 2)],
        "momentum_coeff": 0.9,
    }

    mock_server_proof = json.dumps(
        {
            "proof": {
                "pi_a": ["0x123", "0x456", "0x1"],
                "pi_b": [["0x789", "0xabc"], ["0xdef", "0x123"], ["0x1", "0x0"]],
                "pi_c": ["0x456", "0x789", "0x1"],
            },
            "publicSignals": ["1000", "900"],
        }
    )

    server_verification_result = server_pm.verify_proof(
        mock_server_proof, mock_server_inputs
    )
    print(f"Server proof verification: {server_verification_result}")

    print("ProofManager tests completed!")


if __name__ == "__main__":
    test_proof_managers()
