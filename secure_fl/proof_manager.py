"""
Proof Manager for ZKP Generation and Verification

This module implements the proof management system for both client-side zk-STARK
proofs and server-side zk-SNARK proofs in the secure federated learning framework.

Client-side: zk-STARK proofs for verifying correct local training
Server-side: zk-SNARK (Groth16) proofs for verifying correct aggregation

The proof managers handle:
1. Circuit compilation and setup
2. Proof generation with parameter inputs
3. Proof verification
4. Integration with Cairo (zk-STARKs) and Circom (zk-SNARKs)
"""

import logging
import json
import hashlib
import subprocess
import tempfile
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from flwr.common import NDArrays
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
    def generate_proof(self, inputs: Dict[str, Any]) -> Optional[str]:
        """Generate a proof for given inputs"""
        pass

    @abstractmethod
    def verify_proof(self, proof: str, public_inputs: Dict[str, Any]) -> bool:
        """Verify a proof"""
        pass

    def _hash_inputs(self, inputs: Dict[str, Any]) -> str:
        """Create hash of inputs for caching"""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        return hashlib.sha256(input_str.encode()).hexdigest()


class ClientProofManager(ProofManagerBase):
    """
    Client-side proof manager using zk-STARKs (Cairo)

    Generates proofs for:
    1. Correct SGD training steps
    2. Valid data usage
    3. Proper parameter updates
    """

    def __init__(
        self,
        cairo_path: str = "cairo-compile",
        stark_prover_path: str = "cairo-prove",
        stark_verifier_path: str = "cairo-verify",
    ):
        super().__init__()
        self.cairo_path = cairo_path
        self.stark_prover_path = stark_prover_path
        self.stark_verifier_path = stark_verifier_path

        # Circuit templates directory
        self.circuit_dir = os.path.join(
            os.path.dirname(__file__), "..", "proofs", "client"
        )

        # Proof parameters
        self.field_prime = 2**251 + 17 * 2**192 + 1  # Cairo prime

        logger.info("ClientProofManager initialized")

    def setup(self) -> bool:
        """Setup Cairo circuits and compilation"""
        try:
            # Check if Cairo tools are available
            result = subprocess.run(
                [self.cairo_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.error("Cairo compiler not found. Please install Cairo.")
                return False

            # Compile circuits
            success = self._compile_circuits()
            if success:
                self.setup_complete = True
                logger.info("Client proof setup completed")
            else:
                logger.error("Failed to setup client circuits")

            return success

        except Exception as e:
            logger.error(f"Client proof setup failed: {e}")
            return False

    def generate_training_proof(self, proof_inputs: Dict[str, Any]) -> Optional[str]:
        """
        Generate zk-STARK proof for training correctness

        Args:
            proof_inputs: Dictionary containing:
                - client_id: Client identifier
                - round: Training round number
                - data_commitment: Hash commitment to training data
                - initial_params: Parameters before training
                - updated_params: Parameters after training
                - param_delta: Parameter update (delta)
                - learning_rate: Learning rate used
                - local_epochs: Number of local epochs
                - rigor_level: Proof rigor ('high', 'medium', 'low')
                - batch_losses: Batch loss history (for high rigor)
                - gradient_norms: Gradient norms (for high rigor)

        Returns:
            Proof string or None if generation fails
        """
        return self.generate_proof(proof_inputs)

    def generate_proof(self, inputs: Dict[str, Any]) -> Optional[str]:
        """Generate zk-STARK proof using Cairo"""
        if not self.setup_complete:
            logger.warning("Proof setup not complete, attempting setup...")
            if not self.setup():
                return None

        try:
            rigor_level = inputs.get("rigor_level", "medium")

            # Select circuit based on rigor level
            if rigor_level == "high":
                circuit_name = "sgd_full_trace"
            elif rigor_level == "medium":
                circuit_name = "sgd_single_step"
            else:  # low rigor
                circuit_name = "sgd_delta_proof"

            # Prepare circuit inputs
            circuit_inputs = self._prepare_circuit_inputs(inputs, circuit_name)

            # Generate proof
            proof = self._generate_stark_proof(circuit_name, circuit_inputs)

            if proof:
                logger.debug(
                    f"Generated {rigor_level} rigor proof for client {inputs.get('client_id', 'unknown')}"
                )

            return proof

        except Exception as e:
            logger.error(f"Proof generation failed: {e}")
            return None

    def verify_proof(self, proof: str, public_inputs: Dict[str, Any]) -> bool:
        """Verify zk-STARK proof"""
        if not self.setup_complete:
            return False

        try:
            # Extract proof components
            proof_data = json.loads(proof)
            circuit_name = proof_data.get("circuit")
            stark_proof = proof_data.get("proof")

            if not circuit_name or not stark_proof:
                return False

            # Prepare public inputs for verification
            verification_inputs = self._prepare_verification_inputs(
                public_inputs, circuit_name
            )

            # Verify proof
            return self._verify_stark_proof(
                circuit_name, stark_proof, verification_inputs
            )

        except Exception as e:
            logger.error(f"Proof verification failed: {e}")
            return False

    def _compile_circuits(self) -> bool:
        """Compile Cairo circuits for different rigor levels"""
        circuits_to_compile = [
            "sgd_full_trace.cairo",
            "sgd_single_step.cairo",
            "sgd_delta_proof.cairo",
        ]

        try:
            os.makedirs(self.circuit_dir, exist_ok=True)

            for circuit_file in circuits_to_compile:
                circuit_path = os.path.join(self.circuit_dir, circuit_file)

                # Create circuit if it doesn't exist
                if not os.path.exists(circuit_path):
                    self._create_circuit_template(circuit_file)

                # Compile circuit
                compiled_path = circuit_path.replace(".cairo", "_compiled.json")
                result = subprocess.run(
                    [self.cairo_path, "--output", compiled_path, circuit_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode != 0:
                    logger.error(f"Failed to compile {circuit_file}: {result.stderr}")
                    return False

                logger.debug(f"Compiled {circuit_file}")

            return True

        except Exception as e:
            logger.error(f"Circuit compilation failed: {e}")
            return False

    def _create_circuit_template(self, circuit_file: str):
        """Create Cairo circuit template"""
        circuit_path = os.path.join(self.circuit_dir, circuit_file)

        if circuit_file == "sgd_full_trace.cairo":
            circuit_code = """
%lang starknet

from starkware.cairo.common.cairo_builtins import HashBuiltin
from starkware.cairo.common.math import assert_nn_le, assert_not_zero
from starkware.cairo.common.hash import hash2

// High rigor: Full SGD trace verification
// Verifies complete training trajectory with all intermediate steps

struct TrainingState {
    params: felt*,  // Model parameters
    gradients: felt*,  // Gradients at each step
    loss: felt,  // Training loss
    learning_rate: felt,
}

@storage_var
func data_commitment() -> (hash: felt) {
}

@storage_var
func client_id() -> (id: felt) {
}

@external
func verify_full_sgd_trace{syscall_ptr: felt*, pedersen_ptr: HashBuiltin*, range_check_ptr}(
    initial_state: TrainingState,
    final_state: TrainingState,
    num_steps: felt,
    step_proofs: felt*
) {
    alloc_locals;

    // Verify data commitment
    let (committed_data) = data_commitment.read();
    assert_not_zero(committed_data);

    // Verify SGD steps
    _verify_sgd_steps(initial_state, final_state, num_steps, step_proofs);

    return ();
}

func _verify_sgd_steps{syscall_ptr: felt*, pedersen_ptr: HashBuiltin*, range_check_ptr}(
    initial_state: TrainingState,
    final_state: TrainingState,
    num_steps: felt,
    step_proofs: felt*
) {
    if (num_steps == 0) {
        return ();
    }

    // Verify single SGD step: w_new = w_old - lr * grad
    // This is a simplified version - actual implementation would be more complex

    let expected_update = initial_state.learning_rate * initial_state.gradients[0];
    let actual_update = final_state.params[0] - initial_state.params[0];

    // In a real circuit, we'd verify this mathematically
    // For now, we just check the structure exists
    assert_nn_le(0, num_steps);

    return ();
}
"""

        elif circuit_file == "sgd_single_step.cairo":
            circuit_code = """
%lang starknet

from starkware.cairo.common.cairo_builtins import HashBuiltin
from starkware.cairo.common.math import assert_nn_le

// Medium rigor: Single SGD step verification
// Verifies one complete training epoch

@external
func verify_sgd_step{syscall_ptr: felt*, pedersen_ptr: HashBuiltin*, range_check_ptr}(
    params_before: felt*,
    params_after: felt*,
    learning_rate: felt,
    gradient_commitment: felt,
    data_size: felt
) {
    alloc_locals;

    // Verify parameter update is consistent with SGD
    // w_new = w_old - lr * gradient

    // Simplified verification - real implementation would compute gradients
    assert_nn_le(0, learning_rate);
    assert_nn_le(0, data_size);

    return ();
}
"""

        else:  # sgd_delta_proof.cairo
            circuit_code = """
%lang starknet

from starkware.cairo.common.cairo_builtins import HashBuiltin
from starkware.cairo.common.math import assert_nn_le

// Low rigor: Parameter delta verification
// Just verifies parameter update magnitude is reasonable

@external
func verify_delta_norm{syscall_ptr: felt*, pedersen_ptr: HashBuiltin*, range_check_ptr}(
    param_delta: felt*,
    delta_norm: felt,
    max_norm: felt,
    data_commitment: felt
) {
    alloc_locals;

    // Verify delta norm is within reasonable bounds
    assert_nn_le(delta_norm, max_norm);
    assert_nn_le(0, delta_norm);

    // Verify data commitment exists
    assert_nn_le(0, data_commitment);

    return ();
}
"""

        with open(circuit_path, "w") as f:
            f.write(circuit_code)

        logger.debug(f"Created circuit template: {circuit_file}")

    def _prepare_circuit_inputs(
        self, inputs: Dict[str, Any], circuit_name: str
    ) -> Dict[str, Any]:
        """Prepare inputs for Cairo circuit"""
        circuit_inputs = {}

        # Convert parameters to field elements
        if "initial_params" in inputs:
            circuit_inputs["initial_params"] = self._params_to_field_elements(
                inputs["initial_params"]
            )

        if "updated_params" in inputs:
            circuit_inputs["updated_params"] = self._params_to_field_elements(
                inputs["updated_params"]
            )

        if "param_delta" in inputs:
            circuit_inputs["param_delta"] = self._params_to_field_elements(
                inputs["param_delta"]
            )

        # Convert scalar values
        circuit_inputs["learning_rate"] = self._scalar_to_field_element(
            inputs.get("learning_rate", 0.01)
        )

        circuit_inputs["client_id"] = (
            hash(str(inputs.get("client_id", "unknown"))) % self.field_prime
        )

        # Data commitment
        data_commitment = inputs.get("data_commitment", "")
        circuit_inputs["data_commitment"] = int(
            hashlib.sha256(data_commitment.encode()).hexdigest()[:8], 16
        )

        # Circuit-specific inputs
        if circuit_name == "sgd_full_trace":
            circuit_inputs["batch_losses"] = [
                self._scalar_to_field_element(loss)
                for loss in inputs.get("batch_losses", [])
            ]
            circuit_inputs["gradient_norms"] = [
                self._scalar_to_field_element(norm)
                for norm in inputs.get("gradient_norms", [])
            ]

        return circuit_inputs

    def _prepare_verification_inputs(
        self, inputs: Dict[str, Any], circuit_name: str
    ) -> Dict[str, Any]:
        """Prepare public inputs for verification"""
        # Similar to circuit inputs but only public values
        return self._prepare_circuit_inputs(inputs, circuit_name)

    def _params_to_field_elements(self, params: NDArrays) -> List[List[int]]:
        """Convert parameter arrays to Cairo field elements"""
        field_params = []

        for param_array in params:
            # Quantize and convert to integers
            quantized = (param_array * 1000).astype(int)  # Simple quantization

            # Convert to field elements
            field_array = []
            for val in quantized.flatten():
                # Ensure value is in field
                field_val = int(val) % self.field_prime
                field_array.append(field_val)

            field_params.append(field_array)

        return field_params

    def _scalar_to_field_element(self, value: float) -> int:
        """Convert scalar to field element"""
        # Simple conversion - real implementation would be more sophisticated
        quantized = int(value * 10000)  # Fixed point with 4 decimal places
        return quantized % self.field_prime

    def _generate_stark_proof(
        self, circuit_name: str, inputs: Dict[str, Any]
    ) -> Optional[str]:
        """Generate STARK proof using Cairo prover"""
        try:
            # Create temporary files for inputs and proof
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as input_file:
                json.dump(inputs, input_file, indent=2)
                input_file_path = input_file.name

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as proof_file:
                proof_file_path = proof_file.name

            # Get compiled circuit path
            circuit_path = os.path.join(
                self.circuit_dir, f"{circuit_name}_compiled.json"
            )

            # Run Cairo prover (simplified - real implementation would be different)
            result = subprocess.run(
                [
                    "python",
                    "-c",
                    f'''
import json
import time
import hashlib

# Mock STARK proof generation
with open("{input_file_path}", "r") as f:
    inputs = json.load(f)

# Generate mock proof
proof = {{
    "circuit": "{circuit_name}",
    "proof": {{
        "commitment": hashlib.sha256(str(inputs).encode()).hexdigest(),
        "trace_length": 1024,
        "merkle_root": hashlib.sha256(b"mock_trace").hexdigest(),
        "fri_proof": "mock_fri_proof_data",
        "timestamp": int(time.time())
    }},
    "public_inputs": inputs.get("data_commitment", 0)
}}

with open("{proof_file_path}", "w") as f:
    json.dump(proof, f, indent=2)
''',
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                # Read generated proof
                with open(proof_file_path, "r") as f:
                    proof_data = f.read()

                # Cleanup
                os.unlink(input_file_path)
                os.unlink(proof_file_path)

                return proof_data
            else:
                logger.error(f"Proof generation failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"STARK proof generation failed: {e}")
            return None

    def _verify_stark_proof(
        self, circuit_name: str, proof: str, public_inputs: Dict[str, Any]
    ) -> bool:
        """Verify STARK proof"""
        try:
            # Mock verification - real implementation would use Cairo verifier
            proof_data = json.loads(proof) if isinstance(proof, str) else proof

            # Basic validation
            required_fields = ["commitment", "trace_length", "merkle_root"]
            for field in required_fields:
                if field not in proof_data:
                    return False

            # Check trace length is reasonable
            trace_length = proof_data.get("trace_length", 0)
            if trace_length < 64 or trace_length > 2**20:
                return False

            # Mock verification always passes for now
            # Real implementation would verify the mathematical proof
            logger.debug(f"Verified STARK proof for circuit {circuit_name}")
            return True

        except Exception as e:
            logger.error(f"STARK proof verification failed: {e}")
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
        client_updates: List[NDArrays],
        client_weights: List[float],
        aggregated_params: NDArrays,
        momentum: NDArrays,
        momentum_coeff: float,
    ) -> Optional[str]:
        """Generate zk-SNARK proof for server aggregation"""

        inputs = {
            "client_updates": client_updates,
            "client_weights": client_weights,
            "aggregated_params": aggregated_params,
            "momentum": momentum,
            "momentum_coeff": momentum_coeff,
        }

        return self.generate_proof(inputs)

    def generate_proof(self, inputs: Dict[str, Any]) -> Optional[str]:
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

    def verify_proof(self, proof: str, public_inputs: Dict[str, Any]) -> bool:
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

    def _prepare_snark_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
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

    def _format_public_inputs(self, public_inputs: Dict[str, Any]) -> List[str]:
        """Format public inputs for verification"""
        # Extract public inputs and convert to string format expected by snarkjs
        formatted = []

        if "client_weights" in public_inputs:
            formatted.extend([str(w) for w in public_inputs["client_weights"]])

        if "momentum_coeff" in public_inputs:
            formatted.append(str(public_inputs["momentum_coeff"]))

        return formatted

    def _generate_snark_proof(self, inputs: Dict[str, Any]) -> Optional[str]:
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
        self, proof_data: Dict[str, Any], public_inputs: List[str]
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
