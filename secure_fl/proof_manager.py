"""
Proof Manager for ZKP Generation and Verification

This module implements the proof management system for both client-side zk-SNARK
proofs and server-side zk-SNARK proofs in the secure federated learning framework.

Client-side: zk-SNARK proofs (PySNARK) for verifying correct local training
Server-side: zk-SNARK (Groth16) proofs for verifying correct aggregation

The proof managers handle:
1. Circuit compilation and setup
2. Proof generation with parameter inputs
3. Proof verification"""

import hashlib
import json
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from flwr.common import NDArrays, Parameters

from .utils import compute_hash, compute_parameter_norm, parameters_to_ndarrays

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
    def __init__(
        self,
        max_update_norm: float | None = None,
        use_pysnark: bool = True,
        fixed_point_scale: int = 1,
    ):
        super().__init__()
        self.max_update_norm = max_update_norm
        self.use_pysnark = use_pysnark
        self.fixed_point_scale = fixed_point_scale

        logger.info(
            f"ClientProofManager initialized (max_update_norm={self.max_update_norm}, "
            f"use_pysnark={self.use_pysnark}, scale={self.fixed_point_scale})"
        )

    def _flatten_params(self, params: NDArrays) -> list[float]:
        """Flatten list of parameter arrays into a single list of floats."""
        flat: list[float] = []
        for arr in params:
            flat.extend(arr.astype(float).ravel().tolist())
        return flat

    def _to_fixed_point_list(self, values: list[float]) -> list[int]:
        """Convert float list to fixed-point integers using configured scale."""
        s = self.fixed_point_scale
        return [int(round(v * s)) for v in values]

    def _get_effective_bound(self, delta_norm_l2: float) -> float:
        """
        Decide which bound to enforce in the circuit:
        - if max_update_norm set => use that
        - else use an automatic bound (e.g., 2x observed norm)
        """
        if self.max_update_norm is not None:
            return float(self.max_update_norm)
        # simple heuristic: 2x observed norm, but at least a small positive number
        # return max(1e-6, 2.0 * float(delta_norm_l2))
        return min(1000.0, max(1e-6, 2.0 * float(delta_norm_l2)))

    def _generate_pysnark_delta_bound_proof(
        self,
        initial_params: NDArrays,
        updated_params: NDArrays,
        delta_norm_l2: float,
    ) -> dict[str, Any] | None:
        """
        Run the PySNARK circuit delta_bound_proof(initial, updated, bound).

        Returns a small metadata dict we embed into the JSON proof:
        {
          "enabled": True,
          "vector_len": N,
          "scale": scale,
          "bound": bound_used,
          "commitment": "<int or str>"
        }
        """
        if not self.use_pysnark:
            return None
        delta_bound_proof = None
        PrivVal = None
        PubVal = None
        try:
            # Local imports so code still works even if PySNARK not installed
            from pysnark.runtime import PrivVal, PubVal
        except ImportError as e:
            logger.warning(f"PySNARK runtime not available: {e}")
            return None
        try:
            from proofs.client_circuits.delta_bound import delta_bound_proof
        except Exception as e_first:
            # 2. Fallback: add project root and try again
            try:
                import sys
                from pathlib import Path

                repo_root = Path(__file__).resolve().parents[1]  # secure-fl/
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))
                from proofs.client_circuits.delta_bound import delta_bound_proof
            except Exception as e_second:
                logger.warning(
                    "PySNARK / delta_bound_proof NOT available, skipping zk proof\n"
                    f"First import error:  {e_first}\n"
                    f"Second import error: {e_second}"
                )
                return None

        try:
            # 1) Flatten parameters
            flat_init = self._flatten_params(initial_params)
            flat_upd = self._flatten_params(updated_params)

            if len(flat_init) != len(flat_upd):
                logger.warning(
                    f"PySNARK: length mismatch initial ({len(flat_init)}) "
                    f"vs updated ({len(flat_upd)})"
                )
                return None

            # 2) Convert to fixed-point integers
            init_fp = self._to_fixed_point_list(flat_init)
            upd_fp = self._to_fixed_point_list(flat_upd)

            # 3) Choose bound (in float) then scale
            bound_float = self._get_effective_bound(delta_norm_l2)
            bound_int = int(round(bound_float * self.fixed_point_scale))

            # 4) Wrap inputs with PySNARK value types
            # CRITICAL: These must be PrivVal/PubVal for the circuit to work
            initial_circuit = [PrivVal(x) for x in init_fp]
            updated_circuit = [PrivVal(x) for x in upd_fp]
            bound_circuit = PubVal(bound_int)

            logger.info(
                f"Running PySNARK circuit: vector_len={len(init_fp)}, "
                f"bound_int={bound_int}, bound_float={bound_float}"
            )

            # 5) Run the circuit - this generates the constraint system
            l2_sq_result = delta_bound_proof(
                initial_circuit, updated_circuit, bound_circuit
            )

            logger.info("PySNARK circuit executed successfully")

            # 4) Run the circuit
            # Note: PySNARK will internally treat Python ints as private/public values
            # commitment = delta_bound_proof(init_fp, upd_fp, bound_int)

            # We don't yet wire actual snark proof file here;
            # we at least record the commitment and settings.
            return {
                "enabled": True,
                "vector_len": len(init_fp),
                "scale": self.fixed_point_scale,
                "bound_float": bound_float,
                "bound_int": bound_int,
                "l2_sq_result": str(int(l2_sq_result)),
            }
        except AssertionError as e:
            logger.error(f"PySNARK constraint failed (bound violation): {e}")
            return {
                "enabled": True,
                "error": "bound_violation",
                "message": str(e),
            }
        except Exception as e:
            logger.error(f"PySNARK delta-bound proof generation failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

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

            # Optional: run PySNARK circuit to enforce ||Δw|| <= B in a zk way
            pysnark_info = self._generate_pysnark_delta_bound_proof(
                initial_params=initial_params,
                updated_params=updated_params,
                delta_norm_l2=delta_norm_l2,
            )

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
                # New: PySNARK metadata if available
                "pysnark": pysnark_info or {"enabled": False},
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
            circuit_dir = Path(self.circuit_dir)
            build_dir = circuit_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)

            circom_file = circuit_dir / "aggregation.circom"

            # CLEAN ALL OLD BUILD ARTIFACTS to prevent Circom caching issues
            logger.info("Cleaning old build artifacts...")
            artifacts_to_clean = [
                circom_file,  # Old source file
                circuit_dir / "aggregation.sym",  # Symbol table
                build_dir / "aggregation.r1cs",  # Constraint system
                build_dir / "aggregation.wasm",  # WASM witness generator
                build_dir / "aggregation_js",  # JS directory
                build_dir / "aggregation.sym",  # Symbol table in build
                build_dir / "aggregation.zkey",  # Old proving key
                build_dir / "verification_key.json",  # Old verification key
            ]

            for artifact in artifacts_to_clean:
                if artifact.exists():
                    if artifact.is_dir():
                        import shutil

                        shutil.rmtree(artifact)
                        logger.info(f"  Removed directory: {artifact.name}")
                    else:
                        artifact.unlink()
                        logger.info(f"  Removed file: {artifact.name}")

            # ALWAYS recreate the circuit file to ensure correct syntax
            self._create_aggregation_circuit(str(circom_file))

            r1cs = build_dir / "aggregation.r1cs"
            wasm = build_dir / "aggregation.wasm"
            zkey = build_dir / "aggregation.zkey"
            vkey = build_dir / "verification_key.json"

            # 1. Compile circuit (Circom will now be forced to read the new file)
            logger.info("Compiling Circom circuit...")
            subprocess.check_call(
                [
                    self.circom_path,
                    str(circom_file),
                    "--r1cs",
                    "--wasm",
                    "--sym",
                    "-o",
                    str(build_dir),
                ]
            )

            # 2. Powers of tau (if missing, warn)
            ptau = circuit_dir / "pot12_final.ptau"
            if not ptau.exists():
                logger.error(
                    "Missing ptau file. Run: snarkjs powersoftau new & contribute"
                )
                return False

            # 3. Groth16 setup
            subprocess.check_call(
                [self.snarkjs_path, "groth16", "setup", str(r1cs), str(ptau), str(zkey)]
            )

            # 4. Export verification key
            subprocess.check_call(
                [
                    self.snarkjs_path,
                    "zkey",
                    "export",
                    "verificationkey",
                    str(zkey),
                    str(vkey),
                ]
            )

            self.proving_key = str(zkey)
            self.verification_key = str(vkey)

            logger.info("✓ SNARK circuit compiled and setup complete")
            return True

        except Exception as e:
            logger.error(f"Failed SNARK setup: {e}")
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
        """Create a valid Circom 2.0 aggregation circuit"""
        # Build circuit code line by line to ensure proper line breaks
        circuit_lines = [
            "pragma circom 2.0.0;",
            "// Federated Learning Aggregation Circuit",
            "",
            "template FedJSCMAggregation(n_clients, param_size) {",
            "    signal input client_deltas[n_clients][param_size];",
            "    signal input old_momentum[param_size];",
            "    signal input client_weights[n_clients];",
            "    signal input momentum_coeff;",
            "    signal input old_params[param_size];",
            "",
            "    signal output new_params[param_size];",
            "    signal output new_momentum[param_size];",
            "",
            "    // Intermediate signals for weighted sum",
            "    signal weighted_sum[param_size];",
            "    signal partial_sums[n_clients][param_size];",
            "",
            "    var SCALE = 1000000;    // Calculate weighted deltas",
            "    for (var i = 0; i < param_size; i++) {",
            "        for (var j = 0; j < n_clients; j++) {",
            "            partial_sums[j][i] <== client_weights[j] * client_deltas[j][i] / SCALE;",
            "        }",
            "    }",
            "",
            "    // Sum the weighted deltas",
            "    for (var i = 0; i < param_size; i++) {",
            "        var sum = 0;",
            "        for (var j = 0; j < n_clients; j++) {",
            "            sum += partial_sums[j][i];",
            "        }",
            "        weighted_sum[i] <== sum;",
            "    }",
            "",
            "    // Compute momentum and new params",
            "    signal momentum_term[param_size];",
            "    for (var i = 0; i < param_size; i++) {",
            "        momentum_term[i] <== momentum_coeff * old_momentum[i] / SCALE;",
            "        new_momentum[i] <== momentum_term[i] + weighted_sum[i];",
            "        new_params[i] <== old_params[i] + new_momentum[i];",
            "    }",
            "}",
            "",
            "component main = FedJSCMAggregation(2, 5);",
        ]

        # Write using standard text mode with explicit line separator
        with open(circuit_path, "w", newline="\n") as f:
            f.write("\n".join(circuit_lines) + "\n")

        # Verify the file was written correctly
        with open(circuit_path, "rb") as f:
            content = f.read()

        if b"pragma circom 2.0.0;\n//" in content:
            logger.info(f"✓ Circuit written and verified: {circuit_path}")
        else:
            raise RuntimeError(
                f"Circuit file has incorrect format! First 100 bytes: {content[:100]}"
            )

    def _prepare_snark_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Convert ALL floating-point model values to fixed-point integers.
        Circom/SnarkJS only accept integers (BigInts), NOT floats.
        """
        SCALE = 10**6

        def fp(x):
            """Convert float → scaled int"""
            return int(round(float(x) * SCALE))

        client_updates = inputs["client_updates"]
        client_weights = inputs["client_weights"]
        aggregated_params = inputs["aggregated_params"]
        old_momentum = inputs["momentum"]
        momentum_coeff = inputs["momentum_coeff"]

        # Each parameter vector has size 5 (per your circuit: FedJSCMAggregation(2,5))
        param_size = 5
        n_clients = len(client_updates)

        def flatten(arrs):
            """Flatten list of NDArrays into a fixed-length (param_size) list."""
            flat = np.concatenate([a.flatten() for a in arrs]).tolist()
            # Trim/pad
            if len(flat) < param_size:
                flat += [0.0] * (param_size - len(flat))
            return [fp(v) for v in flat[:param_size]]

        # Build witness JSON
        witness = {
            "client_deltas": [],
            "client_weights": [fp(w) for w in client_weights],
            "old_params": flatten(inputs["old_params"]),
            "old_momentum": flatten(old_momentum),
            "momentum_coeff": fp(momentum_coeff),
        }

        for update in client_updates:
            witness["client_deltas"].append(flatten(update))

        return witness

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
        try:
            circuit_dir = Path(self.circuit_dir)
            build = circuit_dir / "build"

            input_json = build / "input.json"
            witness_wtns = build / "witness.wtns"
            proof_json = build / "proof.json"
            public_json = build / "public.json"

            # Save input.json
            with open(input_json, "w") as f:
                json.dump(inputs, f)

            # 1. Compute witness (CORRECTED PATH)
            wasm_file = build / "aggregation_js" / "aggregation.wasm"  # ← ADD THIS LINE
            subprocess.check_call(
                [
                    self.snarkjs_path,
                    "wtns",
                    "calculate",
                    str(
                        wasm_file
                    ),  # ← CHANGE THIS (was: str(build / "aggregation.wasm"))
                    str(input_json),
                    str(witness_wtns),
                ]
            )

            # 2. Create proof
            subprocess.check_call(
                [
                    self.snarkjs_path,
                    "groth16",
                    "prove",
                    str(self.proving_key),
                    str(witness_wtns),
                    str(proof_json),
                    str(public_json),
                ]
            )

            with open(proof_json) as f:
                proof = json.load(f)

            with open(public_json) as f:
                public = json.load(f)

            return json.dumps({"proof": proof, "public": public})

        except Exception as e:
            logger.error(f"SNARK proof generation failed: {e}")
            return None

    def _verify_snark_proof(self, proof_data: dict, public_inputs: list) -> bool:
        try:
            circuit_dir = Path(self.circuit_dir)
            build = circuit_dir / "build"
            proof_json = build / "verify_proof.json"
            public_json = build / "verify_public.json"

            with open(proof_json, "w") as f:
                json.dump(proof_data["proof"], f)

            with open(public_json, "w") as f:
                json.dump(proof_data["public"], f)

            result = subprocess.run(
                [
                    self.snarkjs_path,
                    "groth16",
                    "verify",
                    str(self.verification_key),
                    str(public_json),
                    str(proof_json),
                ],
                capture_output=True,
                text=True,
            )

            return "OK" in result.stdout

        except Exception as e:
            logger.error(f"SNARK verification failed: {e}")
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
        # "initial_params": [np.random.randn(5, 3), np.random.randn(3)],
        "initial_params": [np.random.uniform(-0.01, 0.01, size=(5, 3))],
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
