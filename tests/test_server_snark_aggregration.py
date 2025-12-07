"""
TEST: Real zk-SNARK Aggregation Proof (Server Side)

This test checks:
1. Circom circuit exists and compiles (or is already compiled)
2. ServerProofManager prepares witness correctly
3. snarkjs generates a Groth16 proof
4. snarkjs verifies the proof

You MUST run this test AFTER placing:
secure_fl/proofs/server/aggregation.circom
secure_fl/proofs/server/pot12_final.ptau
"""

import json
import numpy as np
from pathlib import Path

from secure_fl.proof_manager import ServerProofManager


def make_fake_updates():
    """Create small fake client updates for testing."""
    # Each update contains 2 tensors
    u1 = [np.random.randn(4, 1).astype(np.float32), np.random.randn(1).astype(np.float32)]
    u2 = [np.random.randn(4, 1).astype(np.float32), np.random.randn(1).astype(np.float32)]

    updates = [u1, u2]
    weights = [0.5, 0.5]

    # Fake global params & momentum
    old_params = [np.random.randn(4, 1).astype(np.float32), np.random.randn(1).astype(np.float32)]
    momentum = [np.zeros((4, 1), dtype=np.float32), np.zeros((1,), dtype=np.float32)]

    # Aggregated params = simple weighted avg
    stacked = [(u1[i] + u2[i]) / 2 for i in range(2)]

    return updates, weights, stacked, old_params, momentum


def run_test():
    print("=== SNARK Aggregation Test ===")

    pm = ServerProofManager()
    pm.setup_complete = True  # we call setup manually

    circuit_dir = Path(pm.circuit_dir)
    build_dir = circuit_dir / "build"
    zkey = build_dir / "aggregation.zkey"

    # Step 1: Ensure circuit setup is complete
    if not zkey.exists():
        print("⛔ aggregation.zkey missing — running setup...")
        ok = pm._setup_snark_circuit()
        if not ok:
            print("❌ Circuit setup failed")
            return
        print("✓ Circuit compiled & proving key created.")
    else:
        print("✓ Circuit already built.")

    # Step 2: Prepare fake data
    updates, weights, aggregated, old_params, momentum = make_fake_updates()

    witness_dict = pm._prepare_snark_inputs({
        "client_updates": updates,
        "client_weights": weights,
        "aggregated_params": aggregated,
        "old_params": old_params,
        "momentum": momentum,
        "momentum_coeff": 0.9,
        "new_momentum": aggregated,  # small shortcut: treat as new momentum
    })

    print("Prepared witness:")
    print(json.dumps(witness_dict, indent=2))

    # Step 3: Generate proof
    print("\n=== Generating Proof ===")
    proof_json = pm._generate_snark_proof(witness_dict)

    if proof_json is None:
        print("❌ Proof generation failed")
        return

    proof_data = json.loads(proof_json)
    print("✓ Proof generated")

    # Step 4: Verify proof
    print("\n=== Verifying Proof ===")
    ok = pm._verify_snark_proof(proof_data, proof_data.get("public", []))

    if ok:
        print("🟢 SNARK VERIFICATION SUCCESS")
    else:
        print("🔴 SNARK VERIFICATION FAILED")

    print("\n=== TEST COMPLETE ===")


if __name__ == "__main__":
    run_test()
