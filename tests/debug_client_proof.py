# debug_client_proof.py
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secure_fl.proof_manager import ClientProofManager, ServerProofManager


def main():
    # Fake old parameters
    old_params = [
        np.random.randn(4, 3).astype(np.float32),
        np.random.randn(3).astype(np.float32),
    ]

    # Generate updated params first
    new_params = [
        w + (0.01 * np.random.randn(*w.shape)).astype(np.float32) for w in old_params
    ]

    # Compute delta to avoid rounding mismatch
    delta = [new - old for new, old in zip(new_params, old_params)]

    # Client Proof Manager
    cpm = ClientProofManager(max_update_norm=10.0)
    cpm.setup()

    proof_inputs = {
        "client_id": "client_1",
        "round": 1,
        "data_commitment": "dummy_commitment",
        "initial_params": old_params,
        "updated_params": new_params,
        "param_delta": delta,
        "learning_rate": 0.01,
        "local_epochs": 1,
        "rigor_level": "medium",
        "total_samples": 128,
    }

    proof_str = cpm.generate_training_proof(proof_inputs)
    print("\nGenerated Proof JSON:\n", proof_str)

    # Server Verify
    spm = ServerProofManager()
    spm.setup_complete = True

    ok = spm.verify_client_proof(
        proof=proof_str,
        updated_parameters=new_params,
        old_global_params=old_params,
    )

    print("\nVerification result:", ok)


if __name__ == "__main__":
    main()
