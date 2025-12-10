import json

import numpy as np

from secure_fl.proof_manager import ClientProofManager, ServerProofManager


def main():
    # Create fake parameters
    initial = [
        np.random.randn(5, 3).astype(np.float32),
        np.random.randn(3).astype(np.float32),
    ]
    updated = [
        p + 0.001 * np.random.randn(*p.shape).astype(np.float32) for p in initial
    ]

    # Compute delta
    delta = [u - i for u, i in zip(updated, initial, strict=False)]

    # Create proof manager with PySNARK enabled
    cpm = ClientProofManager(
        max_update_norm=10.0, use_pysnark=True, fixed_point_scale=1
    )

    proof_json = cpm.generate_training_proof(
        {
            "client_id": "client_1",
            "round": 1,
            "data_commitment": "dummy_commitment",
            "initial_params": initial,
            "updated_params": updated,
            "param_delta": delta,
            "learning_rate": 0.01,
            "local_epochs": 1,
            "rigor_level": "medium",
            "total_samples": 128,
        }
    )

    print("Generated proof JSON:")
    print(proof_json)

    proof_obj = json.loads(proof_json)
    print("\nPySNARK section:")
    print(json.dumps(proof_obj.get("pysnark", {}), indent=2))

    # Basic server-side check still works as before
    spm = ServerProofManager()
    spm.setup_complete = True  # we keep SNARK part mocked for now

    ok = spm.verify_client_proof(
        proof_json,
        updated_parameters=updated,
        old_global_params=initial,
    )
    print(f"\nServer verification result: {ok}")


if __name__ == "__main__":
    main()
