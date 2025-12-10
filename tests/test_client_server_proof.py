# tests/test_client_server_proof.py
from types import SimpleNamespace

import numpy as np
from flwr.common import Code, FitRes, Status

from secure_fl.proof_manager import ClientProofManager
from secure_fl.server import SecureFlowerStrategy
from secure_fl.utils import ndarrays_to_parameters


def make_dummy_params():
    return [np.zeros((2, 2), dtype=np.float32), np.zeros((2,), dtype=np.float32)]


def test_server_verifies_valid_client_proof():
    print("\n=== Test 1: Valid Proof ===")

    initial_params = make_dummy_params()
    strategy = SecureFlowerStrategy(
        initial_parameters=ndarrays_to_parameters(initial_params),
        enable_zkp=True,
        proof_rigor="medium",
    )
    strategy.current_global_params = initial_params

    cpm = ClientProofManager(max_update_norm=10, use_pysnark=True)

    updated = [p + 0.01 for p in initial_params]
    delta = [u - i for u, i in zip(updated, initial_params, strict=False)]

    proof_json = cpm.generate_training_proof(
        {
            "client_id": "c1",
            "round": 1,
            "data_commitment": "x",
            "initial_params": initial_params,
            "updated_params": updated,
            "param_delta": delta,
            "learning_rate": 0.01,
            "local_epochs": 1,
            "rigor_level": "medium",
            "total_samples": 10,
        }
    )

    assert proof_json is not None, "Proof generation failed"

    fit_res = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(updated),
        num_examples=10,
        metrics={"zkp_proof": proof_json},
    )
    dummy_client = SimpleNamespace(cid="c1")

    ok = strategy._verify_client_proof(dummy_client, fit_res)
    print(f"Verification result = {ok}")
    assert ok is True, "Valid proof should be accepted"
    print("✓ Test passed: Server accepted valid proof\n")


def test_server_rejects_modified_proof():
    print("\n=== Test 2: Tampered Proof ===")

    initial_params = make_dummy_params()
    strategy = SecureFlowerStrategy(
        initial_parameters=ndarrays_to_parameters(initial_params),
        enable_zkp=True,
    )
    strategy.current_global_params = initial_params

    tampered_proof = '{"type": "client_training_proof", "updated_hash": "WRONG", "delta_hash": "FAKE"}'

    fit_res = FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(make_dummy_params()),
        num_examples=10,
        metrics={"zkp_proof": tampered_proof},
    )
    dummy_client = SimpleNamespace(cid="cX")

    ok = strategy._verify_client_proof(dummy_client, fit_res)
    print(f"Tampered verification result = {ok}")
    assert ok is False, "Tampered proof should be rejected"
    print("✓ Test passed: Server rejected tampered proof\n")


if __name__ == "__main__":
    test_server_verifies_valid_client_proof()
    test_server_rejects_modified_proof()
    print("\n=== ALL TESTS PASSED ===")
