"""
Test: Client PySNARK Proof Generation (with graceful fallback)

This test verifies:
1. The client trains for 1 round
2. A proof JSON object is produced
3. PySNARK metadata is included
4. The test passes even if PySNARK is not fully installed
"""

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from secure_fl.client import SecureFlowerClient
from secure_fl.utils import torch_to_ndarrays


# -----------------------------
# Simple model for testing
# -----------------------------
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# -----------------------------
# Synthetic test dataset
# -----------------------------
def create_toy_dataset():
    X = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    return loader


# -----------------------------
# Run the test
# -----------------------------
def run_client_test():
    print("\n=== Running Client PySNARK Proof Test ===\n")

    train_loader = create_toy_dataset()
    model = SimpleModel()

    client = SecureFlowerClient(
        client_id="client_test_1",
        model=model,
        train_loader=train_loader,
        val_loader=None,
        enable_zkp=True,
        proof_rigor="medium",
        quantize_weights=False,
        local_epochs=1,
        learning_rate=0.01,
    )

    # Initial parameters (random for test)
    initial_params = torch_to_ndarrays(model)

    config = {"server_round": 1, "local_epochs": 1}

    # Run 1 training round
    updated_params, num_examples, metrics = client.fit(initial_params, config)

    print("\n=== CLIENT TEST OUTPUT ===")
    print("Num examples:", num_examples)
    print("Metrics keys:", list(metrics.keys()))

    assert "zkp_proof" in metrics, "Proof not found in client metrics"

    # Proof JSON string → load as dict
    proof_json = metrics["zkp_proof"]
    proof = json.loads(proof_json)

    print("\n--- Proof Object ---")
    print(proof)

    # Extract PySNARK metadata block
    pysnark_info = proof.get("pysnark", {})
    print("\n--- PySNARK Section ---")
    print(pysnark_info)

    # ==============================================
    # Validate PySNARK status (enabled OR disabled)
    # ==============================================
    if pysnark_info.get("enabled"):
        print("\nPySNARK ENABLED ✓")

        assert "vector_len" in pysnark_info
        assert "bound_float" in pysnark_info
        assert "l2_sq_result" in pysnark_info

    else:
        # Safe fallback: PySNARK not installed or backend missing
        print("\nPySNARK DISABLED — Reason:", pysnark_info.get("reason", "unknown"))
        assert "reason" in pysnark_info

    print("\nTEST PASSED ✓\n")


# -----------------------------
# Run when executed directly
# -----------------------------
if __name__ == "__main__":
    run_client_test()
