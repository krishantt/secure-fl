"""
Utility functions for Federated Learning with ZKP integration

This module provides common utility functions for:
1. Parameter conversion between different formats
2. Weight quantization for ZKP circuits
3. Cryptographic utilities
4. Data handling helpers
"""

import hashlib
import logging
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from flwr.common import NDArrays, Parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Convert Flower Parameters to list of NumPy arrays.

    Args:
        parameters: Flower Parameters object

    Returns:
        List of NumPy arrays
    """
    return [
        np.frombuffer(tensor, dtype=np.float32).reshape(shape)
        for tensor, shape in zip(parameters.tensors, parameters.tensor_type)
    ]


def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    """Convert list of NumPy arrays to Flower Parameters.

    Args:
        ndarrays: List of NumPy arrays

    Returns:
        Flower Parameters object
    """
    tensors = []
    tensor_types = []

    for array in ndarrays:
        tensors.append(array.astype(np.float32).tobytes())
        tensor_types.append(array.shape)

    return Parameters(tensors=tensors, tensor_type=tensor_types)


def torch_to_ndarrays(model: torch.nn.Module) -> NDArrays:
    """Convert PyTorch model parameters to NumPy arrays.

    Args:
        model: PyTorch model

    Returns:
        List of NumPy arrays
    """
    return [param.detach().cpu().numpy() for param in model.parameters()]


def ndarrays_to_torch(model: torch.nn.Module, ndarrays: NDArrays) -> None:
    """Load NumPy arrays into PyTorch model.

    Args:
        model: PyTorch model
        ndarrays: List of NumPy arrays (only learnable parameters)
    """
    # Only update learnable parameters to match torch_to_ndarrays
    params = list(model.parameters())

    if len(params) != len(ndarrays):
        raise ValueError(
            f"Number of model parameters ({len(params)}) does not match "
            f"number of arrays ({len(ndarrays)})"
        )

    for param, array in zip(params, ndarrays):
        # Make array writable to avoid PyTorch warning
        if not array.flags.writeable:
            array = array.copy()
        param.data = torch.from_numpy(array)


def compute_parameter_norm(parameters: NDArrays, norm_type: str = "l2") -> float:
    """Compute norm of parameters.

    Args:
        parameters: List of parameter arrays
        norm_type: Type of norm ('l1', 'l2', 'linf')

    Returns:
        Computed norm value
    """
    if norm_type == "l2":
        total_norm = 0.0
        for param in parameters:
            total_norm += np.sum(param**2)
        return float(np.sqrt(total_norm))
    elif norm_type == "l1":
        total_norm = 0.0
        for param in parameters:
            total_norm += np.sum(np.abs(param))
        return float(total_norm)
    elif norm_type == "linf":
        max_norm = 0.0
        for param in parameters:
            max_norm = max(max_norm, np.max(np.abs(param)))
        return float(max_norm)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


def compute_parameter_diff(params1: NDArrays, params2: NDArrays) -> float:
    """Compute difference between two parameter sets.

    Args:
        params1: First parameter set
        params2: Second parameter set

    Returns:
        L2 norm of parameter difference
    """
    if len(params1) != len(params2):
        raise ValueError("Parameter lists must have same length")

    diff_norm = 0.0
    for p1, p2 in zip(params1, params2):
        if p1.shape != p2.shape:
            raise ValueError("Parameter shapes must match")
        diff_norm += np.sum((p1 - p2) ** 2)

    return np.sqrt(diff_norm)


def validate_parameters(parameters: Optional[NDArrays]) -> bool:
    """Validate parameter arrays.

    Args:
        parameters: List of parameter arrays to validate

    Returns:
        True if valid, False otherwise
    """
    if parameters is None:
        return False

    if not isinstance(parameters, list):
        return False

    for param in parameters:
        if not isinstance(param, np.ndarray):
            return False

        if param.size == 0:
            return False

        if np.isnan(param).any() or np.isinf(param).any():
            return False

    return True


def compute_hash(parameters: NDArrays) -> str:
    """Compute SHA-256 hash of parameters.

    Args:
        parameters: List of parameter arrays

    Returns:
        Hexadecimal hash string
    """
    serialized = serialize_parameters(parameters)
    return hashlib.sha256(serialized).hexdigest()


def serialize_parameters(parameters: NDArrays) -> bytes:
    """Serialize parameters to bytes.

    Args:
        parameters: List of parameter arrays

    Returns:
        Serialized bytes
    """
    import pickle

    return pickle.dumps(parameters)


def deserialize_parameters(data: bytes) -> NDArrays:
    """Deserialize parameters from bytes.

    Args:
        data: Serialized parameter data

    Returns:
        List of parameter arrays
    """
    import pickle

    return pickle.loads(data)


def aggregate_weighted_average(
    client_updates: List[NDArrays], client_weights: List[float]
) -> NDArrays:
    """Compute weighted average of client parameter updates.

    Args:
        client_updates: List of client parameter updates
        client_weights: List of client weights

    Returns:
        Aggregated parameters
    """
    if len(client_updates) != len(client_weights):
        raise ValueError("Number of updates must match number of weights")

    if not client_updates:
        raise ValueError("At least one client update required")

    # Initialize aggregated parameters
    aggregated = []
    for param_idx in range(len(client_updates[0])):
        param_shape = client_updates[0][param_idx].shape
        aggregated_param = np.zeros(param_shape, dtype=np.float32)

        # Weighted sum
        for client_idx, weight in enumerate(client_weights):
            aggregated_param += weight * client_updates[client_idx][param_idx]

        aggregated.append(aggregated_param)

    return aggregated


def get_parameter_stats(parameters: NDArrays) -> Dict[str, Any]:
    """Get statistics about parameters.

    Args:
        parameters: List of parameter arrays

    Returns:
        Dictionary with parameter statistics
    """
    if not parameters:
        return {}

    all_values = np.concatenate([param.flatten() for param in parameters])

    return {
        "mean": float(np.mean(all_values)),
        "std": float(np.std(all_values)),
        "min": float(np.min(all_values)),
        "max": float(np.max(all_values)),
        "total_params": int(sum(param.size for param in parameters)),
        "shape_info": [param.shape for param in parameters],
    }
