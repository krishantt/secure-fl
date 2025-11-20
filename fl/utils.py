"""
Utility functions for Federated Learning with ZKP integration

This module provides common utility functions for:
1. Parameter conversion between different formats
2. Weight quantization for ZKP circuits
3. Cryptographic utilities
4. Data handling helpers
"""

import logging
import hashlib
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from flwr.common import NDArrays, Parameters
import struct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Convert Flower Parameters to NumPy arrays"""
    return [np.array(param) for param in parameters.tensors]


def ndarrays_to_parameters(arrays: NDArrays) -> Parameters:
    """Convert NumPy arrays to Flower Parameters"""
    tensors = [arr.astype(np.float32) for arr in arrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def torch_to_ndarrays(model: torch.nn.Module) -> NDArrays:
    """Convert PyTorch model parameters to NumPy arrays"""
    return [param.detach().cpu().numpy() for param in model.parameters()]


def ndarrays_to_torch(arrays: NDArrays, model: torch.nn.Module) -> None:
    """Load NumPy arrays into PyTorch model parameters"""
    params = [torch.tensor(arr) for arr in arrays]
    for param, new_param in zip(model.parameters(), params):
        param.data.copy_(new_param)


def compute_parameter_norm(parameters: NDArrays, norm_type: int = 2) -> float:
    """Compute norm of parameter arrays"""
    if norm_type == 2:
        total_norm = 0.0
        for param in parameters:
            param_norm = np.linalg.norm(param)
            total_norm += param_norm**2
        return total_norm**0.5
    elif norm_type == 1:
        return sum(np.sum(np.abs(param)) for param in parameters)
    elif norm_type == float("inf"):
        return max(np.max(np.abs(param)) for param in parameters)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


def compute_parameter_diff(params1: NDArrays, params2: NDArrays) -> NDArrays:
    """Compute element-wise difference between parameter arrays"""
    if len(params1) != len(params2):
        raise ValueError("Parameter arrays must have same length")

    diff = []
    for p1, p2 in zip(params1, params2):
        if p1.shape != p2.shape:
            raise ValueError(f"Parameter shapes don't match: {p1.shape} vs {p2.shape}")
        diff.append(p2 - p1)

    return diff


def quantize_parameters(
    parameters: NDArrays, bits: int = 8, symmetric: bool = True
) -> NDArrays:
    """
    Quantize parameters to fixed-point representation for ZKP circuits

    Args:
        parameters: List of parameter arrays
        bits: Number of bits for quantization
        symmetric: Whether to use symmetric quantization around zero

    Returns:
        Quantized parameter arrays
    """
    if bits not in [4, 8, 16]:
        logger.warning(f"Unusual bit width {bits}, may cause issues in ZKP circuits")

    quantized = []
    max_val = 2 ** (bits - 1) - 1 if symmetric else 2**bits - 1
    min_val = -(2 ** (bits - 1)) if symmetric else 0

    for param in parameters:
        # Find scaling factor
        param_max = np.max(np.abs(param)) if symmetric else np.max(param)
        param_min = np.min(param) if not symmetric else -param_max

        if param_max == 0:
            # Handle zero parameters
            quantized.append(np.zeros_like(param, dtype=np.int32))
            continue

        if symmetric:
            scale = param_max / max_val
        else:
            scale = (param_max - param_min) / (max_val - min_val)

        # Quantize
        if symmetric:
            quantized_param = np.round(param / scale).astype(np.int32)
        else:
            quantized_param = np.round((param - param_min) / scale).astype(np.int32)

        # Clip to valid range
        quantized_param = np.clip(quantized_param, min_val, max_val)
        quantized.append(quantized_param)

    return quantized


def dequantize_parameters(
    quantized_params: List[np.ndarray],
    original_params: NDArrays,
    bits: int = 8,
    symmetric: bool = True,
) -> NDArrays:
    """
    Dequantize parameters back to floating point

    Args:
        quantized_params: Quantized integer parameters
        original_params: Original parameters to get scaling info
        bits: Number of bits used in quantization
        symmetric: Whether symmetric quantization was used

    Returns:
        Dequantized floating-point parameters
    """
    dequantized = []
    max_val = 2 ** (bits - 1) - 1 if symmetric else 2**bits - 1

    for quant_param, orig_param in zip(quantized_params, original_params):
        # Reconstruct scaling factor
        orig_max = np.max(np.abs(orig_param)) if symmetric else np.max(orig_param)
        orig_min = np.min(orig_param) if not symmetric else -orig_max

        if orig_max == 0:
            dequantized.append(np.zeros_like(orig_param))
            continue

        if symmetric:
            scale = orig_max / max_val
            deq_param = quant_param.astype(np.float32) * scale
        else:
            scale = (orig_max - orig_min) / (2**bits - 1)
            deq_param = quant_param.astype(np.float32) * scale + orig_min

        dequantized.append(deq_param)

    return dequantized


def compute_hash(data: Union[NDArrays, str, bytes], algorithm: str = "sha256") -> str:
    """
    Compute cryptographic hash of data

    Args:
        data: Data to hash (parameters, string, or bytes)
        algorithm: Hash algorithm ('sha256', 'sha3_256', 'blake2b')

    Returns:
        Hex-encoded hash string
    """
    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha3_256":
        hasher = hashlib.sha3_256()
    elif algorithm == "blake2b":
        hasher = hashlib.blake2b()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    if isinstance(data, str):
        hasher.update(data.encode("utf-8"))
    elif isinstance(data, bytes):
        hasher.update(data)
    elif isinstance(data, list):  # NDArrays
        for arr in data:
            hasher.update(arr.tobytes())
    else:
        hasher.update(str(data).encode("utf-8"))

    return hasher.hexdigest()


def serialize_parameters(parameters: NDArrays) -> bytes:
    """Serialize parameters to bytes for storage/transmission"""
    serialized = b""

    # Add header with number of arrays
    serialized += struct.pack("I", len(parameters))

    for param in parameters:
        # Add array metadata
        shape = param.shape
        dtype = param.dtype

        # Serialize shape
        serialized += struct.pack("I", len(shape))
        for dim in shape:
            serialized += struct.pack("I", dim)

        # Serialize dtype
        dtype_str = str(dtype)
        serialized += struct.pack("I", len(dtype_str))
        serialized += dtype_str.encode("utf-8")

        # Serialize data
        data_bytes = param.tobytes()
        serialized += struct.pack("I", len(data_bytes))
        serialized += data_bytes

    return serialized


def deserialize_parameters(data: bytes) -> NDArrays:
    """Deserialize parameters from bytes"""
    parameters = []
    offset = 0

    # Read number of arrays
    num_arrays = struct.unpack("I", data[offset : offset + 4])[0]
    offset += 4

    for _ in range(num_arrays):
        # Read shape
        shape_len = struct.unpack("I", data[offset : offset + 4])[0]
        offset += 4

        shape = []
        for _ in range(shape_len):
            dim = struct.unpack("I", data[offset : offset + 4])[0]
            shape.append(dim)
            offset += 4

        # Read dtype
        dtype_len = struct.unpack("I", data[offset : offset + 4])[0]
        offset += 4
        dtype_str = data[offset : offset + dtype_len].decode("utf-8")
        offset += dtype_len

        # Read data
        data_len = struct.unpack("I", data[offset : offset + 4])[0]
        offset += 4
        array_data = data[offset : offset + data_len]
        offset += data_len

        # Reconstruct array
        param = np.frombuffer(array_data, dtype=dtype_str).reshape(shape)
        parameters.append(param)

    return parameters


def aggregate_weighted_average(
    parameter_lists: List[NDArrays], weights: List[float]
) -> NDArrays:
    """
    Compute weighted average of parameter lists

    Args:
        parameter_lists: List of parameter arrays from different clients
        weights: Weights for each client (should sum to 1)

    Returns:
        Weighted average parameters
    """
    if not parameter_lists:
        raise ValueError("Empty parameter list")

    if len(parameter_lists) != len(weights):
        raise ValueError("Parameter lists and weights must have same length")

    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Total weight is zero")
    weights = [w / total_weight for w in weights]

    # Initialize result with zeros
    num_layers = len(parameter_lists[0])
    result = [np.zeros_like(parameter_lists[0][i]) for i in range(num_layers)]

    # Weighted sum
    for params, weight in zip(parameter_lists, weights):
        for i, layer_params in enumerate(params):
            result[i] += weight * layer_params

    return result


def create_parameter_mask(parameters: NDArrays, sparsity: float) -> List[np.ndarray]:
    """
    Create binary masks for parameter sparsification

    Args:
        parameters: Parameter arrays
        sparsity: Fraction of parameters to mask (0.0 to 1.0)

    Returns:
        Binary masks for each parameter array
    """
    masks = []

    for param in parameters:
        if sparsity == 0.0:
            mask = np.ones_like(param, dtype=bool)
        elif sparsity == 1.0:
            mask = np.zeros_like(param, dtype=bool)
        else:
            # Create random mask
            flat_param = param.flatten()
            num_zeros = int(len(flat_param) * sparsity)

            mask_flat = np.ones(len(flat_param), dtype=bool)
            zero_indices = np.random.choice(len(flat_param), num_zeros, replace=False)
            mask_flat[zero_indices] = False

            mask = mask_flat.reshape(param.shape)

        masks.append(mask)

    return masks


def apply_parameter_mask(parameters: NDArrays, masks: List[np.ndarray]) -> NDArrays:
    """Apply binary masks to parameters"""
    if len(parameters) != len(masks):
        raise ValueError("Parameters and masks must have same length")

    masked_params = []
    for param, mask in zip(parameters, masks):
        if param.shape != mask.shape:
            raise ValueError(
                f"Parameter and mask shapes don't match: {param.shape} vs {mask.shape}"
            )

        masked_param = param * mask.astype(param.dtype)
        masked_params.append(masked_param)

    return masked_params


def validate_parameters(parameters: NDArrays) -> bool:
    """
    Validate parameter arrays for common issues

    Args:
        parameters: Parameter arrays to validate

    Returns:
        True if parameters are valid, False otherwise
    """
    try:
        if not parameters:
            logger.error("Empty parameter list")
            return False

        for i, param in enumerate(parameters):
            if not isinstance(param, np.ndarray):
                logger.error(f"Parameter {i} is not a numpy array: {type(param)}")
                return False

            if param.size == 0:
                logger.error(f"Parameter {i} is empty")
                return False

            if not np.isfinite(param).all():
                logger.error(f"Parameter {i} contains non-finite values")
                return False

            # Check for extreme values that might cause issues
            if np.abs(param).max() > 1e6:
                logger.warning(
                    f"Parameter {i} contains very large values: {np.abs(param).max()}"
                )

            if np.abs(param).max() < 1e-10 and param.size > 1:
                logger.warning(
                    f"Parameter {i} contains very small values: {np.abs(param).max()}"
                )

        return True

    except Exception as e:
        logger.error(f"Parameter validation failed: {e}")
        return False


def get_parameter_stats(parameters: NDArrays) -> Dict[str, Any]:
    """
    Compute statistics for parameter arrays

    Args:
        parameters: Parameter arrays

    Returns:
        Dictionary with parameter statistics
    """
    stats = {
        "num_layers": len(parameters),
        "total_parameters": sum(param.size for param in parameters),
        "layer_shapes": [param.shape for param in parameters],
        "layer_sizes": [param.size for param in parameters],
    }

    # Flatten all parameters
    all_params = np.concatenate([param.flatten() for param in parameters])

    stats.update(
        {
            "mean": float(np.mean(all_params)),
            "std": float(np.std(all_params)),
            "min": float(np.min(all_params)),
            "max": float(np.max(all_params)),
            "norm_l1": float(np.sum(np.abs(all_params))),
            "norm_l2": float(np.linalg.norm(all_params)),
            "sparsity": float(np.mean(all_params == 0)),
        }
    )

    return stats


# Test functions
def test_parameter_conversion():
    """Test parameter conversion functions"""
    print("Testing parameter conversion...")

    # Create test parameters
    params = [np.random.randn(3, 4), np.random.randn(5)]

    # Test Flower conversion
    flower_params = ndarrays_to_parameters(params)
    recovered_params = parameters_to_ndarrays(flower_params)

    assert len(params) == len(recovered_params)
    for orig, rec in zip(params, recovered_params):
        assert np.allclose(orig, rec)

    print("Parameter conversion test passed!")


def test_quantization():
    """Test quantization functions"""
    print("Testing quantization...")

    # Create test parameters
    params = [np.random.randn(10, 5), np.random.randn(3)]

    # Test quantization and dequantization
    quantized = quantize_parameters(params, bits=8)
    dequantized = dequantize_parameters(quantized, params, bits=8)

    # Check that quantized are integers
    for qp in quantized:
        assert qp.dtype in [np.int32, np.int64]

    # Check that dequantized have similar shape
    assert len(params) == len(dequantized)
    for orig, deq in zip(params, dequantized):
        assert orig.shape == deq.shape

    print("Quantization test passed!")


if __name__ == "__main__":
    test_parameter_conversion()
    test_quantization()
    print("All utility tests passed!")
