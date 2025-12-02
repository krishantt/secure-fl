"""
Quantization Module for ZKP Circuit Compatibility

This module provides specialized quantization techniques for making ML parameters
compatible with zero-knowledge proof circuits (zk-STARKs and zk-SNARKs).

Key features:
1. Fixed-point quantization with configurable bit widths
2. Uniform and non-uniform quantization schemes
3. Gradient-aware quantization for better training dynamics
4. Circuit-optimized formats for Cairo and Circom
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from flwr.common import NDArrays

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for quantization parameters"""

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        signed: bool = True,
        per_channel: bool = False,
        stochastic: bool = False,
        gradient_aware: bool = True,
        circuit_friendly: bool = True,
        scale_method: str = "minmax",
        zero_point_dtype: type = np.uint8,
    ):
        # Validate bits
        if not (1 <= bits <= 32):
            raise ValueError("Bits must be between 1 and 32")

        # Validate scale_method
        if scale_method not in ["minmax", "percentile"]:
            raise ValueError("Invalid scale method")

        self.bits = bits
        self.symmetric = symmetric
        self.signed = signed
        self.per_channel = per_channel
        self.stochastic = stochastic
        self.gradient_aware = gradient_aware
        self.circuit_friendly = circuit_friendly
        self.scale_method = scale_method
        self.zero_point_dtype = zero_point_dtype

    @property
    def qmin(self) -> int:
        """Minimum quantization value"""
        if self.signed:
            if self.symmetric:
                return -(2 ** (self.bits - 1) - 1)
            else:
                return -(2 ** (self.bits - 1))
        else:
            return 0

    @property
    def qmax(self) -> int:
        """Maximum quantization value"""
        if self.signed:
            return 2 ** (self.bits - 1) - 1
        else:
            if self.symmetric:
                return 2**self.bits - 1
            else:
                return 2**self.bits - 1

    def __repr__(self):
        return (
            f"QuantizationConfig(bits={self.bits}, symmetric={self.symmetric}, "
            f"signed={self.signed}, per_channel={self.per_channel}, "
            f"stochastic={self.stochastic}, gradient_aware={self.gradient_aware}, "
            f"circuit_friendly={self.circuit_friendly})"
        )


class FixedPointQuantizer:
    """
    Fixed-point quantizer optimized for ZKP circuits

    Converts floating-point parameters to fixed-point representation
    that can be efficiently processed in arithmetic circuits.
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.scale_factors = {}
        self.zero_points = {}

        # Initialize attributes expected by tests
        self.scale = None
        self.zero_point = None

        # Compute quantization bounds
        if self.config.signed:
            self.qmax = 2 ** (self.config.bits - 1) - 1
            self.qmin = -(2 ** (self.config.bits - 1))
        else:
            self.qmax = 2**self.config.bits - 1
            self.qmin = 0

        logger.info(
            f"FixedPointQuantizer initialized: bits={config.bits}, "
            f"range=[{self.qmin}, {self.qmax}]"
        )

    def calibrate(
        self, parameters: NDArrays, layer_names: Optional[List[str]] = None
    ) -> None:
        """
        Calibrate quantization parameters based on data statistics

        Args:
            parameters: Parameter arrays to calibrate on
            layer_names: Optional layer names for tracking
        """
        if layer_names is None:
            layer_names = [f"layer_{i}" for i in range(len(parameters))]

        for i, (param, name) in enumerate(zip(parameters, layer_names)):
            if self.config.per_channel and len(param.shape) > 1:
                # Per-channel quantization for conv/linear layers
                scales, zero_points = self._compute_per_channel_params(param)
            else:
                # Per-tensor quantization
                scales, zero_points = self._compute_per_tensor_params(param)

            self.scale_factors[name] = scales
            self.zero_points[name] = zero_points

        logger.info(f"Calibrated quantization for {len(parameters)} layers")

    def quantize(
        self, parameters: NDArrays, layer_names: Optional[List[str]] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Quantize parameters to fixed-point representation

        Args:
            parameters: Parameter arrays to quantize
            layer_names: Optional layer names

        Returns:
            Tuple of (quantized_parameters, quantization_metadata)
        """
        if layer_names is None:
            layer_names = [f"layer_{i}" for i in range(len(parameters))]

        quantized_params = []
        metadata = {
            "scales": {},
            "zero_points": {},
            "config": self.config,
            "bounds": {"qmin": self.qmin, "qmax": self.qmax},
        }

        for param, name in zip(parameters, layer_names):
            if name not in self.scale_factors:
                # Auto-calibrate if not done before
                if self.config.per_channel and len(param.shape) > 1:
                    scales, zero_points = self._compute_per_channel_params(param)
                else:
                    scales, zero_points = self._compute_per_tensor_params(param)

                self.scale_factors[name] = scales
                self.zero_points[name] = zero_points

            # Quantize parameter
            quantized = self._quantize_array(
                param, self.scale_factors[name], self.zero_points[name]
            )

            quantized_params.append(quantized)
            metadata["scales"][name] = self.scale_factors[name]
            metadata["zero_points"][name] = self.zero_points[name]

        return quantized_params, metadata

    def dequantize(
        self, quantized_params: List[np.ndarray], metadata: Dict[str, Any]
    ) -> NDArrays:
        """
        Dequantize parameters back to floating-point

        Args:
            quantized_params: Quantized parameter arrays
            metadata: Quantization metadata from quantize()

        Returns:
            Dequantized floating-point parameters
        """
        # Validate metadata structure
        if "scales" not in metadata:
            raise ValueError("Missing 'scales' in metadata")
        if "zero_points" not in metadata:
            raise ValueError("Missing 'zero_points' in metadata")

        dequantized_params = []
        layer_names = list(metadata["scales"].keys())

        # Validate lengths match
        if len(quantized_params) != len(layer_names):
            raise ValueError(
                f"Length mismatch: {len(quantized_params)} quantized params vs {len(layer_names)} metadata entries"
            )

        for quantized, name in zip(quantized_params, layer_names):
            if name not in metadata["scales"]:
                raise ValueError(f"Missing scale for layer {name}")
            if name not in metadata["zero_points"]:
                raise ValueError(f"Missing zero_point for layer {name}")

            scales = metadata["scales"][name]
            zero_points = metadata["zero_points"][name]

            dequantized = self._dequantize_array(quantized, scales, zero_points)
            dequantized_params.append(dequantized)

        return dequantized_params

    def _compute_per_tensor_params(self, param: np.ndarray) -> Tuple[float, float]:
        """Compute per-tensor quantization parameters"""
        if self.config.symmetric:
            # Symmetric quantization around zero
            abs_max = np.max(np.abs(param))
            if abs_max == 0:
                scale = 1.0
            else:
                scale = abs_max / self.qmax
            zero_point = 0.0
        else:
            # Asymmetric quantization
            min_val = np.min(param)
            max_val = np.max(param)

            if max_val == min_val:
                scale = 1.0
                zero_point = 0.0
            else:
                scale = (max_val - min_val) / (self.qmax - self.qmin)
                zero_point = self.qmin - min_val / scale
                zero_point = np.clip(zero_point, self.qmin, self.qmax)
                zero_point = int(np.round(zero_point))

        return scale, zero_point

    def _compute_per_channel_params(
        self, param: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-channel quantization parameters for conv/linear layers"""
        if len(param.shape) == 2:  # Linear layer (out_features, in_features)
            axis = 0
        elif len(param.shape) == 4:  # Conv layer (out_channels, in_channels, h, w)
            axis = 0
        else:
            # Fallback to per-tensor
            scale, zero_point = self._compute_per_tensor_params(param)
            return np.array([scale]), np.array([zero_point])

        # Compute per-channel statistics
        channel_max = np.max(np.abs(param), axis=tuple(range(1, len(param.shape))))

        if self.config.symmetric:
            scales = channel_max / self.qmax
            scales = np.where(scales == 0, 1.0, scales)  # Avoid division by zero
            zero_points = np.zeros_like(scales)
        else:
            channel_min = np.min(param, axis=tuple(range(1, len(param.shape))))
            channel_max_val = np.max(param, axis=tuple(range(1, len(param.shape))))

            scales = (channel_max_val - channel_min) / (self.qmax - self.qmin)
            scales = np.where(scales == 0, 1.0, scales)

            zero_points = self.qmin - channel_min / scales
            zero_points = np.clip(zero_points, self.qmin, self.qmax)
            zero_points = np.round(zero_points).astype(int)

        return scales, zero_points

    def _quantize_array(
        self,
        param: np.ndarray,
        scale: Union[float, np.ndarray],
        zero_point: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Quantize a single parameter array"""

        if self.config.stochastic and np.isscalar(scale):
            # Stochastic rounding for better gradient flow
            scaled = param / scale + zero_point
            floor_scaled = np.floor(scaled)
            prob = scaled - floor_scaled
            stochastic_round = floor_scaled + (np.random.random(scaled.shape) < prob)
            quantized = stochastic_round.astype(np.int32)
        else:
            # Deterministic rounding
            if np.isscalar(scale):
                scaled = param / scale + zero_point
            else:
                # Per-channel scaling
                if len(param.shape) == 2:  # Linear
                    scaled = param / scale.reshape(-1, 1) + zero_point.reshape(-1, 1)
                elif len(param.shape) == 4:  # Conv
                    scaled = param / scale.reshape(-1, 1, 1, 1) + zero_point.reshape(
                        -1, 1, 1, 1
                    )
                else:
                    scaled = param / scale + zero_point

            quantized = np.round(scaled).astype(np.int32)

        # Clamp to valid range
        quantized = np.clip(quantized, self.qmin, self.qmax)

        return quantized

    def _dequantize_array(
        self,
        quantized: np.ndarray,
        scale: Union[float, np.ndarray],
        zero_point: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Dequantize a single parameter array"""

        if np.isscalar(scale):
            dequantized = (quantized.astype(np.float32) - zero_point) * scale
        else:
            # Per-channel descaling
            if len(quantized.shape) == 2:  # Linear
                dequantized = (
                    quantized.astype(np.float32) - zero_point.reshape(-1, 1)
                ) * scale.reshape(-1, 1)
            elif len(quantized.shape) == 4:  # Conv
                dequantized = (
                    quantized.astype(np.float32) - zero_point.reshape(-1, 1, 1, 1)
                ) * scale.reshape(-1, 1, 1, 1)
            else:
                dequantized = (quantized.astype(np.float32) - zero_point) * scale

        return dequantized


class GradientAwareQuantizer(FixedPointQuantizer):
    """
    Gradient-aware quantizer that considers gradient magnitudes during quantization

    This helps preserve important parameters that have large gradients,
    which is crucial for maintaining training dynamics in federated learning.
    """

    def __init__(self, config: QuantizationConfig, sensitivity_factor: float = 0.5):
        super().__init__(config)
        self.sensitivity_factor = sensitivity_factor
        self.gradient_history = {}
        self.gradient_weights = {}

    def quantize(
        self,
        parameters: NDArrays,
        layer_names: Optional[List[str]] = None,
        gradients: Optional[NDArrays] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Quantize parameters with optional gradient awareness

        Args:
            parameters: Parameter arrays to quantize
            layer_names: Optional layer names
            gradients: Optional gradient arrays for gradient-aware quantization

        Returns:
            Tuple of (quantized_parameters, quantization_metadata)
        """
        if gradients is not None and self.config.gradient_aware:
            self.update_gradients(gradients, layer_names)

        quantized_params, metadata = super().quantize(parameters, layer_names)

        # Add gradient information to metadata if gradients were provided
        if gradients is not None and self.config.gradient_aware:
            if layer_names is None:
                layer_names = [f"layer_{i}" for i in range(len(gradients))]

            gradient_norms = {}
            for grad, name in zip(gradients, layer_names):
                gradient_norms[name] = float(np.linalg.norm(grad))

            metadata["gradient_norms"] = gradient_norms

        return quantized_params, metadata

    def update_gradients(
        self, gradients: NDArrays, layer_names: Optional[List[str]] = None
    ):
        """Update gradient history for gradient-aware quantization"""
        if not self.config.gradient_aware:
            return

        if layer_names is None:
            layer_names = [f"layer_{i}" for i in range(len(gradients))]

        for grad, name in zip(gradients, layer_names):
            if name not in self.gradient_history:
                self.gradient_history[name] = []

            # Store gradient magnitude
            grad_mag = np.abs(grad)
            self.gradient_history[name].append(grad_mag)

            # Keep only recent history
            if len(self.gradient_history[name]) > 10:
                self.gradient_history[name].pop(0)

            # Compute gradient weights (higher for important parameters)
            avg_grad = np.mean(self.gradient_history[name], axis=0)
            grad_weight = avg_grad / (np.mean(avg_grad) + 1e-8)  # Normalize
            self.gradient_weights[name] = grad_weight

    def _compute_per_tensor_params(
        self, param: np.ndarray, layer_name: str = None
    ) -> Tuple[float, float]:
        """Compute gradient-aware per-tensor quantization parameters"""
        if not self.config.gradient_aware or layer_name not in self.gradient_weights:
            return super()._compute_per_tensor_params(param)

        # Weight quantization range by gradient importance
        grad_weights = self.gradient_weights[layer_name]
        weighted_param = param * np.sqrt(grad_weights + 0.1)  # Add small constant

        if self.config.symmetric:
            abs_max = np.max(np.abs(weighted_param))
            if abs_max == 0:
                scale = 1.0
            else:
                scale = abs_max / self.qmax
            zero_point = 0.0
        else:
            min_val = np.min(weighted_param)
            max_val = np.max(weighted_param)

            if max_val == min_val:
                scale = 1.0
                zero_point = 0.0
            else:
                scale = (max_val - min_val) / (self.qmax - self.qmin)
                zero_point = self.qmin - min_val / scale
                zero_point = np.clip(zero_point, self.qmin, self.qmax)
                zero_point = int(np.round(zero_point))

        return scale, zero_point


def quantize_parameters(
    parameters: NDArrays, config: QuantizationConfig
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Convenience function for parameter quantization

    Args:
        parameters: Parameter arrays to quantize
        config: Quantization configuration

    Returns:
        Tuple of (quantized_parameters, quantization_metadata)
    """
    quantizer = FixedPointQuantizer(config)
    return quantizer.quantize(parameters)


def dequantize_parameters(
    quantized_params: List[np.ndarray],
    metadata: Dict[str, Any],
    config: QuantizationConfig,
) -> NDArrays:
    """
    Convenience function for parameter dequantization

    Args:
        quantized_params: Quantized parameter arrays
        metadata: Quantization metadata from quantize_parameters
        config: Quantization configuration

    Returns:
        Dequantized floating-point parameters
    """
    quantizer = FixedPointQuantizer(config)
    return quantizer.dequantize(quantized_params, metadata)


def compute_quantization_error(
    original: NDArrays, dequantized: NDArrays
) -> Dict[str, float]:
    """
    Compute quantization error metrics

    Args:
        original: Original floating-point parameters
        dequantized: Dequantized floating-point parameters

    Returns:
        Dictionary with error metrics
    """
    errors = {}

    # Compute per-layer errors
    layer_mse = []
    layer_mae = []
    layer_snr = []

    for orig, dequant in zip(original, dequantized):
        mse = np.mean((orig - dequant) ** 2)
        mae = np.mean(np.abs(orig - dequant))

        signal_power = np.mean(orig**2)
        noise_power = mse
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

        layer_mse.append(mse)
        layer_mae.append(mae)
        layer_snr.append(snr)

    errors.update(
        {
            "mse_per_layer": layer_mse,
            "mae_per_layer": layer_mae,
            "snr_per_layer": layer_snr,
            "avg_mse": np.mean(layer_mse),
            "avg_mae": np.mean(layer_mae),
            "avg_snr": np.mean(layer_snr),
        }
    )

    # Overall statistics
    all_orig = np.concatenate([p.flatten() for p in original])
    all_dequant = np.concatenate([p.flatten() for p in dequantized])

    overall_mse = np.mean((all_orig - all_dequant) ** 2)
    overall_mae = np.mean(np.abs(all_orig - all_dequant))
    overall_snr = 10 * np.log10(np.mean(all_orig**2) / (overall_mse + 1e-10))

    errors.update(
        {
            "overall_mse": overall_mse,
            "overall_mae": overall_mae,
            "overall_snr": overall_snr,
            "mse": overall_mse,  # Test expects this key
            "mae": overall_mae,  # Test expects this key
            "max_error": np.max(
                np.abs(all_orig - all_dequant)
            ),  # Test expects this key
            "snr_db": overall_snr,  # Test expects this key
        }
    )

    return errors


def test_quantization():
    """Test quantization functionality"""
    print("Testing quantization module...")

    # Create test parameters
    params = [
        np.random.randn(64, 32).astype(np.float32),  # Linear layer
        np.random.randn(32).astype(np.float32),  # Bias
        np.random.randn(16, 32, 3, 3).astype(np.float32),  # Conv layer
    ]

    # Test basic quantization
    config = QuantizationConfig(bits=8, symmetric=True)
    quantizer = FixedPointQuantizer(config)

    quantized, metadata = quantizer.quantize(params)
    dequantized = quantizer.dequantize(quantized, metadata)

    # Check shapes
    assert len(params) == len(quantized) == len(dequantized)
    for orig, quant, dequant in zip(params, quantized, dequantized):
        assert orig.shape == quant.shape == dequant.shape
        assert quant.dtype in [np.int32, np.int64]
        assert dequant.dtype in [np.float32, np.float64]

    # Test error computation
    errors = compute_quantization_error(params, quantized, dequantized)
    assert "avg_mse" in errors
    assert "avg_snr" in errors

    print(f"Quantization test passed! Average SNR: {errors['avg_snr']:.2f} dB")

    # Test gradient-aware quantization
    print("Testing gradient-aware quantization...")

    config_ga = QuantizationConfig(bits=8, symmetric=True, gradient_aware=True)
    ga_quantizer = GradientAwareQuantizer(config_ga)

    # Simulate gradients
    gradients = [np.random.randn(*p.shape) for p in params]
    ga_quantizer.update_gradients(gradients)

    ga_quantized, ga_metadata = ga_quantizer.quantize(params)
    ga_dequantized = ga_quantizer.dequantize(ga_quantized, ga_metadata)

    ga_errors = compute_quantization_error(params, ga_quantized, ga_dequantized)
    print(
        f"Gradient-aware quantization test passed! Average SNR: {ga_errors['avg_snr']:.2f} dB"
    )

    print("All quantization tests passed!")


if __name__ == "__main__":
    test_quantization()
