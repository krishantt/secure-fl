"""
Unit Tests for Quantization Module

This module contains comprehensive unit tests for the quantization functionality
in the secure_fl.quantization module.
"""

import numpy as np
import pytest

from secure_fl.quantization import (
    FixedPointQuantizer,
    GradientAwareQuantizer,
    QuantizationConfig,
    compute_quantization_error,
    dequantize_parameters,
    quantize_parameters,
)


class TestQuantizationConfig:
    """Test QuantizationConfig class"""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default quantization configuration"""
        config = QuantizationConfig()

        assert config.bits == 8
        assert config.symmetric is True
        assert config.signed is True
        assert config.per_channel is False
        assert config.scale_method == "minmax"
        assert config.zero_point_dtype == np.uint8

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom quantization configuration"""
        config = QuantizationConfig(
            bits=16,
            symmetric=False,
            signed=False,
            per_channel=True,
            scale_method="percentile",
        )

        assert config.bits == 16
        assert config.symmetric is False
        assert config.signed is False
        assert config.per_channel is True
        assert config.scale_method == "percentile"

    @pytest.mark.unit
    def test_invalid_bits(self):
        """Test invalid bit configuration"""
        with pytest.raises(ValueError, match="Bits must be between 1 and 32"):
            QuantizationConfig(bits=0)

        with pytest.raises(ValueError, match="Bits must be between 1 and 32"):
            QuantizationConfig(bits=33)

    @pytest.mark.unit
    def test_invalid_scale_method(self):
        """Test invalid scale method"""
        with pytest.raises(ValueError, match="Invalid scale method"):
            QuantizationConfig(scale_method="invalid")

    @pytest.mark.unit
    def test_qmin_qmax_computation(self):
        """Test quantization range computation"""
        # Signed 8-bit symmetric
        config = QuantizationConfig(bits=8, signed=True, symmetric=True)
        assert config.qmin == -127
        assert config.qmax == 127

        # Unsigned 8-bit symmetric
        config = QuantizationConfig(bits=8, signed=False, symmetric=True)
        assert config.qmin == 0
        assert config.qmax == 255

        # Signed 8-bit asymmetric
        config = QuantizationConfig(bits=8, signed=True, symmetric=False)
        assert config.qmin == -128
        assert config.qmax == 127

        # 4-bit signed
        config = QuantizationConfig(bits=4, signed=True, symmetric=True)
        assert config.qmin == -7
        assert config.qmax == 7


class TestFixedPointQuantizer:
    """Test FixedPointQuantizer class"""

    @pytest.mark.unit
    def test_initialization(self, quantization_config):
        """Test quantizer initialization"""
        quantizer = FixedPointQuantizer(quantization_config)

        assert quantizer.config == quantization_config
        assert quantizer.scale is None
        assert quantizer.zero_point is None

    @pytest.mark.unit
    def test_quantize_single_tensor(self):
        """Test quantization of single tensor"""
        config = QuantizationConfig(bits=8, symmetric=True, signed=True)
        quantizer = FixedPointQuantizer(config)

        # Create test tensor
        tensor = np.random.randn(5, 3).astype(np.float32)

        quantized, metadata = quantizer.quantize([tensor])

        assert len(quantized) == 1
        assert isinstance(quantized[0], np.ndarray)
        assert quantized[0].dtype in [np.int32, np.int64]
        assert isinstance(metadata, dict)
        assert "scales" in metadata
        assert "zero_points" in metadata

    @pytest.mark.unit
    def test_quantize_multiple_tensors(self, model_parameters):
        """Test quantization of multiple tensors"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        quantized, metadata = quantizer.quantize(model_parameters)

        assert len(quantized) == len(model_parameters)
        assert len(metadata["scales"]) == len(model_parameters)
        assert len(metadata["zero_points"]) == len(model_parameters)

        # Check shapes are preserved
        for orig, quant in zip(model_parameters, quantized):
            assert orig.shape == quant.shape

    @pytest.mark.unit
    def test_dequantize_basic(self, model_parameters):
        """Test basic dequantization"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        quantized, metadata = quantizer.quantize(model_parameters)
        dequantized = quantizer.dequantize(quantized, metadata)

        assert len(dequantized) == len(model_parameters)

        # Check shapes are preserved
        for orig, dequant in zip(model_parameters, dequantized):
            assert orig.shape == dequant.shape
            assert dequant.dtype == np.float32

    @pytest.mark.unit
    def test_quantization_roundtrip(self, model_parameters):
        """Test quantization round-trip accuracy"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        quantized, metadata = quantizer.quantize(model_parameters)
        dequantized = quantizer.dequantize(quantized, metadata)

        # Check round-trip error is reasonable
        for orig, dequant in zip(model_parameters, dequantized):
            # Allow for quantization error
            relative_error = np.mean(np.abs(orig - dequant) / (np.abs(orig) + 1e-8))
            assert relative_error < 0.1  # Less than 10% relative error on average

    @pytest.mark.unit
    def test_different_bit_widths(self):
        """Test quantization with different bit widths"""
        tensor = np.random.randn(10, 10).astype(np.float32)

        for bits in [4, 8, 16]:
            config = QuantizationConfig(bits=bits)
            quantizer = FixedPointQuantizer(config)

            quantized, metadata = quantizer.quantize([tensor])
            dequantized = quantizer.dequantize(quantized, metadata)

            # Higher bit width should give better accuracy
            error = np.mean(np.abs(tensor - dequantized[0]))
            assert error < 1.0  # Reasonable error bound

    @pytest.mark.unit
    def test_symmetric_vs_asymmetric(self):
        """Test symmetric vs asymmetric quantization"""
        # Create tensor with asymmetric range
        tensor = np.random.uniform(-1, 3, (10, 10)).astype(np.float32)

        # Symmetric quantization
        config_sym = QuantizationConfig(bits=8, symmetric=True)
        quantizer_sym = FixedPointQuantizer(config_sym)

        # Asymmetric quantization
        config_asym = QuantizationConfig(bits=8, symmetric=False)
        quantizer_asym = FixedPointQuantizer(config_asym)

        quant_sym, meta_sym = quantizer_sym.quantize([tensor])
        quant_asym, meta_asym = quantizer_asym.quantize([tensor])

        dequant_sym = quantizer_sym.dequantize(quant_sym, meta_sym)
        dequant_asym = quantizer_asym.dequantize(quant_asym, meta_asym)

        # Both should work, but asymmetric might be more accurate for this tensor
        error_sym = np.mean(np.abs(tensor - dequant_sym[0]))
        error_asym = np.mean(np.abs(tensor - dequant_asym[0]))

        assert error_sym < 1.0
        assert error_asym < 1.0

    @pytest.mark.unit
    def test_per_channel_quantization(self):
        """Test per-channel vs per-tensor quantization"""
        # Create tensor where per-channel might help
        tensor = np.random.randn(5, 10).astype(np.float32)
        # Make different channels have different scales
        for i in range(5):
            tensor[i] *= i + 1

        # Per-tensor quantization
        config_tensor = QuantizationConfig(bits=8, per_channel=False)
        quantizer_tensor = FixedPointQuantizer(config_tensor)

        # Per-channel quantization
        config_channel = QuantizationConfig(bits=8, per_channel=True)
        quantizer_channel = FixedPointQuantizer(config_channel)

        quant_tensor, meta_tensor = quantizer_tensor.quantize([tensor])
        quant_channel, meta_channel = quantizer_channel.quantize([tensor])

        dequant_tensor = quantizer_tensor.dequantize(quant_tensor, meta_tensor)
        dequant_channel = quantizer_channel.dequantize(quant_channel, meta_channel)

        # Per-channel should typically be more accurate
        error_tensor = np.mean(np.abs(tensor - dequant_tensor[0]))
        error_channel = np.mean(np.abs(tensor - dequant_channel[0]))

        # Both should work
        assert error_tensor < 1.0
        assert error_channel < 1.0

    @pytest.mark.unit
    def test_zero_tensor(self):
        """Test quantization of zero tensor"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        zero_tensor = np.zeros((5, 3), dtype=np.float32)

        quantized, metadata = quantizer.quantize([zero_tensor])
        dequantized = quantizer.dequantize(quantized, metadata)

        # Should handle zero tensor gracefully
        assert np.allclose(dequantized[0], zero_tensor, atol=1e-6)

    @pytest.mark.unit
    def test_constant_tensor(self):
        """Test quantization of constant tensor"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        constant_tensor = np.full((5, 3), 2.5, dtype=np.float32)

        quantized, metadata = quantizer.quantize([constant_tensor])
        dequantized = quantizer.dequantize(quantized, metadata)

        # Should preserve constant values well
        assert np.allclose(dequantized[0], constant_tensor, rtol=0.01)

    @pytest.mark.unit
    def test_extreme_values(self):
        """Test quantization with extreme values"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        # Very large values
        large_tensor = np.random.randn(5, 3).astype(np.float32) * 1000

        quantized, metadata = quantizer.quantize([large_tensor])
        dequantized = quantizer.dequantize(quantized, metadata)

        # Should handle large values without overflow
        assert np.isfinite(dequantized[0]).all()

    @pytest.mark.unit
    def test_small_values(self):
        """Test quantization with very small values"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        # Very small values
        small_tensor = np.random.randn(5, 3).astype(np.float32) * 1e-6

        quantized, metadata = quantizer.quantize([small_tensor])
        dequantized = quantizer.dequantize(quantized, metadata)

        # Should handle small values
        assert np.isfinite(dequantized[0]).all()


class TestGradientAwareQuantizer:
    """Test GradientAwareQuantizer class"""

    @pytest.mark.unit
    def test_initialization(self, quantization_config):
        """Test gradient-aware quantizer initialization"""
        quantizer = GradientAwareQuantizer(quantization_config, sensitivity_factor=0.1)

        assert quantizer.config == quantization_config
        assert quantizer.sensitivity_factor == 0.1

    @pytest.mark.unit
    def test_quantize_with_gradients(self, model_parameters):
        """Test quantization with gradient information"""
        config = QuantizationConfig(bits=8)
        quantizer = GradientAwareQuantizer(config)

        # Create mock gradients
        gradients = []
        for param in model_parameters:
            grad = np.random.randn(*param.shape).astype(np.float32) * 0.1
            gradients.append(grad)

        quantized, metadata = quantizer.quantize(model_parameters, gradients=gradients)

        assert len(quantized) == len(model_parameters)
        assert "gradient_norms" in metadata
        assert len(metadata["gradient_norms"]) == len(model_parameters)

    @pytest.mark.unit
    def test_quantize_without_gradients(self, model_parameters):
        """Test quantization without gradient information"""
        config = QuantizationConfig(bits=8)
        quantizer = GradientAwareQuantizer(config)

        # Should fall back to standard quantization
        quantized, metadata = quantizer.quantize(model_parameters)

        assert len(quantized) == len(model_parameters)

    @pytest.mark.unit
    def test_different_sensitivity_factors(self, model_parameters):
        """Test different sensitivity factors"""
        gradients = []
        for param in model_parameters:
            grad = np.random.randn(*param.shape).astype(np.float32) * 0.1
            gradients.append(grad)

        config = QuantizationConfig(bits=8)

        # High sensitivity
        quantizer_high = GradientAwareQuantizer(config, sensitivity_factor=1.0)
        quant_high, meta_high = quantizer_high.quantize(
            model_parameters, gradients=gradients
        )

        # Low sensitivity
        quantizer_low = GradientAwareQuantizer(config, sensitivity_factor=0.01)
        quant_low, meta_low = quantizer_low.quantize(
            model_parameters, gradients=gradients
        )

        # Both should work
        assert len(quant_high) == len(model_parameters)
        assert len(quant_low) == len(model_parameters)


class TestUtilityFunctions:
    """Test utility functions for quantization"""

    @pytest.mark.unit
    def test_quantize_parameters_function(self, model_parameters):
        """Test standalone quantize_parameters function"""
        config = QuantizationConfig(bits=8)

        quantized, metadata = quantize_parameters(model_parameters, config)

        assert len(quantized) == len(model_parameters)
        assert isinstance(metadata, dict)

    @pytest.mark.unit
    def test_dequantize_parameters_function(self, model_parameters):
        """Test standalone dequantize_parameters function"""
        config = QuantizationConfig(bits=8)

        quantized, metadata = quantize_parameters(model_parameters, config)
        dequantized = dequantize_parameters(quantized, metadata, config)

        assert len(dequantized) == len(model_parameters)

        for orig, dequant in zip(model_parameters, dequantized):
            assert orig.shape == dequant.shape

    @pytest.mark.unit
    def test_compute_quantization_error(self, model_parameters):
        """Test quantization error computation"""
        config = QuantizationConfig(bits=8)

        quantized, metadata = quantize_parameters(model_parameters, config)
        dequantized = dequantize_parameters(quantized, metadata, config)

        error = compute_quantization_error(model_parameters, dequantized)

        assert isinstance(error, dict)
        assert "mse" in error
        assert "mae" in error
        assert "max_error" in error
        assert "snr_db" in error

        # All error metrics should be non-negative
        assert error["mse"] >= 0
        assert error["mae"] >= 0
        assert error["max_error"] >= 0

    @pytest.mark.unit
    def test_compute_quantization_error_perfect(self):
        """Test error computation with identical tensors"""
        params = [np.random.randn(5, 3).astype(np.float32)]

        error = compute_quantization_error(params, params)

        # Should be zero error
        assert error["mse"] < 1e-10
        assert error["mae"] < 1e-10
        assert error["max_error"] < 1e-10


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.unit
    def test_empty_parameter_list(self):
        """Test quantization with empty parameter list"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        quantized, metadata = quantizer.quantize([])

        assert len(quantized) == 0
        assert len(metadata["scales"]) == 0
        assert len(metadata["zero_points"]) == 0

    @pytest.mark.unit
    def test_single_element_tensor(self):
        """Test quantization with single-element tensors"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        single_element = [np.array([3.14])]

        quantized, metadata = quantizer.quantize(single_element)
        dequantized = quantizer.dequantize(quantized, metadata)

        assert len(dequantized) == 1
        assert dequantized[0].shape == (1,)

    @pytest.mark.unit
    def test_mixed_tensor_shapes(self):
        """Test quantization with mixed tensor shapes"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        mixed_shapes = [
            np.random.randn(5, 3).astype(np.float32),
            np.random.randn(10).astype(np.float32),
            np.random.randn(2, 2, 3).astype(np.float32),
        ]

        quantized, metadata = quantizer.quantize(mixed_shapes)
        dequantized = quantizer.dequantize(quantized, metadata)

        assert len(dequantized) == 3
        for orig, dequant in zip(mixed_shapes, dequantized):
            assert orig.shape == dequant.shape

    @pytest.mark.unit
    def test_invalid_metadata(self, model_parameters):
        """Test dequantization with invalid metadata"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        quantized, metadata = quantizer.quantize(model_parameters)

        # Corrupt metadata by removing one layer's scale
        corrupted_metadata = metadata.copy()
        corrupted_metadata["scales"] = corrupted_metadata["scales"].copy()
        # Remove the first layer's scale
        first_layer_key = list(corrupted_metadata["scales"].keys())[0]
        del corrupted_metadata["scales"][first_layer_key]

        with pytest.raises((ValueError, IndexError)):
            quantizer.dequantize(quantized, corrupted_metadata)

    @pytest.mark.unit
    def test_mismatched_quantized_length(self, model_parameters):
        """Test dequantization with mismatched lengths"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        quantized, metadata = quantizer.quantize(model_parameters)

        # Remove one quantized tensor
        corrupted_quantized = quantized[:-1]

        with pytest.raises((ValueError, IndexError)):
            quantizer.dequantize(corrupted_quantized, metadata)

    @pytest.mark.unit
    def test_nan_input(self):
        """Test quantization with NaN inputs"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        nan_tensor = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)

        # Should handle NaN gracefully (either raise error or handle it)
        try:
            quantized, metadata = quantizer.quantize([nan_tensor])
            # If it doesn't raise an error, check result is valid
            assert len(quantized) == 1
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for NaN inputs
            pass

    @pytest.mark.unit
    def test_inf_input(self):
        """Test quantization with infinite inputs"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        inf_tensor = np.array([1.0, 2.0, np.inf, 4.0], dtype=np.float32)

        # Should handle infinity gracefully
        try:
            quantized, metadata = quantizer.quantize([inf_tensor])
            assert len(quantized) == 1
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for infinite inputs
            pass


class TestPerformance:
    """Test performance aspects of quantization"""

    @pytest.mark.unit
    def test_quantization_preserves_memory_layout(self, model_parameters):
        """Test that quantization preserves memory layout when possible"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        # Ensure contiguous arrays
        contiguous_params = [np.ascontiguousarray(p) for p in model_parameters]

        quantized, metadata = quantizer.quantize(contiguous_params)

        # Quantized arrays should also be contiguous
        for quant in quantized:
            assert quant.flags.c_contiguous

    @pytest.mark.unit
    def test_large_tensor_quantization(self):
        """Test quantization with large tensors"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        # Large tensor (but not too large for CI)
        large_tensor = np.random.randn(1000, 1000).astype(np.float32)

        quantized, metadata = quantizer.quantize([large_tensor])
        dequantized = quantizer.dequantize(quantized, metadata)

        assert dequantized[0].shape == large_tensor.shape

        # Check memory usage is reasonable
        # Quantized should use less memory (in practice, this depends on implementation)
        assert quantized[0].nbytes <= large_tensor.nbytes

    @pytest.mark.unit
    def test_batch_quantization_efficiency(self):
        """Test that batch quantization is efficient"""
        config = QuantizationConfig(bits=8)
        quantizer = FixedPointQuantizer(config)

        # Create multiple tensors
        tensors = [np.random.randn(100, 100).astype(np.float32) for _ in range(10)]

        # Batch quantization
        quantized, metadata = quantizer.quantize(tensors)
        dequantized = quantizer.dequantize(quantized, metadata)

        assert len(quantized) == 10
        assert len(dequantized) == 10

        # All shapes should be preserved
        for orig, dequant in zip(tensors, dequantized):
            assert orig.shape == dequant.shape
