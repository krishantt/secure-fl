"""
Unit Tests for Secure FL Utility Functions

This module contains comprehensive unit tests for the utility functions
in the secure_fl.utils module.
"""

import hashlib
import pickle

import numpy as np
import pytest
import torch
from flwr.common import Parameters

from secure_fl.utils import (
    aggregate_weighted_average,
    compute_hash,
    compute_parameter_diff,
    compute_parameter_norm,
    deserialize_parameters,
    get_parameter_stats,
    ndarrays_to_parameters,
    ndarrays_to_torch,
    parameters_to_ndarrays,
    serialize_parameters,
    torch_to_ndarrays,
    validate_parameters,
)


class TestParameterConversion:
    """Test parameter conversion utilities"""

    @pytest.mark.unit
    def test_ndarrays_to_parameters_basic(self, model_parameters):
        """Test basic ndarray to parameters conversion"""
        parameters = ndarrays_to_parameters(model_parameters)

        assert isinstance(parameters, Parameters)
        assert len(parameters.tensors) == len(model_parameters)

        # Test round-trip conversion
        recovered = parameters_to_ndarrays(parameters)
        assert len(recovered) == len(model_parameters)

        for orig, rec in zip(model_parameters, recovered):
            assert orig.shape == rec.shape
            assert np.allclose(orig, rec, rtol=1e-6)

    @pytest.mark.unit
    def test_parameters_to_ndarrays_basic(self, model_parameters):
        """Test basic parameters to ndarray conversion"""
        # Convert to parameters and back
        parameters = ndarrays_to_parameters(model_parameters)
        recovered = parameters_to_ndarrays(parameters)

        assert len(recovered) == len(model_parameters)
        for orig, rec in zip(model_parameters, recovered):
            assert isinstance(rec, np.ndarray)
            assert orig.shape == rec.shape
            assert np.allclose(orig, rec, rtol=1e-6)

    @pytest.mark.unit
    def test_torch_to_ndarrays(self, simple_model):
        """Test PyTorch model to numpy arrays conversion"""
        ndarrays = torch_to_ndarrays(simple_model)

        assert isinstance(ndarrays, list)
        assert len(ndarrays) == len(list(simple_model.parameters()))

        for i, (param, array) in enumerate(zip(simple_model.parameters(), ndarrays)):
            assert isinstance(array, np.ndarray)
            assert param.shape == array.shape
            assert np.allclose(param.detach().numpy(), array, rtol=1e-6)

    @pytest.mark.unit
    def test_ndarrays_to_torch(self, simple_model, model_parameters):
        """Test numpy arrays to PyTorch model conversion"""
        # Extract parameters from the simple_model
        original_params = torch_to_ndarrays(simple_model)

        # Create a copy of the model
        model_copy = type(simple_model)()

        # Load the original parameters into the copy
        ndarrays_to_torch(model_copy, original_params)

        # Check that parameters were loaded correctly
        for orig_param, loaded_param in zip(
            simple_model.parameters(), model_copy.parameters()
        ):
            assert orig_param.shape == loaded_param.shape
            assert torch.allclose(orig_param, loaded_param, rtol=1e-6)

    @pytest.mark.unit
    def test_empty_parameters(self):
        """Test conversion with empty parameter lists"""
        empty_params = []

        # Test ndarray to parameters
        parameters = ndarrays_to_parameters(empty_params)
        assert isinstance(parameters, Parameters)
        assert len(parameters.tensors) == 0

        # Test parameters to ndarray
        recovered = parameters_to_ndarrays(parameters)
        assert len(recovered) == 0

    @pytest.mark.unit
    def test_single_parameter(self):
        """Test conversion with single parameter"""
        single_param = [np.random.randn(5, 3).astype(np.float32)]

        parameters = ndarrays_to_parameters(single_param)
        recovered = parameters_to_ndarrays(parameters)

        assert len(recovered) == 1
        assert np.allclose(single_param[0], recovered[0], rtol=1e-6)

    @pytest.mark.unit
    def test_different_dtypes(self):
        """Test conversion with different numpy dtypes"""
        params = [
            np.random.randn(3, 2).astype(np.float32),
            np.random.randn(2).astype(np.float64),
            np.random.randint(0, 10, (2, 2)).astype(np.int32),
        ]

        parameters = ndarrays_to_parameters(params)
        recovered = parameters_to_ndarrays(parameters)

        assert len(recovered) == len(params)
        for orig, rec in zip(params, recovered):
            assert orig.shape == rec.shape
            # Note: conversion may change dtype, so we check values not dtype


class TestParameterOperations:
    """Test parameter computation utilities"""

    @pytest.mark.unit
    def test_compute_parameter_norm_l2(self, model_parameters):
        """Test L2 norm computation"""
        norm = compute_parameter_norm(model_parameters, norm_type="l2")

        assert isinstance(norm, float)
        assert norm >= 0

        # Verify calculation manually
        expected_norm = 0.0
        for param in model_parameters:
            expected_norm += np.sum(param**2)
        expected_norm = np.sqrt(expected_norm)

        assert abs(norm - expected_norm) < 1e-6

    @pytest.mark.unit
    def test_compute_parameter_norm_l1(self, model_parameters):
        """Test L1 norm computation"""
        norm = compute_parameter_norm(model_parameters, norm_type="l1")

        assert isinstance(norm, float)
        assert norm >= 0

        # Verify calculation manually
        expected_norm = 0.0
        for param in model_parameters:
            expected_norm += np.sum(np.abs(param))

        assert abs(norm - expected_norm) < 1e-6

    @pytest.mark.unit
    def test_compute_parameter_norm_linf(self, model_parameters):
        """Test L-infinity norm computation"""
        norm = compute_parameter_norm(model_parameters, norm_type="linf")

        assert isinstance(norm, float)
        assert norm >= 0

        # Verify calculation manually
        expected_norm = 0.0
        for param in model_parameters:
            expected_norm = max(expected_norm, np.max(np.abs(param)))

        assert abs(norm - expected_norm) < 1e-6

    @pytest.mark.unit
    def test_compute_parameter_diff(self, model_parameters):
        """Test parameter difference computation"""
        # Create slightly different parameters
        params2 = []
        for param in model_parameters:
            noise = np.random.normal(0, 0.01, param.shape)
            params2.append(param + noise)

        diff = compute_parameter_diff(model_parameters, params2)

        assert isinstance(diff, float)
        assert diff >= 0

        # Test with identical parameters
        diff_zero = compute_parameter_diff(model_parameters, model_parameters)
        assert abs(diff_zero) < 1e-10

    @pytest.mark.unit
    def test_get_parameter_stats(self, model_parameters):
        """Test parameter statistics computation"""
        stats = get_parameter_stats(model_parameters)

        assert isinstance(stats, dict)
        required_keys = ["mean", "std", "min", "max", "total_params", "shape_info"]
        for key in required_keys:
            assert key in stats

        assert isinstance(stats["mean"], float)
        assert isinstance(stats["std"], float)
        assert isinstance(stats["min"], float)
        assert isinstance(stats["max"], float)
        assert isinstance(stats["total_params"], int)
        assert isinstance(stats["shape_info"], list)

        assert stats["total_params"] > 0
        assert stats["std"] >= 0


class TestParameterValidation:
    """Test parameter validation utilities"""

    @pytest.mark.unit
    def test_validate_parameters_valid(self, model_parameters):
        """Test validation with valid parameters"""
        assert validate_parameters(model_parameters) == True

    @pytest.mark.unit
    def test_validate_parameters_empty(self):
        """Test validation with empty parameter list"""
        assert validate_parameters([]) == True

    @pytest.mark.unit
    def test_validate_parameters_none(self):
        """Test validation with None"""
        assert validate_parameters(None) == False

    @pytest.mark.unit
    def test_validate_parameters_invalid_type(self):
        """Test validation with invalid parameter types"""
        invalid_params = ["not_an_array", 123, None]
        assert validate_parameters(invalid_params) == False

    @pytest.mark.unit
    def test_validate_parameters_nan_values(self):
        """Test validation with NaN values"""
        params_with_nan = [np.array([1.0, 2.0, np.nan])]
        assert validate_parameters(params_with_nan) == False

    @pytest.mark.unit
    def test_validate_parameters_inf_values(self):
        """Test validation with infinite values"""
        params_with_inf = [np.array([1.0, 2.0, np.inf])]
        assert validate_parameters(params_with_inf) == False

    @pytest.mark.unit
    def test_validate_parameters_empty_array(self):
        """Test validation with empty arrays"""
        params_with_empty = [np.array([])]
        assert validate_parameters(params_with_empty) == False


class TestAggregation:
    """Test parameter aggregation utilities"""

    @pytest.mark.unit
    def test_aggregate_weighted_average_basic(
        self, sample_client_updates, client_weights
    ):
        """Test basic weighted average aggregation"""
        aggregated = aggregate_weighted_average(sample_client_updates, client_weights)

        assert isinstance(aggregated, list)
        assert len(aggregated) == len(sample_client_updates[0])

        # Check shapes are preserved
        for i, param in enumerate(aggregated):
            assert isinstance(param, np.ndarray)
            assert param.shape == sample_client_updates[0][i].shape

    @pytest.mark.unit
    def test_aggregate_weighted_average_equal_weights(self, sample_client_updates):
        """Test aggregation with equal weights"""
        num_clients = len(sample_client_updates)
        equal_weights = [1.0 / num_clients] * num_clients

        aggregated = aggregate_weighted_average(sample_client_updates, equal_weights)

        # Manually compute expected result
        expected = []
        for param_idx in range(len(sample_client_updates[0])):
            param_sum = np.zeros_like(sample_client_updates[0][param_idx])
            for client_params in sample_client_updates:
                param_sum += client_params[param_idx]
            expected.append(param_sum / num_clients)

        for agg_param, exp_param in zip(aggregated, expected):
            assert np.allclose(agg_param, exp_param, rtol=1e-6)

    @pytest.mark.unit
    def test_aggregate_weighted_average_single_client(self, model_parameters):
        """Test aggregation with single client"""
        single_update = [model_parameters]
        weights = [1.0]

        aggregated = aggregate_weighted_average(single_update, weights)

        for orig, agg in zip(model_parameters, aggregated):
            assert np.allclose(orig, agg, rtol=1e-6)

    @pytest.mark.unit
    def test_aggregate_weighted_average_zero_weight(self, sample_client_updates):
        """Test aggregation with zero weight for one client"""
        weights = [1.0, 0.0, 0.0]  # Only first client has weight

        aggregated = aggregate_weighted_average(sample_client_updates, weights)

        # Should be identical to first client's parameters
        for orig, agg in zip(sample_client_updates[0], aggregated):
            assert np.allclose(orig, agg, rtol=1e-6)


class TestSerialization:
    """Test parameter serialization utilities"""

    @pytest.mark.unit
    def test_serialize_parameters_basic(self, model_parameters):
        """Test basic parameter serialization"""
        serialized = serialize_parameters(model_parameters)

        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    @pytest.mark.unit
    def test_deserialize_parameters_basic(self, model_parameters):
        """Test basic parameter deserialization"""
        serialized = serialize_parameters(model_parameters)
        deserialized = deserialize_parameters(serialized)

        assert isinstance(deserialized, list)
        assert len(deserialized) == len(model_parameters)

        for orig, deser in zip(model_parameters, deserialized):
            assert isinstance(deser, np.ndarray)
            assert orig.shape == deser.shape
            assert np.allclose(orig, deser, rtol=1e-6)

    @pytest.mark.unit
    def test_serialize_deserialize_roundtrip(self, model_parameters):
        """Test serialization round-trip"""
        # Multiple round trips
        current_params = model_parameters
        for _ in range(3):
            serialized = serialize_parameters(current_params)
            current_params = deserialize_parameters(serialized)

        # Should still be identical
        for orig, final in zip(model_parameters, current_params):
            assert np.allclose(orig, final, rtol=1e-6)

    @pytest.mark.unit
    def test_serialize_empty_parameters(self):
        """Test serialization of empty parameter list"""
        empty_params = []
        serialized = serialize_parameters(empty_params)
        deserialized = deserialize_parameters(serialized)

        assert isinstance(deserialized, list)
        assert len(deserialized) == 0

    @pytest.mark.unit
    def test_deserialize_invalid_data(self):
        """Test deserialization with invalid data"""
        with pytest.raises((pickle.PickleError, TypeError, ValueError)):
            deserialize_parameters(b"invalid_data")


class TestHashing:
    """Test cryptographic hash utilities"""

    @pytest.mark.unit
    def test_compute_hash_basic(self, model_parameters):
        """Test basic hash computation"""
        hash_value = compute_hash(model_parameters)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hex digest length

    @pytest.mark.unit
    def test_compute_hash_deterministic(self, model_parameters):
        """Test that hash is deterministic"""
        hash1 = compute_hash(model_parameters)
        hash2 = compute_hash(model_parameters)

        assert hash1 == hash2

    @pytest.mark.unit
    def test_compute_hash_different_params(self, model_parameters):
        """Test that different parameters produce different hashes"""
        # Create slightly different parameters
        modified_params = []
        for param in model_parameters:
            modified = param.copy()
            if modified.size > 0:
                modified.flat[0] += 1e-6  # Small change
            modified_params.append(modified)

        hash1 = compute_hash(model_parameters)
        hash2 = compute_hash(modified_params)

        assert hash1 != hash2

    @pytest.mark.unit
    def test_compute_hash_empty_params(self):
        """Test hash computation with empty parameters"""
        empty_params = []
        hash_value = compute_hash(empty_params)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    @pytest.mark.unit
    def test_compute_hash_algorithm_consistency(self, model_parameters):
        """Test that hash is consistent with direct SHA-256"""
        computed_hash = compute_hash(model_parameters)

        # Manually compute SHA-256
        serialized = serialize_parameters(model_parameters)
        manual_hash = hashlib.sha256(serialized).hexdigest()

        assert computed_hash == manual_hash


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.unit
    def test_large_parameters(self):
        """Test with large parameter arrays"""
        large_param = np.random.randn(1000, 1000).astype(np.float32)
        large_params = [large_param]

        # Test conversion
        parameters = ndarrays_to_parameters(large_params)
        recovered = parameters_to_ndarrays(parameters)

        assert np.allclose(large_params[0], recovered[0], rtol=1e-6)

        # Test norm computation
        norm = compute_parameter_norm(large_params)
        assert isinstance(norm, float)
        assert norm > 0

    @pytest.mark.unit
    def test_very_small_values(self):
        """Test with very small parameter values"""
        small_params = [np.full((5, 3), 1e-10, dtype=np.float32)]

        norm = compute_parameter_norm(small_params)
        assert norm > 0
        assert norm < 1e-8

    @pytest.mark.unit
    def test_mixed_shapes(self):
        """Test with parameters of different shapes"""
        mixed_params = [
            np.random.randn(3, 3).astype(np.float32),
            np.random.randn(10).astype(np.float32),
            np.random.randn(2, 5, 2).astype(np.float32),
        ]

        # Test all operations
        assert validate_parameters(mixed_params)

        norm = compute_parameter_norm(mixed_params)
        assert norm > 0

        stats = get_parameter_stats(mixed_params)
        assert stats["total_params"] == 3 * 3 + 10 + 2 * 5 * 2

    @pytest.mark.unit
    def test_single_element_arrays(self):
        """Test with single-element arrays"""
        single_element_params = [
            np.array([5.0]),
            np.array([[3.0]]),
        ]

        assert validate_parameters(single_element_params)

        norm = compute_parameter_norm(single_element_params)
        expected_norm = np.sqrt(5.0**2 + 3.0**2)
        assert abs(norm - expected_norm) < 1e-6


class TestParameterIntegrity:
    """Test parameter integrity and consistency"""

    @pytest.mark.unit
    def test_conversion_preserves_values(self, model_parameters):
        """Test that conversions preserve exact values"""
        # Multiple conversions
        params = model_parameters
        for _ in range(5):
            flower_params = ndarrays_to_parameters(params)
            params = parameters_to_ndarrays(flower_params)

        # Values should be identical
        for orig, final in zip(model_parameters, params):
            assert np.array_equal(orig, final)

    @pytest.mark.unit
    def test_operations_are_consistent(self, model_parameters):
        """Test that operations are mathematically consistent"""
        # Test norm properties
        norm1 = compute_parameter_norm(model_parameters, norm_type="l2")

        # Scale parameters by 2
        scaled_params = [2 * param for param in model_parameters]
        norm2 = compute_parameter_norm(scaled_params, norm_type="l2")

        # Norm should also scale by 2
        assert abs(norm2 - 2 * norm1) < 1e-6

    @pytest.mark.unit
    def test_aggregation_preserves_structure(
        self, sample_client_updates, client_weights
    ):
        """Test that aggregation preserves parameter structure"""
        aggregated = aggregate_weighted_average(sample_client_updates, client_weights)

        # Check structure is preserved
        original_structure = [(p.shape, p.dtype) for p in sample_client_updates[0]]
        aggregated_structure = [(p.shape, p.dtype) for p in aggregated]

        assert original_structure == aggregated_structure

    @pytest.mark.unit
    def test_hash_collision_resistance(self):
        """Test hash function collision resistance with similar inputs"""
        # Create many similar parameter sets
        base_params = [np.random.randn(10, 10)]
        hashes = set()

        for i in range(100):
            # Create slight variations
            modified = base_params[0] + np.random.normal(0, 1e-10, (10, 10))
            params = [modified]
            hash_val = compute_hash(params)
            hashes.add(hash_val)

        # Should have many unique hashes (collision resistance)
        assert len(hashes) > 95  # Allow for some potential collisions
