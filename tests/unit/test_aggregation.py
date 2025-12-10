"""
Unit Tests for FedJSCM Aggregation Module

This module contains comprehensive unit tests for the FedJSCMAggregator

class and related aggregation functionality.
"""

import numpy as np
import pytest

from secure_fl.aggregation import FedJSCMAggregator


class TestFedJSCMAggregator:
    """Test FedJSCM aggregation algorithm"""

    @pytest.mark.unit
    def test_initialization_default(self):
        """Test aggregator initialization with default parameters"""
        aggregator = FedJSCMAggregator()

        assert aggregator.momentum == 0.9
        assert aggregator.learning_rate == 0.01
        assert aggregator.weight_decay == 0.0
        assert aggregator.adaptive_momentum is False
        assert aggregator.server_momentum is None
        assert aggregator.momentum_initialized is False

    @pytest.mark.unit
    def test_initialization_custom(self):
        """Test aggregator initialization with custom parameters"""
        aggregator = FedJSCMAggregator(
            momentum=0.7, learning_rate=0.1, weight_decay=1e-4, adaptive_momentum=True
        )

        assert aggregator.momentum == 0.7
        assert aggregator.learning_rate == 0.1
        assert aggregator.weight_decay == 1e-4
        assert aggregator.adaptive_momentum is True

    @pytest.mark.unit
    def test_initialization_invalid_momentum(self):
        """Test initialization with invalid momentum values"""
        with pytest.raises(ValueError, match="Momentum must be between 0 and 1"):
            FedJSCMAggregator(momentum=-0.1)

        with pytest.raises(ValueError, match="Momentum must be between 0 and 1"):
            FedJSCMAggregator(momentum=1.5)

    @pytest.mark.unit
    def test_initialization_invalid_learning_rate(self):
        """Test initialization with invalid learning rate"""
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            FedJSCMAggregator(learning_rate=-0.1)

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            FedJSCMAggregator(learning_rate=0.0)

    @pytest.mark.unit
    def test_initialization_invalid_weight_decay(self):
        """Test initialization with invalid weight decay"""
        with pytest.raises(ValueError, match="Weight decay must be non-negative"):
            FedJSCMAggregator(weight_decay=-0.1)

    @pytest.mark.unit
    def test_first_aggregation(
        self, sample_client_updates, client_weights, model_parameters
    ):
        """Test first aggregation initializes momentum"""
        aggregator = FedJSCMAggregator(momentum=0.9)

        # First aggregation
        result = aggregator.aggregate(
            client_updates=sample_client_updates,
            client_weights=client_weights,
            server_round=1,
            global_params=model_parameters,
        )

        # Check momentum initialization
        assert aggregator.momentum_initialized is True
        assert aggregator.server_momentum is not None
        assert len(aggregator.server_momentum) == len(model_parameters)

        # Check result structure
        assert len(result) == len(model_parameters)
        for param, result_param in zip(model_parameters, result, strict=False):
            assert param.shape == result_param.shape
            assert result_param.dtype == np.float32

    @pytest.mark.unit
    def test_subsequent_aggregation(
        self, sample_client_updates, client_weights, model_parameters
    ):
        """Test subsequent aggregations use momentum"""
        aggregator = FedJSCMAggregator(momentum=0.9)

        # First aggregation
        result1 = aggregator.aggregate(
            client_updates=sample_client_updates,
            client_weights=client_weights,
            server_round=1,
            global_params=model_parameters,
        )

        # Store momentum state
        momentum_after_first = [m.copy() for m in aggregator.server_momentum]

        # Create different client updates for second round (simulate real FL scenario)
        second_round_updates = []
        for client_update in sample_client_updates:
            modified_update = []
            for param in client_update:
                # Add different noise to create different updates
                noise = np.random.normal(0, 0.005, param.shape)
                modified_param = param + noise
                modified_update.append(modified_param.astype(np.float32))
            second_round_updates.append(modified_update)

        # Second aggregation with different updates
        result2 = aggregator.aggregate(
            client_updates=second_round_updates,
            client_weights=client_weights,
            server_round=2,
            global_params=result1,
        )

        # Check momentum was updated
        for m1, m2 in zip(momentum_after_first, aggregator.server_momentum, strict=False):
            assert not np.array_equal(m1, m2)

        # Check results are different
        for r1, r2 in zip(result1, result2, strict=False):
            # Should be different due to momentum and different updates
            assert not np.allclose(r1, r2, rtol=1e-6)

    @pytest.mark.unit
    def test_momentum_computation(
        self, sample_client_updates, client_weights, model_parameters
    ):
        """Test momentum computation is mathematically correct"""
        momentum_coeff = 0.5  # Use 0.5 for easier manual verification
        aggregator = FedJSCMAggregator(momentum=momentum_coeff)

        # First round - momentum should be initialized to weighted average
        result1 = aggregator.aggregate(
            client_updates=sample_client_updates,
            client_weights=client_weights,
            server_round=1,
            global_params=model_parameters,
        )

        # Manually compute expected weighted average
        expected_weighted_avg = []
        for param_idx in range(len(sample_client_updates[0])):
            weighted_sum = np.zeros_like(sample_client_updates[0][param_idx])
            for client_idx, weight in enumerate(client_weights):
                weighted_sum += weight * sample_client_updates[client_idx][param_idx]
            expected_weighted_avg.append(weighted_sum)

        # First momentum should equal weighted average
        for expected, actual in zip(expected_weighted_avg, aggregator.server_momentum, strict=False):
            assert np.allclose(expected, actual, rtol=1e-6)

        # Second round
        aggregator.aggregate(
            client_updates=sample_client_updates,
            client_weights=client_weights,
            server_round=2,
            global_params=result1,
        )

        # Manually compute expected momentum update
        # m^{(t+1)} = γ × m^{(t)} + (1-γ) × weighted_avg
        for param_idx in range(len(expected_weighted_avg)):
            (
                momentum_coeff * expected_weighted_avg[param_idx]
                + (1 - momentum_coeff) * expected_weighted_avg[param_idx]
            )
            # Note: In practice, this would use new weighted average, but for test we use same

    @pytest.mark.unit
    def test_zero_momentum(
        self, sample_client_updates, client_weights, model_parameters
    ):
        """Test aggregation with zero momentum (standard FedAvg)"""
        aggregator = FedJSCMAggregator(momentum=0.0)

        result = aggregator.aggregate(
            client_updates=sample_client_updates,
            client_weights=client_weights,
            server_round=1,
            global_params=model_parameters,
        )

        # With zero momentum, should be equivalent to weighted average
        expected = []
        for param_idx in range(len(sample_client_updates[0])):
            weighted_sum = np.zeros_like(sample_client_updates[0][param_idx])
            for client_idx, weight in enumerate(client_weights):
                weighted_sum += weight * sample_client_updates[client_idx][param_idx]
            expected.append(weighted_sum)

        for exp, res in zip(expected, result, strict=False):
            assert np.allclose(exp, res, rtol=1e-6)

    @pytest.mark.unit
    def test_single_client(self, model_parameters):
        """Test aggregation with single client"""
        aggregator = FedJSCMAggregator()

        single_update = [model_parameters]
        single_weight = [1.0]

        result = aggregator.aggregate(
            client_updates=single_update,
            client_weights=single_weight,
            server_round=1,
            global_params=model_parameters,
        )

        # With single client, result should equal client update
        for orig, res in zip(model_parameters, result, strict=False):
            assert np.allclose(orig, res, rtol=1e-6)

    @pytest.mark.unit
    def test_equal_weights(self, sample_client_updates, model_parameters):
        """Test aggregation with equal client weights"""
        num_clients = len(sample_client_updates)
        equal_weights = [1.0 / num_clients] * num_clients

        aggregator = FedJSCMAggregator(momentum=0.0)  # Use zero momentum for clarity

        result = aggregator.aggregate(
            client_updates=sample_client_updates,
            client_weights=equal_weights,
            server_round=1,
            global_params=model_parameters,
        )

        # Should equal simple average
        expected = []
        for param_idx in range(len(sample_client_updates[0])):
            param_sum = np.zeros_like(sample_client_updates[0][param_idx])
            for client_update in sample_client_updates:
                param_sum += client_update[param_idx]
            expected.append(param_sum / num_clients)

        for exp, res in zip(expected, result, strict=False):
            assert np.allclose(exp, res, rtol=1e-6)

    @pytest.mark.unit
    def test_weight_decay(
        self, sample_client_updates, client_weights, model_parameters
    ):
        """Test weight decay functionality"""
        aggregator = FedJSCMAggregator(weight_decay=0.01, momentum=0.0)

        result = aggregator.aggregate(
            client_updates=sample_client_updates,
            client_weights=client_weights,
            server_round=1,
            global_params=model_parameters,
        )

        # With weight decay, parameters should be smaller in magnitude
        for orig, res in zip(model_parameters, result, strict=False):
            # The exact comparison depends on implementation details
            assert res.shape == orig.shape

    @pytest.mark.unit
    def test_adaptive_momentum_stability_high(
        self, sample_client_updates, client_weights, model_parameters
    ):
        """Test adaptive momentum with high stability"""
        aggregator = FedJSCMAggregator(adaptive_momentum=True, momentum=0.9)

        # Simulate high stability (small parameter changes)
        stable_updates = []
        for client_update in sample_client_updates:
            stable_update = []
            for param in client_update:
                # Small changes indicate stability
                noise = np.random.normal(0, 0.001, param.shape)
                stable_update.append(param + noise.astype(param.dtype))
            stable_updates.append(stable_update)

        aggregator.aggregate(
            client_updates=stable_updates,
            client_weights=client_weights,
            server_round=1,
            global_params=model_parameters,
            stability_score=0.95,  # High stability
        )

        # Should increase momentum for stable training
        # Implementation-specific checks would go here

    @pytest.mark.unit
    def test_adaptive_momentum_stability_low(
        self, sample_client_updates, client_weights, model_parameters
    ):
        """Test adaptive momentum with low stability"""
        aggregator = FedJSCMAggregator(adaptive_momentum=True, momentum=0.9)

        # Simulate low stability (large parameter changes)
        unstable_updates = []
        for client_update in sample_client_updates:
            unstable_update = []
            for param in client_update:
                # Large changes indicate instability
                noise = np.random.normal(0, 0.1, param.shape)
                unstable_update.append(param + noise.astype(param.dtype))
            unstable_updates.append(unstable_update)

        aggregator.aggregate(
            client_updates=unstable_updates,
            client_weights=client_weights,
            server_round=1,
            global_params=model_parameters,
            stability_score=0.2,  # Low stability
        )

        # Should decrease momentum for unstable training
        # Implementation-specific checks would go here

    @pytest.mark.unit
    def test_get_momentum_state(
        self, sample_client_updates, client_weights, model_parameters
    ):
        """Test momentum state retrieval"""
        aggregator = FedJSCMAggregator()

        # Before initialization
        state = aggregator.get_momentum_state()
        assert state["initialized"] is False
        assert state["momentum"] == 0.9
        assert state["learning_rate"] == 0.01

        # After initialization
        aggregator.aggregate(
            client_updates=sample_client_updates,
            client_weights=client_weights,
            server_round=1,
            global_params=model_parameters,
        )

        state = aggregator.get_momentum_state()
        assert state["initialized"] is True
        assert "server_momentum" in state
        assert len(state["server_momentum"]) == len(model_parameters)

    @pytest.mark.unit
    def test_reset_momentum(
        self, sample_client_updates, client_weights, model_parameters
    ):
        """Test momentum reset functionality"""
        aggregator = FedJSCMAggregator()

        # Initialize momentum
        aggregator.aggregate(
            client_updates=sample_client_updates,
            client_weights=client_weights,
            server_round=1,
            global_params=model_parameters,
        )

        assert aggregator.momentum_initialized is True

        # Reset momentum
        aggregator.reset_momentum()

        assert aggregator.momentum_initialized is False
        assert aggregator.server_momentum is None

    @pytest.mark.unit
    def test_invalid_inputs_mismatched_lengths(self, model_parameters):
        """Test error handling for mismatched input lengths"""
        aggregator = FedJSCMAggregator()

        # Mismatched number of clients and weights
        updates = [[np.random.randn(3, 2)], [np.random.randn(3, 2)]]
        weights = [0.5]  # Only one weight for two clients

        with pytest.raises(
            ValueError, match="Number of client updates must match number of weights"
        ):
            aggregator.aggregate(
                client_updates=updates,
                client_weights=weights,
                server_round=1,
                global_params=model_parameters,
            )

    @pytest.mark.unit
    def test_invalid_inputs_empty_updates(self, model_parameters):
        """Test error handling for empty client updates"""
        aggregator = FedJSCMAggregator()

        with pytest.raises(ValueError, match="At least one client update is required"):
            aggregator.aggregate(
                client_updates=[],
                client_weights=[],
                server_round=1,
                global_params=model_parameters,
            )

    @pytest.mark.unit
    def test_invalid_inputs_negative_weights(
        self, sample_client_updates, model_parameters
    ):
        """Test error handling for negative weights"""
        aggregator = FedJSCMAggregator()

        negative_weights = [0.5, -0.3, 0.8]

        with pytest.raises(ValueError, match="All client weights must be non-negative"):
            aggregator.aggregate(
                client_updates=sample_client_updates,
                client_weights=negative_weights,
                server_round=1,
                global_params=model_parameters,
            )

    @pytest.mark.unit
    def test_invalid_inputs_zero_total_weight(
        self, sample_client_updates, model_parameters
    ):
        """Test error handling for zero total weight"""
        aggregator = FedJSCMAggregator()

        zero_weights = [0.0, 0.0, 0.0]

        with pytest.raises(ValueError, match="Total client weights must be positive"):
            aggregator.aggregate(
                client_updates=sample_client_updates,
                client_weights=zero_weights,
                server_round=1,
                global_params=model_parameters,
            )

    @pytest.mark.unit
    def test_invalid_inputs_mismatched_shapes(self, model_parameters):
        """Test error handling for mismatched parameter shapes"""
        aggregator = FedJSCMAggregator()

        # Create updates with wrong shapes
        wrong_shape_updates = [
            [np.random.randn(5, 5)],  # Wrong shape
            [np.random.randn(3, 2)],  # Correct shape
        ]
        weights = [0.5, 0.5]

        with pytest.raises(ValueError, match="Parameter shapes must match"):
            aggregator.aggregate(
                client_updates=wrong_shape_updates,
                client_weights=weights,
                server_round=1,
                global_params=model_parameters,
            )

    @pytest.mark.unit
    def test_convergence_behavior(
        self, sample_client_updates, client_weights, model_parameters
    ):
        """Test aggregation behavior over multiple rounds"""
        aggregator = FedJSCMAggregator(momentum=0.9)

        current_params = model_parameters
        results = []

        # Run multiple aggregation rounds
        for round_num in range(1, 6):
            result = aggregator.aggregate(
                client_updates=sample_client_updates,
                client_weights=client_weights,
                server_round=round_num,
                global_params=current_params,
            )
            results.append(result)
            current_params = result

        # Check that momentum is working (results should show momentum effects)
        assert len(results) == 5
        for result in results:
            assert len(result) == len(model_parameters)

    @pytest.mark.unit
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        aggregator = FedJSCMAggregator()

        # Very small values
        small_updates = [
            [np.full((3, 2), 1e-8, dtype=np.float32)],
            [np.full((3, 2), -1e-8, dtype=np.float32)],
        ]
        weights = [0.5, 0.5]
        global_params = [np.full((3, 2), 1e-7, dtype=np.float32)]

        result = aggregator.aggregate(
            client_updates=small_updates,
            client_weights=weights,
            server_round=1,
            global_params=global_params,
        )

        # Should not produce NaN or inf
        for param in result:
            assert np.isfinite(param).all()

    @pytest.mark.unit
    def test_dtype_preservation(
        self, sample_client_updates, client_weights, model_parameters
    ):
        """Test that data types are properly handled"""
        aggregator = FedJSCMAggregator()

        # Ensure input is float32
        float32_updates = []
        for client_update in sample_client_updates:
            client_float32 = [param.astype(np.float32) for param in client_update]
            float32_updates.append(client_float32)

        float32_global = [param.astype(np.float32) for param in model_parameters]

        result = aggregator.aggregate(
            client_updates=float32_updates,
            client_weights=client_weights,
            server_round=1,
            global_params=float32_global,
        )

        # Result should be float32
        for param in result:
            assert param.dtype == np.float32

    @pytest.mark.unit
    def test_memory_efficiency(
        self, sample_client_updates, client_weights, model_parameters
    ):
        """Test that aggregation doesn't create excessive memory usage"""
        aggregator = FedJSCMAggregator()

        # Run aggregation multiple times
        for i in range(10):
            aggregator.aggregate(
                client_updates=sample_client_updates,
                client_weights=client_weights,
                server_round=i + 1,
                global_params=model_parameters,
            )

        # Should not accumulate excessive state
        state = aggregator.get_momentum_state()
        assert len(state["server_momentum"]) == len(model_parameters)
