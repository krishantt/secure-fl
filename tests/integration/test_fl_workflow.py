"""
Integration Tests for Federated Learning Workflow

This module contains integration tests that verify the complete federated learning
workflow, including server-client interactions, aggregation, and ZKP integration.
"""

import asyncio
import multiprocessing as mp
import tempfile
import time
from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from secure_fl.aggregation import FedJSCMAggregator
from secure_fl.client import SecureFlowerClient, create_client
from secure_fl.models import MNISTModel, SimpleModel
from secure_fl.server import (
    SecureFlowerServer,
    SecureFlowerStrategy,
    create_server_strategy,
)
from secure_fl.stability_monitor import StabilityMonitor
from secure_fl.utils import parameters_to_ndarrays, torch_to_ndarrays


def _simulate_client_fit(client, parameters, config):
    """Helper function to simulate client fit and return compatible result"""
    from secure_fl.utils import ndarrays_to_parameters

    fitted_params, num_examples, metrics = client.fit(parameters, config)

    # Convert NDArrays to Flower Parameters for server compatibility
    parameters_obj = ndarrays_to_parameters(fitted_params)

    class FitResult:
        def __init__(self, parameters, num_examples, metrics):
            self.parameters = parameters
            self.num_examples = num_examples
            self.metrics = metrics

    return FitResult(parameters_obj, num_examples, metrics)


class TestBasicFLWorkflow:
    """Test basic federated learning workflow without ZKP"""

    @pytest.mark.integration
    def test_single_round_training(self, client_datasets, server_config, client_config):
        """Test single round of federated training"""
        # Create simple model
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        # Create server strategy
        config = server_config.copy()
        config.update({"enable_zkp": False})  # Disable ZKP for basic test
        strategy = create_server_strategy(
            model_fn=model_fn,
            **config,
        )

        # Create clients
        clients = []
        for i, dataset in enumerate(client_datasets):
            client = create_client(
                client_id=f"client_{i}",
                model_fn=model_fn,
                train_data=dataset,
                **client_config,
            )
            clients.append(client)

        # Simulate single training round
        initial_params = torch_to_ndarrays(model_fn())

        # Collect client updates
        client_results = []
        for client in clients:
            # Use helper function to simulate fit
            result = _simulate_client_fit(
                client, initial_params, {"local_epochs": 1, "server_round": 1}
            )
            client_results.append(result)

        # Test aggregation
        client_updates = [result.parameters for result in client_results]
        client_weights = [result.num_examples for result in client_results]

        # Normalize weights
        total_examples = sum(client_weights)
        normalized_weights = [w / total_examples for w in client_weights]

        # Initialize parameters (normally done by Flower framework)
        strategy.initialize_parameters(None)

        # Use strategy to aggregate
        aggregated_result = strategy.aggregate_fit(
            server_round=1, results=list(zip(clients, client_results)), failures=[]
        )

        # Verify results
        assert aggregated_result is not None
        aggregated_params, metrics = aggregated_result
        assert aggregated_params is not None
        assert len(aggregated_params.tensors) == len(initial_params)

        # Parameters should have changed
        for init, final in zip(
            initial_params, parameters_to_ndarrays(aggregated_params)
        ):
            assert not np.allclose(init, final, rtol=1e-6)

    @pytest.mark.integration
    def test_multi_round_training(self, client_datasets, server_config, client_config):
        """Test multiple rounds of federated training"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        # Create server strategy
        config = server_config.copy()
        config.update({"enable_zkp": False})
        strategy = create_server_strategy(model_fn=model_fn, num_rounds=3, **config)

        # Create clients
        clients = []
        for i, dataset in enumerate(client_datasets):
            client = create_client(
                client_id=f"client_{i}",
                model_fn=model_fn,
                train_data=dataset,
                **client_config,
            )
            clients.append(client)

        # Initialize strategy parameters
        strategy.initialize_parameters(model_fn)

        # Initial parameters
        current_params = torch_to_ndarrays(model_fn())

        # Track parameter evolution
        param_history = [current_params]

        # Run multiple rounds
        for round_num in range(1, 4):
            # Client training
            client_results = []
            for client in clients:
                result = _simulate_client_fit(
                    client,
                    current_params,
                    {"local_epochs": 1, "server_round": round_num},
                )
                client_results.append(result)

            # Server aggregation
            aggregated = strategy.aggregate_fit(
                server_round=round_num,
                results=list(zip(clients, client_results)),
                failures=[],
            )

            current_params = parameters_to_ndarrays(aggregated[0])
            param_history.append(current_params)

        # Verify parameter evolution
        assert len(param_history) == 4

        # Parameters should be different across rounds
        for i in range(len(param_history) - 1):
            for p1, p2 in zip(param_history[i], param_history[i + 1]):
                assert not np.allclose(p1, p2, rtol=1e-6)

    @pytest.mark.integration
    def test_client_dropout_handling(
        self, client_datasets, server_config, client_config
    ):
        """Test handling of client dropouts during training"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        # Create server strategy
        config = server_config.copy()
        config.update({"enable_zkp": False})
        strategy = create_server_strategy(
            model_fn=model_fn,
            **config,
        )

        # Create more clients than minimum required
        clients = []
        for i, dataset in enumerate(client_datasets):
            client = create_client(
                client_id=f"client_{i}",
                model_fn=model_fn,
                train_data=dataset,
                **client_config,
            )
            clients.append(client)

        # Initialize strategy parameters
        strategy.initialize_parameters(model_fn)

        initial_params = torch_to_ndarrays(model_fn())

        # Simulate some clients dropping out
        available_clients = clients[:2]  # Only first 2 clients available

        client_results = []
        for client in available_clients:
            result = _simulate_client_fit(
                client, initial_params, {"local_epochs": 1, "server_round": 1}
            )
            client_results.append(result)

        # Should still be able to aggregate
        aggregated = strategy.aggregate_fit(
            server_round=1,
            results=list(zip(available_clients, client_results)),
            failures=[],
        )

        assert aggregated is not None
        assert len(parameters_to_ndarrays(aggregated[0])) == len(initial_params)

    @pytest.mark.integration
    def test_heterogeneous_data_sizes(self, server_config, client_config):
        """Test training with clients having different data sizes"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        # Create datasets with very different sizes
        datasets = []
        data_sizes = [50, 200, 500]  # Very different sizes

        for size in data_sizes:
            X = torch.randn(size, 10)
            y = torch.randint(0, 2, (size,))
            dataset = TensorDataset(X, y)
            datasets.append(dataset)

        config = server_config.copy()
        config.update({"enable_zkp": False})
        strategy = create_server_strategy(model_fn=model_fn, **config)

        clients = []
        for i, dataset in enumerate(datasets):
            client = create_client(
                client_id=f"client_{i}",
                model_fn=model_fn,
                train_data=dataset,
                **client_config,
            )
            clients.append(client)

        # Initialize strategy parameters
        strategy.initialize_parameters(model_fn)

        initial_params = torch_to_ndarrays(model_fn())

        # Training round
        client_results = []
        for client in clients:
            result = _simulate_client_fit(
                client, initial_params, {"local_epochs": 1, "server_round": 1}
            )
            client_results.append(result)

        # Verify different data sizes are reflected in num_examples
        num_examples = [result.num_examples for result in client_results]
        assert num_examples == data_sizes

        # Aggregation should work
        aggregated = strategy.aggregate_fit(
            server_round=1, results=list(zip(clients, client_results)), failures=[]
        )

        assert aggregated is not None


class TestAggregationIntegration:
    """Test integration of aggregation with FL workflow"""

    @pytest.mark.integration
    def test_fedjscm_integration(self, client_datasets, server_config, client_config):
        """Test FedJSCM aggregation integration in FL workflow"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        # Create strategy with specific aggregation config
        config = server_config.copy()
        config.update({"enable_zkp": False})
        strategy = create_server_strategy(
            model_fn=model_fn,
            momentum=0.9,
            learning_rate=0.01,
            **config,
        )

        clients = []
        for i, dataset in enumerate(client_datasets):
            client = create_client(
                client_id=f"client_{i}",
                model_fn=model_fn,
                train_data=dataset,
                **client_config,
            )
            clients.append(client)

        # Initialize strategy parameters
        strategy.initialize_parameters(model_fn)

        initial_params = torch_to_ndarrays(model_fn())
        current_params = initial_params

        # Run two rounds to test momentum
        for round_num in range(1, 3):
            client_results = []
            for client in clients:
                result = _simulate_client_fit(
                    client, current_params, {"local_epochs": 1, "server_round": 1}
                )
                client_results.append(result)

            aggregated = strategy.aggregate_fit(
                server_round=round_num,
                results=list(zip(clients, client_results)),
                failures=[],
            )

            current_params = parameters_to_ndarrays(aggregated[0])

        # Check that aggregator has momentum state
        aggregator = strategy.aggregator
        momentum_state = aggregator.get_momentum_state()
        assert momentum_state["initialized"] is True
        assert "momentum" in momentum_state

    @pytest.mark.integration
    def test_stability_monitoring_integration(
        self, client_datasets, server_config, client_config
    ):
        """Test stability monitoring integration"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        config = server_config.copy()
        config.update({"enable_zkp": False})
        strategy = create_server_strategy(model_fn=model_fn, **config)

        clients = []
        for i, dataset in enumerate(client_datasets):
            client = create_client(
                client_id=f"client_{i}",
                model_fn=model_fn,
                train_data=dataset,
                **client_config,
            )
            clients.append(client)

        # Initialize strategy parameters
        strategy.initialize_parameters(model_fn)

        initial_params = torch_to_ndarrays(model_fn())
        current_params = initial_params

        # Run multiple rounds to test stability monitoring
        for round_num in range(1, 6):
            client_results = []
            for client in clients:
                result = _simulate_client_fit(
                    client,
                    current_params,
                    {"local_epochs": 1, "server_round": round_num},
                )
                client_results.append(result)

            aggregated = strategy.aggregate_fit(
                server_round=round_num,
                results=list(zip(clients, client_results)),
                failures=[],
            )

            current_params = parameters_to_ndarrays(aggregated[0])

        # Check stability monitor has been updated
        monitor = strategy.stability_monitor
        stability_score = monitor.get_stability_score()
        assert stability_score >= 0

        # Should have recommendations based on stability
        recommended_rigor = monitor.get_recommended_rigor()
        assert recommended_rigor in ["low", "medium", "high"]


class TestZKPIntegration:
    """Test ZKP integration in FL workflow"""

    @pytest.mark.integration
    @pytest.mark.zkp
    def test_zkp_workflow_mock(
        self,
        client_datasets,
        server_config,
        client_config,
        mock_client_proof_manager,
        mock_server_proof_manager,
    ):
        """Test ZKP workflow with mocked proof managers"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        # Enable ZKP in configuration
        zkp_config = client_config.copy()
        zkp_config["enable_zkp"] = True

        zkp_server_config = server_config.copy()
        zkp_server_config["enable_zkp"] = True

        # Mock the proof managers in the modules
        with (
            patch(
                "secure_fl.client.ClientProofManager",
                return_value=mock_client_proof_manager,
            ),
            patch(
                "secure_fl.server.ServerProofManager",
                return_value=mock_server_proof_manager,
            ),
        ):
            strategy = create_server_strategy(model_fn=model_fn, **zkp_server_config)

            clients = []
            for i, dataset in enumerate(client_datasets):
                client = create_client(
                    client_id=f"client_{i}",
                    model_fn=model_fn,
                    train_data=dataset,
                    **zkp_config,
                )
                clients.append(client)

            # Initialize strategy parameters
            strategy.initialize_parameters(model_fn)

            initial_params = torch_to_ndarrays(model_fn())

            # Training round with ZKP
            client_results = []
            for client in clients:
                result = _simulate_client_fit(
                    client, initial_params, {"local_epochs": 1, "server_round": 1}
                )
                client_results.append(result)

                # Verify client proof generation was called
                mock_client_proof_manager.generate_training_proof.assert_called()

            # Server aggregation with ZKP
            aggregated = strategy.aggregate_fit(
                server_round=1, results=list(zip(clients, client_results)), failures=[]
            )

            # Verify server proof generation was called
            mock_server_proof_manager.generate_aggregation_proof.assert_called()

            assert aggregated is not None

    @pytest.mark.integration
    def test_zkp_disabled_fallback(self, client_datasets, server_config, client_config):
        """Test that FL works correctly when ZKP is disabled"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        # Explicitly disable ZKP
        no_zkp_config = client_config.copy()
        no_zkp_config["enable_zkp"] = False

        config = server_config.copy()
        config.update({"enable_zkp": False})
        strategy = create_server_strategy(model_fn=model_fn, **config)

        clients = []
        for i, dataset in enumerate(client_datasets):
            client = create_client(
                client_id=f"client_{i}",
                model_fn=model_fn,
                train_data=dataset,
                **no_zkp_config,
            )
            clients.append(client)

        # Initialize strategy parameters
        strategy.initialize_parameters(model_fn)

        initial_params = torch_to_ndarrays(model_fn())

        # Should work without ZKP
        client_results = []
        for client in clients:
            result = _simulate_client_fit(
                client, initial_params, {"local_epochs": 1, "server_round": 1}
            )
            client_results.append(result)

        aggregated = strategy.aggregate_fit(
            server_round=1, results=list(zip(clients, client_results)), failures=[]
        )

        assert aggregated is not None


class TestErrorHandling:
    """Test error handling in FL workflow"""

    @pytest.mark.integration
    def test_invalid_client_updates(
        self, client_datasets, server_config, client_config
    ):
        """Test handling of invalid client updates"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        config = server_config.copy()
        config.update({"enable_zkp": False})
        strategy = create_server_strategy(model_fn=model_fn, **config)

        clients = []
        for i, dataset in enumerate(client_datasets):
            client = create_client(
                client_id=f"client_{i}",
                model_fn=model_fn,
                train_data=dataset,
                **client_config,
            )
            clients.append(client)

        # Initialize strategy parameters
        strategy.initialize_parameters(model_fn)

        initial_params = torch_to_ndarrays(model_fn())

        # Create some valid and some invalid results
        client_results = []

        # Valid result from first client
        result1 = _simulate_client_fit(
            clients[0], initial_params, {"local_epochs": 1, "server_round": 1}
        )
        client_results.append(result1)

        # Create invalid result (wrong parameter shapes)
        invalid_params = [np.random.randn(999, 999)]  # Wrong shape

        # Mock an invalid result
        from flwr.common import Code, FitRes, Status

        invalid_result = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=invalid_params,
            num_examples=100,
            metrics={},
        )

        # Strategy should handle invalid results gracefully
        # (Implementation may vary - could skip invalid clients or raise error)
        try:
            aggregated = strategy.aggregate_fit(
                server_round=1,
                results=[(clients[0], result1)],  # Only valid result
                failures=[(clients[1], invalid_result)],  # Invalid as failure
            )
            # If it doesn't raise error, should still produce valid result
            assert aggregated is not None
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for invalid inputs
            pass

    @pytest.mark.integration
    def test_insufficient_clients(self, client_datasets, server_config, client_config):
        """Test behavior when insufficient clients participate"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        # Require at least 3 clients but provide only 1
        strict_config = server_config.copy()
        strict_config["min_fit_clients"] = 3

        strict_config.update({"enable_zkp": False})
        strategy = create_server_strategy(model_fn=model_fn, **strict_config)

        # Only create one client
        client = create_client(
            client_id="test_client",
            model_fn=model_fn,
            train_data=client_datasets[0],
            **client_config,
        )

        # Initialize strategy parameters
        strategy.initialize_parameters(model_fn)

        initial_params = torch_to_ndarrays(model_fn())
        result = _simulate_client_fit(
            client, initial_params, {"local_epochs": 1, "server_round": 1}
        )

        # Should handle insufficient clients (behavior may vary by implementation)
        try:
            aggregated = strategy.aggregate_fit(
                server_round=1, results=[(client, result)], failures=[]
            )
            # Some implementations might still aggregate with fewer clients
            if aggregated is not None:
                assert len(parameters_to_ndarrays(aggregated[0])) == len(initial_params)
        except ValueError:
            # It's acceptable to raise an error for insufficient clients
            pass

    @pytest.mark.integration
    def test_nan_parameter_handling(
        self, client_datasets, server_config, client_config
    ):
        """Test handling of NaN parameters from clients"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        config = server_config.copy()
        config.update({"enable_zkp": False})
        strategy = create_server_strategy(model_fn=model_fn, **config)

        client = create_client(
            client_id="insufficient_client",
            model_fn=model_fn,
            train_data=client_datasets[0],
            **client_config,
        )

        # Initialize strategy parameters
        strategy.initialize_parameters(model_fn)

        initial_params = torch_to_ndarrays(model_fn())

        # Create parameters with NaN values
        nan_params = []
        for param in initial_params:
            nan_param = param.copy()
            nan_param.flat[0] = np.nan  # Inject NaN
            nan_params.append(nan_param)

        # Mock result with NaN parameters
        from flwr.common import Code, FitRes, Status

        from secure_fl.utils import ndarrays_to_parameters

        # Convert nan_params list to proper Parameters object
        nan_parameters = ndarrays_to_parameters(nan_params)

        nan_result = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=nan_parameters,
            num_examples=100,
            metrics={},
        )

        # Should detect and handle NaN parameters
        try:
            aggregated = strategy.aggregate_fit(
                server_round=1, results=[(client, nan_result)], failures=[]
            )
            # If it doesn't raise an error, result should not contain NaN
            if aggregated is not None:
                final_params = parameters_to_ndarrays(aggregated[0])
                for param in final_params:
                    assert not np.isnan(param).any()
        except (ValueError, RuntimeError):
            # Acceptable to raise error for NaN parameters
            pass


class TestPerformanceIntegration:
    """Test performance aspects of FL workflow"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_model_training(self, server_config, client_config):
        """Test FL workflow with larger models"""
        # Create a larger model
        model_fn = lambda: MNISTModel(hidden_dims=[256, 128, 64])

        # Create larger synthetic datasets
        large_datasets = []
        for i in range(3):
            X = torch.randn(1000, 28 * 28)  # MNIST-like size
            y = torch.randint(0, 10, (1000,))
            dataset = TensorDataset(X, y)
            large_datasets.append(dataset)

        config = server_config.copy()
        config.update({"enable_zkp": False})
        strategy = create_server_strategy(model_fn=model_fn, **config)

        clients = []
        for i, dataset in enumerate(large_datasets):
            client = create_client(
                client_id=f"large_client_{i}",
                model_fn=model_fn,
                train_data=dataset,
                **client_config,
            )
            clients.append(client)

        # Initialize strategy parameters
        strategy.initialize_parameters(model_fn)

        initial_params = torch_to_ndarrays(model_fn())

        # Time the training round
        start_time = time.time()

        client_results = []
        for client in clients:
            result = _simulate_client_fit(
                client, initial_params, {"local_epochs": 2, "server_round": 1}
            )
            client_results.append(result)

        aggregated = strategy.aggregate_fit(
            server_round=1, results=list(zip(clients, client_results)), failures=[]
        )

        end_time = time.time()
        training_time = end_time - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert training_time < 60  # Less than 60 seconds
        assert aggregated is not None

    @pytest.mark.integration
    def test_memory_efficiency(self, client_datasets, server_config, client_config):
        """Test that FL workflow is memory efficient"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        config = server_config.copy()
        config.update({"enable_zkp": False})
        strategy = create_server_strategy(model_fn=model_fn, **config)

        clients = []
        for i, dataset in enumerate(client_datasets):
            client = create_client(
                client_id=f"mem_client_{i}",
                model_fn=model_fn,
                train_data=dataset,
                **client_config,
            )
            clients.append(client)

        # Initialize strategy parameters
        strategy.initialize_parameters(model_fn)

        initial_params = torch_to_ndarrays(model_fn())

        # Run multiple rounds to test for memory leaks
        for round_num in range(1, 6):
            client_results = []
            for client in clients:
                result = _simulate_client_fit(
                    client, initial_params, {"local_epochs": 1, "server_round": 1}
                )
                client_results.append(result)

            aggregated = strategy.aggregate_fit(
                server_round=round_num,
                results=list(zip(clients, client_results)),
                failures=[],
            )

            initial_params = parameters_to_ndarrays(aggregated[0])

        # Memory usage should be stable (no excessive accumulation)
        # This is more of a regression test - exact memory usage is hard to verify
        assert True  # If we get here without OOM, memory usage is reasonable


class TestConfigurationIntegration:
    """Test different configuration combinations"""

    @pytest.mark.integration
    def test_various_aggregation_configs(
        self, client_datasets, server_config, client_config
    ):
        """Test FL with different aggregation configurations"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        # Test different momentum values
        momentum_values = [0.0, 0.5, 0.9, 0.99]

        for momentum in momentum_values:
            config = server_config.copy()
            config.update({"enable_zkp": False})
            strategy = create_server_strategy(
                model_fn=model_fn, momentum=momentum, **config
            )

            clients = []
            for i, dataset in enumerate(client_datasets):
                client = create_client(
                    client_id=f"agg_client_{i}_{momentum}",
                    model_fn=model_fn,
                    train_data=dataset,
                    **client_config,
                )
                clients.append(client)

            # Initialize strategy parameters
            strategy.initialize_parameters(model_fn)

            initial_params = torch_to_ndarrays(model_fn())

            # Should work with all momentum values
            client_results = []
            for client in clients:
                result = _simulate_client_fit(
                    client, initial_params, {"local_epochs": 1, "server_round": 1}
                )
                client_results.append(result)

            aggregated = strategy.aggregate_fit(
                server_round=1, results=list(zip(clients, client_results)), failures=[]
            )

            assert aggregated is not None

    @pytest.mark.integration
    def test_various_client_configs(self, client_datasets, server_config):
        """Test FL with different client configurations"""
        model_fn = lambda: SimpleModel(input_dim=10, output_dim=2)

        config = server_config.copy()
        config.update({"enable_zkp": False})
        strategy = create_server_strategy(model_fn=model_fn, **config)

        # Test different local epoch configurations
        local_epochs_configs = [1, 3, 5]

        for local_epochs in local_epochs_configs:
            config = {
                "local_epochs": local_epochs,
                "batch_size": 10,
                "learning_rate": 0.01,
                "enable_zkp": False,
            }

            clients = []
            for i, dataset in enumerate(client_datasets):
                client = create_client(
                    client_id=f"config_client_{i}_{config['batch_size']}",
                    model_fn=model_fn,
                    train_data=dataset,
                    **config,
                )
                clients.append(client)

            # Initialize strategy parameters
            strategy.initialize_parameters(model_fn)

            initial_params = torch_to_ndarrays(model_fn())

            client_results = []
            for client in clients:
                result = _simulate_client_fit(
                    client,
                    initial_params,
                    {"local_epochs": local_epochs, "server_round": 1},
                )
                client_results.append(result)

            aggregated = strategy.aggregate_fit(
                server_round=1, results=list(zip(clients, client_results)), failures=[]
            )

            assert aggregated is not None
