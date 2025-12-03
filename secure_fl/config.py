"""
Production-Grade Configuration Management for Secure FL

This module provides comprehensive configuration management with environment
variable support, validation, and type safety for production deployments.

Features:
- Environment-based configuration
- Type validation with Pydantic
- Hierarchical configuration loading
- Secret management integration
- Configuration caching and hot-reloading
- Environment-specific defaults

Usage:
    from secure_fl.config import get_config, ConfigManager

    # Get current configuration
    config = get_config()

    # Access configuration values
    server_port = config.server.port
    zkp_enabled = config.zkp.enable_zkp

    # Create custom configuration
    config_manager = ConfigManager(env="production")
    prod_config = config_manager.load_config()
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

try:
    from pydantic import BaseModel
    from pydantic import BaseSettings as PydanticBaseSettings
    from pydantic import Field, root_validator, validator

    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for environments without pydantic
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    PydanticBaseSettings = object

    def Field(*args, **kwargs):
        return None

    def validator(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def root_validator(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Supported deployment environments"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Supported logging levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ServerConfig:
    """Server configuration settings"""

    host: str = "localhost"
    port: int = 8080
    num_rounds: int = 10
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    timeout: int = 300
    max_workers: int = 10
    strategy: str = "FedJSCM"

    # SSL/TLS configuration
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

    # Authentication
    auth_enabled: bool = False
    jwt_secret: Optional[str] = None
    jwt_expiration: int = 3600

    # Rate limiting
    rate_limit_enabled: bool = True
    max_requests_per_minute: int = 100


@dataclass
class ClientConfig:
    """Client configuration settings"""

    client_id: Optional[str] = None
    server_address: str = "localhost:8080"
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    optimizer: str = "sgd"
    timeout: int = 300
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Data configuration
    data_path: Optional[str] = None
    dataset_name: str = "synthetic"
    validation_split: float = 0.1

    # Privacy settings
    differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5


@dataclass
class ZKPConfig:
    """Zero-Knowledge Proof configuration"""

    enable_zkp: bool = True
    proof_rigor: str = "medium"  # low, medium, high
    blockchain_verification: bool = False
    quantize_weights: bool = True
    quantization_bits: int = 8
    proof_timeout: int = 120

    # Client-side zk-STARK configuration
    cairo_path: Optional[str] = None
    cairo_compile_timeout: int = 60
    max_trace_length: int = 1024

    # Server-side zk-SNARK configuration
    circom_path: Optional[str] = None
    snarkjs_path: Optional[str] = None
    circuit_size: int = 1000
    trusted_setup_path: Optional[str] = None

    # Blockchain configuration
    blockchain_network: str = "ethereum"
    contract_address: Optional[str] = None
    private_key: Optional[str] = None
    gas_limit: int = 2000000


@dataclass
class AggregationConfig:
    """Aggregation algorithm configuration"""

    momentum: float = 0.9
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    adaptive_momentum: bool = False
    min_momentum: float = 0.1
    max_momentum: float = 0.99

    # Aggregation strategy
    strategy: str = "fedjscm"  # fedjscm, fedavg, fedprox, scaffold

    # FedProx specific
    proximal_mu: float = 0.1

    # SCAFFOLD specific
    server_control_variates: bool = False


@dataclass
class StabilityConfig:
    """Stability monitoring configuration"""

    window_size: int = 10
    stability_threshold_high: float = 0.9
    stability_threshold_medium: float = 0.7
    convergence_patience: int = 5
    min_rounds_for_adjustment: int = 3

    # Adaptive rigor parameters
    rigor_adjustment_factor: float = 0.1
    min_rigor_rounds: int = 2


@dataclass
class SecurityConfig:
    """Security and privacy configuration"""

    enable_tls: bool = False
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None

    # Authentication
    enable_auth: bool = False
    auth_method: str = "jwt"  # jwt, oauth, api_key
    jwt_secret_key: Optional[str] = None

    # Input validation
    max_model_size_mb: float = 100.0
    max_batch_size: int = 1000
    allowed_file_types: List[str] = field(
        default_factory=lambda: [".pt", ".pth", ".pkl"]
    )

    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60


@dataclass
class DatabaseConfig:
    """Database configuration for metrics and logs"""

    enabled: bool = False
    database_url: Optional[str] = None
    driver: str = "sqlite"  # sqlite, postgresql, mysql

    # Connection settings
    max_connections: int = 10
    connection_timeout: int = 30

    # Retention policies
    metrics_retention_days: int = 30
    logs_retention_days: int = 7


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""

    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"

    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"  # json, text
    log_file: Optional[str] = None
    log_rotation: bool = True
    log_max_size: str = "10MB"
    log_backup_count: int = 5

    # Health checks
    health_check_enabled: bool = True
    health_check_interval: int = 30
    health_check_timeout: int = 10

    # Distributed tracing
    tracing_enabled: bool = False
    tracing_service_name: str = "secure-fl"
    jaeger_endpoint: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Experiment and research configuration"""

    experiment_name: Optional[str] = None
    results_dir: str = "results"
    save_checkpoints: bool = True
    checkpoint_interval: int = 5

    # Reproducibility
    random_seed: Optional[int] = 42
    deterministic: bool = False

    # Benchmarking
    benchmark_mode: bool = False
    profile_memory: bool = False
    profile_cpu: bool = False


@dataclass
class SecureFlConfig:
    """Main configuration class containing all settings"""

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # Component configurations
    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    zkp: ZKPConfig = field(default_factory=ZKPConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def validate(self) -> None:
        """Validate configuration consistency"""
        # Check server configuration
        if self.server.port < 1 or self.server.port > 65535:
            raise ValueError(f"Invalid server port: {self.server.port}")

        if self.server.num_rounds < 1:
            raise ValueError(
                f"Number of rounds must be positive: {self.server.num_rounds}"
            )

        if self.server.min_fit_clients > self.server.min_available_clients:
            raise ValueError("min_fit_clients cannot exceed min_available_clients")

        # Check ZKP configuration
        if self.zkp.enable_zkp and self.zkp.proof_rigor not in [
            "low",
            "medium",
            "high",
        ]:
            raise ValueError(f"Invalid proof rigor: {self.zkp.proof_rigor}")

        if self.zkp.quantization_bits < 1 or self.zkp.quantization_bits > 32:
            raise ValueError(f"Invalid quantization bits: {self.zkp.quantization_bits}")

        # Check aggregation configuration
        if not 0 <= self.aggregation.momentum <= 1:
            raise ValueError(
                f"Momentum must be between 0 and 1: {self.aggregation.momentum}"
            )

        if self.aggregation.learning_rate <= 0:
            raise ValueError(
                f"Learning rate must be positive: {self.aggregation.learning_rate}"
            )

        # Check client configuration
        if self.client.local_epochs < 1:
            raise ValueError(
                f"Local epochs must be positive: {self.client.local_epochs}"
            )

        if self.client.batch_size < 1:
            raise ValueError(f"Batch size must be positive: {self.client.batch_size}")

        # Security checks for production
        if self.environment == Environment.PRODUCTION:
            if self.zkp.enable_zkp and not self.security.enable_tls:
                logger.warning("TLS is recommended for production ZKP deployments")

            if not self.security.enable_auth:
                logger.warning(
                    "Authentication is recommended for production deployments"
                )

        logger.info("Configuration validation passed")


class ConfigManager:
    """Configuration manager for loading and managing settings"""

    def __init__(self, env: Optional[str] = None, config_dir: Optional[Path] = None):
        self.env = Environment(env) if env else self._detect_environment()
        self.config_dir = config_dir or self._get_default_config_dir()
        self._config_cache: Optional[SecureFlConfig] = None

    def _detect_environment(self) -> Environment:
        """Detect current environment from environment variables"""
        env_var = os.getenv("SECURE_FL_ENV", "development").lower()
        try:
            return Environment(env_var)
        except ValueError:
            logger.warning(
                f"Unknown environment '{env_var}', defaulting to development"
            )
            return Environment.DEVELOPMENT

    def _get_default_config_dir(self) -> Path:
        """Get default configuration directory"""
        # Look for config in multiple locations
        locations = [
            Path.cwd() / "config",
            Path.cwd() / "configs",
            Path.home() / ".secure-fl",
            Path("/etc/secure-fl"),
        ]

        for location in locations:
            if location.exists():
                return location

        # Default to current directory
        return Path.cwd()

    def load_config(self) -> SecureFlConfig:
        """Load configuration from files and environment variables"""
        if self._config_cache is not None:
            return self._config_cache

        # Start with default configuration
        config_dict = {}

        # Load base configuration
        base_config_path = self.config_dir / "base.yaml"
        if base_config_path.exists():
            config_dict.update(self._load_yaml_config(base_config_path))

        # Load environment-specific configuration
        env_config_path = self.config_dir / f"{self.env.value}.yaml"
        if env_config_path.exists():
            env_config = self._load_yaml_config(env_config_path)
            config_dict = self._deep_merge(config_dict, env_config)

        # Override with environment variables
        env_overrides = self._load_env_overrides()
        config_dict = self._deep_merge(config_dict, env_overrides)

        # Create configuration object
        config = self._dict_to_config(config_dict)

        # Validate configuration
        config.validate()

        # Cache configuration
        self._config_cache = config

        return config

    def _load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables"""
        overrides = {}
        prefix = "SECURE_FL_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert SECURE_FL_SERVER_PORT to server.port
                config_key = key[len(prefix) :].lower()
                config_path = config_key.split("_")

                # Parse value based on type inference
                parsed_value = self._parse_env_value(value)

                # Set nested configuration
                current = overrides
                for path_part in config_path[:-1]:
                    if path_part not in current:
                        current[path_part] = {}
                    current = current[path_part]
                current[config_path[-1]] = parsed_value

        return overrides

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        if value.isdigit():
            return int(value)

        try:
            return float(value)
        except ValueError:
            pass

        # Try to parse as JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        return value

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SecureFlConfig:
        """Convert dictionary to configuration object"""
        # Create configuration with defaults
        config = SecureFlConfig()

        # Override environment if specified
        if "environment" in config_dict:
            config.environment = Environment(config_dict["environment"])
        else:
            config.environment = self.env

        if "debug" in config_dict:
            config.debug = config_dict["debug"]

        # Update component configurations
        components = {
            "server": ServerConfig,
            "client": ClientConfig,
            "zkp": ZKPConfig,
            "aggregation": AggregationConfig,
            "stability": StabilityConfig,
            "security": SecurityConfig,
            "database": DatabaseConfig,
            "monitoring": MonitoringConfig,
            "experiment": ExperimentConfig,
        }

        for component_name, component_class in components.items():
            if component_name in config_dict:
                component_dict = config_dict[component_name]
                component_config = getattr(config, component_name)

                # Update fields
                for field_name, field_value in component_dict.items():
                    if hasattr(component_config, field_name):
                        setattr(component_config, field_name, field_value)
                    else:
                        logger.warning(
                            f"Unknown configuration field: {component_name}.{field_name}"
                        )

        return config

    def save_config(self, config: SecureFlConfig, path: Optional[Path] = None) -> None:
        """Save configuration to file"""
        if path is None:
            path = self.config_dir / f"{self.env.value}.yaml"

        config_dict = self._config_to_dict(config)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {path}")

    def _config_to_dict(self, config: SecureFlConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        result = {
            "environment": config.environment.value,
            "debug": config.debug,
        }

        # Convert component configurations
        components = [
            "server",
            "client",
            "zkp",
            "aggregation",
            "stability",
            "security",
            "database",
            "monitoring",
            "experiment",
        ]

        for component_name in components:
            component_config = getattr(config, component_name)
            result[component_name] = {
                field.name: getattr(component_config, field.name)
                for field in component_config.__dataclass_fields__.values()
            }

        return result

    def reload_config(self) -> SecureFlConfig:
        """Reload configuration from files"""
        self._config_cache = None
        return self.load_config()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None
_config: Optional[SecureFlConfig] = None


@lru_cache(maxsize=1)
def get_config_manager(env: Optional[str] = None) -> ConfigManager:
    """Get or create global configuration manager"""
    global _config_manager
    if _config_manager is None or (env and _config_manager.env.value != env):
        _config_manager = ConfigManager(env=env)
    return _config_manager


def get_config(env: Optional[str] = None, reload: bool = False) -> SecureFlConfig:
    """Get current configuration"""
    global _config

    if _config is None or reload or (env and _config.environment.value != env):
        config_manager = get_config_manager(env)
        _config = config_manager.load_config()

    return _config


def set_config(config: SecureFlConfig) -> None:
    """Set global configuration"""
    global _config
    _config = config


def create_default_configs(config_dir: Path) -> None:
    """Create default configuration files"""
    config_dir.mkdir(parents=True, exist_ok=True)

    # Base configuration
    base_config = {
        "server": {
            "host": "localhost",
            "port": 8080,
            "num_rounds": 10,
        },
        "client": {
            "local_epochs": 1,
            "batch_size": 32,
        },
        "zkp": {
            "enable_zkp": True,
            "proof_rigor": "medium",
        },
        "monitoring": {
            "log_level": "INFO",
        },
    }

    # Environment-specific configurations
    configs = {
        "base.yaml": base_config,
        "development.yaml": {
            "debug": True,
            "monitoring": {"log_level": "DEBUG"},
            "zkp": {"enable_zkp": False},  # Faster development
        },
        "testing.yaml": {
            "server": {"num_rounds": 2},
            "client": {"local_epochs": 1},
            "zkp": {"enable_zkp": False},
        },
        "production.yaml": {
            "debug": False,
            "security": {
                "enable_tls": True,
                "enable_auth": True,
            },
            "monitoring": {
                "metrics_enabled": True,
                "tracing_enabled": True,
            },
        },
    }

    for filename, config_data in configs.items():
        config_path = config_dir / filename
        if not config_path.exists():
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            logger.info(f"Created default config: {config_path}")


# Convenience functions for common configuration access patterns


def get_server_config() -> ServerConfig:
    """Get server configuration"""
    return get_config().server


def get_client_config() -> ClientConfig:
    """Get client configuration"""
    return get_config().client


def get_zkp_config() -> ZKPConfig:
    """Get ZKP configuration"""
    return get_config().zkp


def is_production() -> bool:
    """Check if running in production environment"""
    return get_config().environment == Environment.PRODUCTION


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled"""
    return get_config().debug or get_config().environment == Environment.DEVELOPMENT


# Export all public classes and functions
__all__ = [
    "SecureFlConfig",
    "ServerConfig",
    "ClientConfig",
    "ZKPConfig",
    "AggregationConfig",
    "StabilityConfig",
    "SecurityConfig",
    "DatabaseConfig",
    "MonitoringConfig",
    "ExperimentConfig",
    "ConfigManager",
    "Environment",
    "LogLevel",
    "get_config",
    "get_config_manager",
    "set_config",
    "create_default_configs",
    "get_server_config",
    "get_client_config",
    "get_zkp_config",
    "is_production",
    "is_debug_enabled",
]
