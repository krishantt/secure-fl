"""
Production-Grade Monitoring and Health Check System for Secure FL

This module provides comprehensive monitoring, health checks, metrics collection,
and observability features for production deployments of the Secure FL framework.

Features:
- Health check endpoints and status monitoring
- Performance metrics collection and export
- System resource monitoring
- Custom metrics for FL training progress
- Prometheus metrics integration
- Distributed tracing support
- Alerting and notification system
- Log aggregation and analysis

Usage:
    from secure_fl.monitoring import HealthChecker, MetricsCollector, setup_monitoring

    # Setup monitoring
    health_checker = HealthChecker()
    metrics_collector = MetricsCollector()

    # Start monitoring services
    setup_monitoring(health_checker, metrics_collector)

    # Check system health
    health_status = health_checker.check_all()

    # Record custom metrics
    metrics_collector.record_training_round(round_num, loss, accuracy)
"""

import asyncio
import logging
import os
import platform
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

import psutil
import requests
import torch

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        Summary,
        generate_latest,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import opentelemetry
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status levels"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types of system components to monitor"""

    SERVER = "server"
    CLIENT = "client"
    DATABASE = "database"
    NETWORK = "network"
    STORAGE = "storage"
    ZKP_TOOLS = "zkp_tools"
    ML_FRAMEWORK = "ml_framework"


@dataclass
class HealthCheckResult:
    """Result of a health check operation"""

    component: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "component": self.component,
            "component_type": self.component_type.value,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
            "details": self.details,
        }


@dataclass
class SystemMetrics:
    """System performance metrics"""

    cpu_usage_percent: float
    memory_usage_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    gpu_usage_percent: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_percent": self.memory_usage_percent,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_used_gb": self.disk_used_gb,
            "disk_total_gb": self.disk_total_gb,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "gpu_usage_percent": self.gpu_usage_percent,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "gpu_memory_total_gb": self.gpu_memory_total_gb,
            "timestamp": self.timestamp.isoformat(),
        }


class HealthChecker:
    """Comprehensive health checking system"""

    def __init__(self, check_interval: int = 30, timeout: int = 10):
        self.check_interval = check_interval
        self.timeout = timeout
        self.checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register_check(
        self, name: str, check_func: Callable[[], HealthCheckResult]
    ) -> None:
        """Register a custom health check function"""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource availability"""
        start_time = time.time()

        try:
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory check
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk check
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent

            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            issues = []

            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > 70:
                status = HealthStatus.WARNING
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")

            if memory_percent > 90:
                status = HealthStatus.CRITICAL
                issues.append(f"Memory usage critical: {memory_percent:.1f}%")
            elif memory_percent > 70:
                status = HealthStatus.WARNING
                issues.append(f"Memory usage high: {memory_percent:.1f}%")

            if disk_percent > 90:
                status = HealthStatus.CRITICAL
                issues.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent > 80:
                status = HealthStatus.WARNING
                issues.append(f"Disk usage high: {disk_percent:.1f}%")

            message = "System resources healthy" if not issues else "; ".join(issues)

            return HealthCheckResult(
                component="system_resources",
                component_type=ComponentType.SERVER,
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                component_type=ComponentType.SERVER,
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {e}",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    def check_pytorch(self) -> HealthCheckResult:
        """Check PyTorch availability and GPU support"""
        start_time = time.time()

        try:
            # Basic PyTorch check
            import torch

            torch_version = torch.__version__

            # GPU availability check
            cuda_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if cuda_available else 0

            details = {
                "torch_version": torch_version,
                "cuda_available": cuda_available,
                "gpu_count": gpu_count,
            }

            if cuda_available:
                details["cuda_version"] = torch.version.cuda
                details["cudnn_version"] = torch.backends.cudnn.version()

            status = HealthStatus.HEALTHY
            message = f"PyTorch {torch_version} available"

            if cuda_available:
                message += f" with {gpu_count} GPU(s)"

            return HealthCheckResult(
                component="pytorch",
                component_type=ComponentType.ML_FRAMEWORK,
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                details=details,
            )

        except Exception as e:
            return HealthCheckResult(
                component="pytorch",
                component_type=ComponentType.ML_FRAMEWORK,
                status=HealthStatus.CRITICAL,
                message=f"PyTorch not available: {e}",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
            )

    def check_zkp_tools(self) -> HealthCheckResult:
        """Check ZKP tools availability"""
        start_time = time.time()

        tools_status = {}
        overall_status = HealthStatus.HEALTHY

        # Check Cairo
        try:
            import subprocess

            result = subprocess.run(
                ["cairo-compile", "--version"],
                capture_output=True,
                timeout=5,
                text=True,
            )
            tools_status["cairo"] = {
                "available": result.returncode == 0,
                "version": result.stdout.strip() if result.returncode == 0 else None,
                "error": result.stderr if result.returncode != 0 else None,
            }
        except Exception as e:
            tools_status["cairo"] = {"available": False, "error": str(e)}

        # Check Circom
        try:
            result = subprocess.run(
                ["circom", "--version"], capture_output=True, timeout=5, text=True
            )
            tools_status["circom"] = {
                "available": result.returncode == 0,
                "version": result.stdout.strip() if result.returncode == 0 else None,
                "error": result.stderr if result.returncode != 0 else None,
            }
        except Exception as e:
            tools_status["circom"] = {"available": False, "error": str(e)}

        # Check SnarkJS
        try:
            result = subprocess.run(
                ["snarkjs", "help"], capture_output=True, timeout=5, text=True
            )
            tools_status["snarkjs"] = {
                "available": result.returncode == 0,
                "version": "available" if result.returncode == 0 else None,
                "error": result.stderr if result.returncode != 0 else None,
            }
        except Exception as e:
            tools_status["snarkjs"] = {"available": False, "error": str(e)}

        # Determine overall status
        available_tools = [
            name for name, info in tools_status.items() if info["available"]
        ]

        if len(available_tools) == 0:
            overall_status = HealthStatus.CRITICAL
            message = "No ZKP tools available"
        elif len(available_tools) < len(tools_status):
            overall_status = HealthStatus.WARNING
            message = (
                f"Some ZKP tools unavailable: {', '.join(available_tools)} available"
            )
        else:
            message = f"All ZKP tools available: {', '.join(available_tools)}"

        return HealthCheckResult(
            component="zkp_tools",
            component_type=ComponentType.ZKP_TOOLS,
            status=overall_status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=(time.time() - start_time) * 1000,
            details=tools_status,
        )

    def check_network_connectivity(
        self, targets: Optional[List[str]] = None
    ) -> HealthCheckResult:
        """Check network connectivity to specified targets"""
        start_time = time.time()

        if targets is None:
            targets = ["8.8.8.8", "1.1.1.1"]  # Google DNS, Cloudflare DNS

        connectivity_results = {}

        for target in targets:
            try:
                # Simple socket connection test
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((target, 53))  # DNS port
                sock.close()

                connectivity_results[target] = {
                    "reachable": result == 0,
                    "error": (
                        None if result == 0 else f"Connection failed (code: {result})"
                    ),
                }
            except Exception as e:
                connectivity_results[target] = {"reachable": False, "error": str(e)}

        # Determine overall status
        reachable_targets = [
            target for target, info in connectivity_results.items() if info["reachable"]
        ]

        if len(reachable_targets) == 0:
            status = HealthStatus.CRITICAL
            message = "No network connectivity"
        elif len(reachable_targets) < len(targets):
            status = HealthStatus.WARNING
            message = f"Partial network connectivity: {len(reachable_targets)}/{len(targets)} targets reachable"
        else:
            status = HealthStatus.HEALTHY
            message = "Network connectivity healthy"

        return HealthCheckResult(
            component="network",
            component_type=ComponentType.NETWORK,
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=(time.time() - start_time) * 1000,
            details=connectivity_results,
        )

    def check_disk_space(self, paths: Optional[List[str]] = None) -> HealthCheckResult:
        """Check disk space for specified paths"""
        start_time = time.time()

        if paths is None:
            paths = ["/", "/tmp"]

        disk_info = {}
        overall_status = HealthStatus.HEALTHY
        issues = []

        for path in paths:
            try:
                if os.path.exists(path):
                    usage = psutil.disk_usage(path)
                    percent_used = (usage.used / usage.total) * 100

                    disk_info[path] = {
                        "total_gb": usage.total / (1024**3),
                        "used_gb": usage.used / (1024**3),
                        "free_gb": usage.free / (1024**3),
                        "percent_used": percent_used,
                    }

                    if percent_used > 95:
                        overall_status = HealthStatus.CRITICAL
                        issues.append(f"{path}: {percent_used:.1f}% full")
                    elif percent_used > 85:
                        if overall_status != HealthStatus.CRITICAL:
                            overall_status = HealthStatus.WARNING
                        issues.append(f"{path}: {percent_used:.1f}% full")
                else:
                    disk_info[path] = {"error": "Path does not exist"}

            except Exception as e:
                disk_info[path] = {"error": str(e)}
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING

        message = "Disk space healthy" if not issues else "; ".join(issues)

        return HealthCheckResult(
            component="disk_space",
            component_type=ComponentType.STORAGE,
            status=overall_status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=(time.time() - start_time) * 1000,
            details=disk_info,
        )

    def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}

        # Run built-in checks
        builtin_checks = [
            ("system_resources", self.check_system_resources),
            ("pytorch", self.check_pytorch),
            ("zkp_tools", self.check_zkp_tools),
            ("network", self.check_network_connectivity),
            ("disk_space", self.check_disk_space),
        ]

        for name, check_func in builtin_checks:
            try:
                results[name] = check_func()
            except Exception as e:
                results[name] = HealthCheckResult(
                    component=name,
                    component_type=ComponentType.SERVER,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {e}",
                    timestamp=datetime.now(),
                    response_time_ms=0,
                )

        # Run custom checks
        for name, check_func in self.checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                results[name] = HealthCheckResult(
                    component=name,
                    component_type=ComponentType.SERVER,
                    status=HealthStatus.CRITICAL,
                    message=f"Custom check failed: {e}",
                    timestamp=datetime.now(),
                    response_time_ms=0,
                )

        self.last_results = results
        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.last_results:
            self.check_all()

        statuses = [result.status for result in self.last_results.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def start_periodic_checks(self) -> None:
        """Start periodic health checks in background thread"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._periodic_check_loop, daemon=True)
        self._thread.start()
        logger.info("Started periodic health checks")

    def stop_periodic_checks(self) -> None:
        """Stop periodic health checks"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Stopped periodic health checks")

    def _periodic_check_loop(self) -> None:
        """Main loop for periodic health checks"""
        while self._running:
            try:
                self.check_all()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in periodic health check: {e}")
                time.sleep(self.check_interval)


class MetricsCollector:
    """Production metrics collection and export"""

    def __init__(self, enable_prometheus: bool = True, prometheus_port: int = 9090):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.prometheus_port = prometheus_port

        # Initialize Prometheus metrics if available
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
            self._prometheus_server_started = False

        # Internal metrics storage
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self._metrics_lock = threading.Lock()

    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics"""
        if not self.enable_prometheus:
            return

        # System metrics
        self.cpu_usage = Gauge(
            "secure_fl_cpu_usage_percent",
            "CPU usage percentage",
            registry=self.registry,
        )
        self.memory_usage = Gauge(
            "secure_fl_memory_usage_percent",
            "Memory usage percentage",
            registry=self.registry,
        )
        self.disk_usage = Gauge(
            "secure_fl_disk_usage_percent",
            "Disk usage percentage",
            registry=self.registry,
        )

        # FL training metrics
        self.training_rounds_total = Counter(
            "secure_fl_training_rounds_total",
            "Total number of training rounds completed",
            registry=self.registry,
        )
        self.training_loss = Gauge(
            "secure_fl_training_loss",
            "Current training loss",
            ["client_id", "round"],
            registry=self.registry,
        )
        self.training_accuracy = Gauge(
            "secure_fl_training_accuracy",
            "Current training accuracy",
            ["client_id", "round"],
            registry=self.registry,
        )

        # ZKP metrics
        self.proof_generation_duration = Histogram(
            "secure_fl_proof_generation_duration_seconds",
            "Time taken to generate proofs",
            ["proof_type"],
            registry=self.registry,
        )
        self.proof_verification_duration = Histogram(
            "secure_fl_proof_verification_duration_seconds",
            "Time taken to verify proofs",
            ["proof_type"],
            registry=self.registry,
        )

        # Network metrics
        self.network_bytes_sent = Counter(
            "secure_fl_network_bytes_sent_total",
            "Total bytes sent over network",
            registry=self.registry,
        )
        self.network_bytes_received = Counter(
            "secure_fl_network_bytes_received_total",
            "Total bytes received over network",
            registry=self.registry,
        )

        # Health check metrics
        self.health_check_duration = Histogram(
            "secure_fl_health_check_duration_seconds",
            "Time taken for health checks",
            ["check_name"],
            registry=self.registry,
        )
        self.health_check_status = Gauge(
            "secure_fl_health_check_status",
            "Health check status (0=healthy, 1=warning, 2=critical)",
            ["check_name"],
            registry=self.registry,
        )

    def start_prometheus_server(self) -> None:
        """Start Prometheus metrics server"""
        if not self.enable_prometheus or self._prometheus_server_started:
            return

        try:
            start_http_server(self.prometheus_port, registry=self.registry)
            self._prometheus_server_started = True
            logger.info(
                f"Started Prometheus metrics server on port {self.prometheus_port}"
            )
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = disk.percent
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)

        # Network I/O
        net_io = psutil.net_io_counters()

        # GPU metrics (if available)
        gpu_usage_percent = None
        gpu_memory_used_gb = None
        gpu_memory_total_gb = None

        if torch.cuda.is_available():
            try:
                # Simple GPU utilization (this is basic - for production, use nvidia-ml-py)
                gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total_gb = torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024**3)
                gpu_usage_percent = (gpu_memory_used_gb / gpu_memory_total_gb) * 100
            except:
                pass

        metrics = SystemMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            disk_usage_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            network_bytes_sent=net_io.bytes_sent,
            network_bytes_recv=net_io.bytes_recv,
            gpu_usage_percent=gpu_usage_percent,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
        )

        # Update Prometheus metrics
        if self.enable_prometheus:
            self.cpu_usage.set(cpu_percent)
            self.memory_usage.set(memory_percent)
            self.disk_usage.set(disk_percent)

        # Store in history
        self._store_metric("system_metrics", metrics.to_dict())

        return metrics

    def record_training_round(
        self,
        round_num: int,
        client_id: str,
        loss: float,
        accuracy: float,
        duration_seconds: float,
    ) -> None:
        """Record training round metrics"""
        metrics = {
            "round_num": round_num,
            "client_id": client_id,
            "loss": loss,
            "accuracy": accuracy,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat(),
        }

        # Update Prometheus metrics
        if self.enable_prometheus:
            self.training_rounds_total.inc()
            self.training_loss.labels(client_id=client_id, round=str(round_num)).set(
                loss
            )
            self.training_accuracy.labels(
                client_id=client_id, round=str(round_num)
            ).set(accuracy)

        # Store in history
        self._store_metric("training_rounds", metrics)

        logger.info(
            f"Recorded training metrics: round {round_num}, client {client_id}, "
            f"loss {loss:.4f}, accuracy {accuracy:.4f}"
        )

    def record_proof_generation(
        self,
        proof_type: str,
        duration_seconds: float,
        success: bool,
        proof_size_bytes: Optional[int] = None,
    ) -> None:
        """Record ZKP proof generation metrics"""
        metrics = {
            "proof_type": proof_type,
            "duration_seconds": duration_seconds,
            "success": success,
            "proof_size_bytes": proof_size_bytes,
            "timestamp": datetime.now().isoformat(),
        }

        # Update Prometheus metrics
        if self.enable_prometheus:
            self.proof_generation_duration.labels(proof_type=proof_type).observe(
                duration_seconds
            )

        # Store in history
        self._store_metric("proof_generation", metrics)

        logger.info(
            f"Recorded proof generation: {proof_type}, "
            f"duration {duration_seconds:.3f}s, success {success}"
        )

    def record_proof_verification(
        self, proof_type: str, duration_seconds: float, success: bool
    ) -> None:
        """Record ZKP proof verification metrics"""
        metrics = {
            "proof_type": proof_type,
            "duration_seconds": duration_seconds,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }

        # Update Prometheus metrics
        if self.enable_prometheus:
            self.proof_verification_duration.labels(proof_type=proof_type).observe(
                duration_seconds
            )

        # Store in history
        self._store_metric("proof_verification", metrics)

    def record_health_check(self, check_name: str, result: HealthCheckResult) -> None:
        """Record health check metrics"""
        # Map status to numeric value for Prometheus
        status_map = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 1,
            HealthStatus.CRITICAL: 2,
            HealthStatus.UNKNOWN: 3,
        }

        if self.enable_prometheus:
            self.health_check_duration.labels(check_name=check_name).observe(
                result.response_time_ms / 1000
            )
            self.health_check_status.labels(check_name=check_name).set(
                status_map[result.status]
            )

        # Store in history
        self._store_metric("health_checks", result.to_dict())

    def _store_metric(self, metric_type: str, data: Dict[str, Any]) -> None:
        """Store metric in internal history"""
        with self._metrics_lock:
            if metric_type not in self.metrics_history:
                self.metrics_history[metric_type] = []

            self.metrics_history[metric_type].append(data)

            # Keep only last 1000 entries per metric type
            if len(self.metrics_history[metric_type]) > 1000:
                self.metrics_history[metric_type] = self.metrics_history[metric_type][
                    -1000:
                ]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        with self._metrics_lock:
            summary = {}

            for metric_type, history in self.metrics_history.items():
                if history:
                    summary[metric_type] = {
                        "count": len(history),
                        "latest": history[-1] if history else None,
                        "time_range": {
                            "start": history[0].get("timestamp") if history else None,
                            "end": history[-1].get("timestamp") if history else None,
                        },
                    }

            return summary

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if not self.enable_prometheus:
            return ""

        return generate_latest(self.registry).decode("utf-8")


class TracingManager:
    """Distributed tracing management"""

    def __init__(
        self, service_name: str = "secure-fl", jaeger_endpoint: Optional[str] = None
    ):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.tracer = None

        if TRACING_AVAILABLE:
            self._setup_tracing()

    def _setup_tracing(self) -> None:
        """Setup OpenTelemetry tracing"""
        if not TRACING_AVAILABLE:
            return

        # Configure tracing
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(self.service_name)

        # Setup Jaeger exporter if endpoint provided
        if self.jaeger_endpoint:
            try:
                jaeger_exporter = JaegerExporter(
                    agent_host_name=self.jaeger_endpoint.split(":")[0],
                    agent_port=int(self.jaeger_endpoint.split(":")[1]),
                )

                span_processor = BatchSpanProcessor(jaeger_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)

                logger.info(f"Configured Jaeger tracing to {self.jaeger_endpoint}")
            except Exception as e:
                logger.error(f"Failed to setup Jaeger tracing: {e}")

        # Instrument requests
        RequestsInstrumentor().instrument()

    def create_span(self, name: str, **kwargs):
        """Create a new tracing span"""
        if self.tracer:
            return self.tracer.start_span(name, **kwargs)
        else:
            # Return a dummy context manager if tracing not available
            class DummySpan:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def set_attribute(self, key, value):
                    pass

                def set_status(self, status):
                    pass

            return DummySpan()


def setup_monitoring(
    health_checker: Optional[HealthChecker] = None,
    metrics_collector: Optional[MetricsCollector] = None,
    enable_tracing: bool = False,
    jaeger_endpoint: Optional[str] = None,
) -> tuple:
    """Setup complete monitoring infrastructure"""

    # Create components if not provided
    if health_checker is None:
        health_checker = HealthChecker()

    if metrics_collector is None:
        metrics_collector = MetricsCollector()

    tracing_manager = None
    if enable_tracing:
        tracing_manager = TracingManager(jaeger_endpoint=jaeger_endpoint)

    # Start services
    health_checker.start_periodic_checks()
    metrics_collector.start_prometheus_server()

    logger.info("Monitoring infrastructure initialized")

    return health_checker, metrics_collector, tracing_manager


def create_health_endpoint(health_checker: HealthChecker):
    """Create FastAPI health check endpoint"""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        app = FastAPI()

        @app.get("/health")
        async def health_check():
            results = health_checker.check_all()
            overall_status = health_checker.get_overall_status()

            response_data = {
                "status": overall_status.value,
                "timestamp": datetime.now().isoformat(),
                "checks": {name: result.to_dict() for name, result in results.items()},
            }

            status_code = 200 if overall_status == HealthStatus.HEALTHY else 503
            return JSONResponse(content=response_data, status_code=status_code)

        @app.get("/health/{component}")
        async def component_health_check(component: str):
            results = health_checker.check_all()

            if component in results:
                result = results[component]
                status_code = 200 if result.status == HealthStatus.HEALTHY else 503
                return JSONResponse(content=result.to_dict(), status_code=status_code)
            else:
                return JSONResponse(
                    content={"error": f"Component '{component}' not found"},
                    status_code=404,
                )

        return app

    except ImportError:
        logger.warning("FastAPI not available, cannot create health endpoint")
        return None


# Export all public classes and functions
__all__ = [
    "HealthChecker",
    "HealthCheckResult",
    "HealthStatus",
    "ComponentType",
    "MetricsCollector",
    "SystemMetrics",
    "TracingManager",
    "setup_monitoring",
    "create_health_endpoint",
]
