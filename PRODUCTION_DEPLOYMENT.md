# Production Deployment Guide for Secure FL

This guide provides comprehensive instructions for deploying the Secure Federated Learning framework in production environments.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Configuration Management](#configuration-management)
5. [Security Hardening](#security-hardening)
6. [Deployment Options](#deployment-options)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Performance Optimization](#performance-optimization)
9. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance](#maintenance)

## Overview

Secure FL is designed for production deployment with enterprise-grade features including:

- **High Availability**: Redundant server deployment
- **Scalability**: Support for thousands of federated clients
- **Security**: End-to-end encryption, authentication, and ZKP verification
- **Monitoring**: Comprehensive metrics and health checking
- **Compliance**: Audit trails and regulatory compliance features

## Prerequisites

### Hardware Requirements

#### Minimum Requirements (Development/Testing)
- **CPU**: 4 cores, 2.5GHz
- **Memory**: 8GB RAM
- **Storage**: 50GB SSD
- **Network**: 100 Mbps
- **GPU**: Optional (CUDA-compatible for acceleration)

#### Recommended Production Requirements
- **CPU**: 16+ cores, 3.0GHz (Intel Xeon or AMD EPYC)
- **Memory**: 64GB+ RAM
- **Storage**: 500GB+ NVMe SSD
- **Network**: 1+ Gbps with low latency
- **GPU**: NVIDIA Tesla/Quadro with 16GB+ VRAM (optional)

#### High-Scale Production Requirements
- **CPU**: 32+ cores, 3.2GHz
- **Memory**: 128GB+ RAM
- **Storage**: 1TB+ NVMe SSD RAID
- **Network**: 10+ Gbps dedicated connection
- **GPU**: Multiple NVIDIA A100/H100 GPUs

### Software Requirements

#### Operating System
- **Linux**: Ubuntu 20.04+ LTS, RHEL 8+, CentOS 8+
- **Container**: Docker 20.10+, Kubernetes 1.20+
- **Python**: 3.8-3.12

#### Dependencies
```bash
# System packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    postgresql-client \
    redis-tools \
    htop \
    tmux \
    nginx

# Node.js (for ZKP tools)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

## Environment Setup

### 1. Installation Methods

#### Method 1: PyPI Installation (Recommended)
```bash
# Create virtual environment
python3 -m venv /opt/secure-fl
source /opt/secure-fl/bin/activate

# Install secure-fl
pip install secure-fl[all]

# Setup ZKP tools
secure-fl setup zkp
```

#### Method 2: Docker Deployment
```bash
# Pull latest image
docker pull krishantt/secure-fl:latest

# Run server
docker run -d \
  --name secure-fl-server \
  -p 8080:8080 \
  -p 9090:9090 \
  -e SECURE_FL_ENV=production \
  -v /opt/secure-fl/config:/app/config \
  -v /opt/secure-fl/data:/app/data \
  krishantt/secure-fl:server
```

#### Method 3: Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

### 2. ZKP Tools Installation

#### Cairo (for zk-STARKs)
```bash
# Install Cairo
pip install cairo-lang

# Verify installation
cairo-compile --version
```

#### Circom & SnarkJS (for zk-SNARKs)
```bash
# Install Circom
npm install -g circom

# Install SnarkJS
npm install -g snarkjs

# Verify installation
circom --version
snarkjs help
```

### 3. Database Setup (Optional)

#### PostgreSQL for Metrics Storage
```bash
# Install PostgreSQL
sudo apt-get install -y postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE secure_fl_metrics;
CREATE USER secure_fl_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE secure_fl_metrics TO secure_fl_user;
\q
EOF
```

#### Redis for Caching
```bash
# Install Redis
sudo apt-get install -y redis-server

# Configure Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

## Configuration Management

### 1. Environment Configuration

Create configuration files for different environments:

#### `/opt/secure-fl/config/production.yaml`
```yaml
environment: production
debug: false

server:
  host: "0.0.0.0"
  port: 8080
  num_rounds: 100
  min_fit_clients: 10
  min_evaluate_clients: 5
  timeout: 600
  ssl_enabled: true
  ssl_cert_path: "/etc/ssl/certs/secure-fl.crt"
  ssl_key_path: "/etc/ssl/private/secure-fl.key"
  auth_enabled: true
  jwt_secret: "${JWT_SECRET}"

client:
  server_address: "secure-fl.yourcompany.com:8080"
  local_epochs: 5
  batch_size: 64
  timeout: 300
  retry_attempts: 3

zkp:
  enable_zkp: true
  proof_rigor: "high"
  blockchain_verification: true
  quantization_bits: 16
  proof_timeout: 300

aggregation:
  momentum: 0.95
  learning_rate: 0.001
  adaptive_momentum: true

security:
  enable_tls: true
  enable_auth: true
  auth_method: "jwt"
  max_model_size_mb: 1000.0
  enable_rate_limiting: true
  rate_limit_requests: 1000
  rate_limit_window: 60

monitoring:
  metrics_enabled: true
  metrics_port: 9090
  log_level: "INFO"
  log_format: "json"
  log_file: "/var/log/secure-fl/server.log"
  health_check_enabled: true
  tracing_enabled: true
  jaeger_endpoint: "jaeger-collector:14268"

database:
  enabled: true
  database_url: "postgresql://secure_fl_user:password@localhost:5432/secure_fl_metrics"
  metrics_retention_days: 90
  logs_retention_days: 30
```

### 2. Environment Variables

Create `/opt/secure-fl/.env`:
```bash
# Environment
SECURE_FL_ENV=production

# Security
JWT_SECRET=your-very-secure-jwt-secret-key
TLS_CERT_PATH=/etc/ssl/certs/secure-fl.crt
TLS_KEY_PATH=/etc/ssl/private/secure-fl.key

# Database
DATABASE_URL=postgresql://secure_fl_user:password@localhost:5432/secure_fl_metrics
REDIS_URL=redis://localhost:6379/0

# Monitoring
JAEGER_ENDPOINT=http://jaeger-collector:14268/api/traces
PROMETHEUS_GATEWAY=http://prometheus-pushgateway:9091

# ZKP Configuration
CAIRO_PATH=/usr/local/bin/cairo-compile
CIRCOM_PATH=/usr/local/bin/circom
SNARKJS_PATH=/usr/local/bin/snarkjs

# Blockchain (if enabled)
ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/your-project-id
CONTRACT_ADDRESS=0x1234567890123456789012345678901234567890
PRIVATE_KEY_PATH=/opt/secure-fl/keys/ethereum-key.json
```

## Security Hardening

### 1. TLS/SSL Configuration

#### Generate SSL Certificates
```bash
# Self-signed certificate (for testing)
openssl req -x509 -newkey rsa:4096 -keyout /etc/ssl/private/secure-fl.key \
  -out /etc/ssl/certs/secure-fl.crt -days 365 -nodes \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=secure-fl.yourcompany.com"

# Set proper permissions
sudo chmod 600 /etc/ssl/private/secure-fl.key
sudo chmod 644 /etc/ssl/certs/secure-fl.crt
```

#### Production Certificate (Let's Encrypt)
```bash
# Install certbot
sudo apt-get install -y certbot

# Generate certificate
sudo certbot certonly --standalone \
  -d secure-fl.yourcompany.com \
  --email admin@yourcompany.com \
  --agree-tos --no-eff-email

# Setup auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 2. Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8080/tcp  # Secure FL server
sudo ufw allow 9090/tcp  # Prometheus metrics
sudo ufw enable
```

### 3. User and Permissions

```bash
# Create secure-fl user
sudo useradd -r -s /bin/false secure-fl
sudo mkdir -p /opt/secure-fl/{config,data,logs}
sudo chown -R secure-fl:secure-fl /opt/secure-fl
sudo chmod 750 /opt/secure-fl
```

### 4. Authentication Setup

#### JWT Configuration
```bash
# Generate JWT secret
python3 -c "import secrets; print(secrets.token_urlsafe(64))"
```

#### API Key Management
```python
# generate_api_keys.py
import secrets
import json

api_keys = {}
for i in range(10):  # Generate 10 API keys
    key_id = f"api_key_{i:03d}"
    api_key = secrets.token_urlsafe(32)
    api_keys[key_id] = {
        "key": api_key,
        "permissions": ["read", "write"],
        "created": "2024-01-01T00:00:00Z"
    }

with open('/opt/secure-fl/config/api_keys.json', 'w') as f:
    json.dump(api_keys, f, indent=2)

print("API keys generated and saved to api_keys.json")
```

## Deployment Options

### 1. Standalone Deployment

#### Systemd Service Configuration

Create `/etc/systemd/system/secure-fl-server.service`:
```ini
[Unit]
Description=Secure FL Server
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=secure-fl
Group=secure-fl
WorkingDirectory=/opt/secure-fl
Environment=SECURE_FL_ENV=production
EnvironmentFile=/opt/secure-fl/.env
ExecStart=/opt/secure-fl/bin/secure-fl server --config /opt/secure-fl/config/production.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=secure-fl-server

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/secure-fl/data /opt/secure-fl/logs

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

#### Start Services
```bash
# Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl enable secure-fl-server
sudo systemctl start secure-fl-server

# Check status
sudo systemctl status secure-fl-server
sudo journalctl -u secure-fl-server -f
```

### 2. Docker Deployment

#### Docker Compose Configuration

Create `docker-compose.production.yml`:
```yaml
version: '3.8'

services:
  secure-fl-server:
    image: krishantt/secure-fl:latest
    container_name: secure-fl-server
    command: ["secure-fl", "server", "--config", "/app/config/production.yaml"]
    ports:
      - "8080:8080"
      - "9090:9090"
    environment:
      - SECURE_FL_ENV=production
    env_file:
      - .env
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./logs:/app/logs
      - /etc/ssl:/etc/ssl:ro
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G

  postgres:
    image: postgres:14
    container_name: secure-fl-postgres
    environment:
      POSTGRES_DB: secure_fl_metrics
      POSTGRES_USER: secure_fl_user
      POSTGRES_PASSWORD: ${DATABASE_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U secure_fl_user"]
      interval: 30s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: secure-fl-redis
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 5s
      retries: 5

  nginx:
    image: nginx:alpine
    container_name: secure-fl-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - /etc/ssl:/etc/ssl:ro
    depends_on:
      - secure-fl-server
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: secure-fl-prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: secure-fl-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

#### Start Docker Services
```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f secure-fl-server
```

### 3. Kubernetes Deployment

#### Namespace
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: secure-fl
```

#### ConfigMap
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: secure-fl-config
  namespace: secure-fl
data:
  production.yaml: |
    environment: production
    debug: false
    server:
      host: "0.0.0.0"
      port: 8080
      num_rounds: 100
      min_fit_clients: 10
    # ... (rest of config)
```

#### Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-fl-server
  namespace: secure-fl
spec:
  replicas: 3
  selector:
    matchLabels:
      app: secure-fl-server
  template:
    metadata:
      labels:
        app: secure-fl-server
    spec:
      containers:
      - name: secure-fl-server
        image: krishantt/secure-fl:latest
        ports:
        - containerPort: 8080
        - containerPort: 9090
        env:
        - name: SECURE_FL_ENV
          value: "production"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: tls
          mountPath: /etc/ssl
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: secure-fl-config
      - name: tls
        secret:
          secretName: secure-fl-tls
```

#### Service and Ingress
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: secure-fl-service
  namespace: secure-fl
spec:
  selector:
    app: secure-fl-server
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: secure-fl-ingress
  namespace: secure-fl
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - secure-fl.yourcompany.com
    secretName: secure-fl-tls
  rules:
  - host: secure-fl.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: secure-fl-service
            port:
              number: 8080
```

## Monitoring and Observability

### 1. Prometheus Configuration

Create `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "secure_fl_rules.yml"

scrape_configs:
  - job_name: 'secure-fl-server'
    static_configs:
      - targets: ['secure-fl-server:9090']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### 2. Grafana Dashboards

Create dashboard JSON for Grafana with panels for:
- Training progress metrics
- System resource utilization
- ZKP performance metrics
- Client participation rates
- Error rates and response times

### 3. Log Management

#### Structured Logging Configuration
```yaml
# In production.yaml
monitoring:
  log_format: "json"
  log_level: "INFO"
  log_file: "/var/log/secure-fl/server.log"
  log_rotation: true
  log_max_size: "100MB"
  log_backup_count: 10
```

#### Log Aggregation with ELK Stack
```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/secure-fl/*.log
  fields:
    service: secure-fl
    environment: production

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "secure-fl-logs-%{+yyyy.MM.dd}"

logging.level: info
```

## Performance Optimization

### 1. System Tuning

#### Kernel Parameters
```bash
# /etc/sysctl.d/99-secure-fl.conf
# Network optimization
net.core.rmem_max = 268435456
net.core.wmem_max = 268435456
net.ipv4.tcp_rmem = 4096 87380 268435456
net.ipv4.tcp_wmem = 4096 65536 268435456
net.core.netdev_max_backlog = 5000

# File descriptor limits
fs.file-max = 2097152
```

#### Apply settings
```bash
sudo sysctl -p /etc/sysctl.d/99-secure-fl.conf
```

#### Limits Configuration
```bash
# /etc/security/limits.d/secure-fl.conf
secure-fl soft nofile 65536
secure-fl hard nofile 65536
secure-fl soft nproc 32768
secure-fl hard nproc 32768
```

### 2. Application Performance

#### Memory Optimization
```yaml
# In production.yaml
aggregation:
  batch_size: 100  # Process clients in batches
  memory_optimization: true

zkp:
  proof_caching: true
  cache_size: 1000

client:
  prefetch_batches: 2
  pin_memory: true
```

#### GPU Acceleration
```yaml
# Enable GPU acceleration
training:
  use_gpu: true
  gpu_memory_fraction: 0.8
  mixed_precision: true

zkp:
  gpu_acceleration: true
```

### 3. Database Optimization

#### PostgreSQL Tuning
```sql
-- postgresql.conf optimizations
shared_buffers = '2GB'
effective_cache_size = '8GB'
maintenance_work_mem = '256MB'
checkpoint_completion_target = 0.9
wal_buffers = '16MB'
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

## Backup and Disaster Recovery

### 1. Data Backup Strategy

#### Database Backup
```bash
#!/bin/bash
# backup-database.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups/secure-fl"
DB_NAME="secure_fl_metrics"
DB_USER="secure_fl_user"

mkdir -p $BACKUP_DIR

# Create database backup
pg_dump -U $DB_USER -h localhost -d $DB_NAME \
  --clean --if-exists --create \
  | gzip > $BACKUP_DIR/db_backup_$DATE.sql.gz

# Retain only last 7 days of backups
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +7 -delete

echo "Database backup completed: db_backup_$DATE.sql.gz"
```

#### Configuration Backup
```bash
#!/bin/bash
# backup-config.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups/secure-fl"
CONFIG_DIR="/opt/secure-fl/config"

mkdir -p $BACKUP_DIR

# Backup configuration files
tar -czf $BACKUP_DIR/config_backup_$DATE.tar.gz -C $CONFIG_DIR .

# Backup certificates
tar -czf $BACKUP_DIR/certs_backup_$DATE.tar.gz -C /etc/ssl .

echo "Configuration backup completed"
```

### 2. Disaster Recovery Plan

#### Recovery Procedures
1. **Infrastructure Recovery**
   - Provision new servers
   - Restore network connectivity
   - Install base system packages

2. **Application Recovery**
   ```bash
   # Restore from backup
   sudo -u postgres psql << EOF
   DROP DATABASE IF EXISTS secure_fl_metrics;
   EOF
   
   gunzip -c /opt/backups/secure-fl/db_backup_latest.sql.gz | \
     sudo -u postgres psql
   
   # Restore configuration
   tar -xzf /opt/backups/secure-fl/config_backup_latest.tar.gz \
     -C /opt/secure-fl/config
   ```

3. **Service Recovery**
   ```bash
   # Restart services
   sudo systemctl start secure-fl-server
   sudo systemctl status secure-fl-server
   ```

#### Recovery Time Objectives
- **RTO (Recovery Time Objective)**: 4 hours
- **RPO (Recovery Point Objective)**: 1 hour
- **Data Loss Tolerance**: < 15 minutes

### 3. Automated Backup Schedule

```bash
# Add to crontab
0 2 * * * /opt/secure-fl/scripts/backup-database.sh
0 3 * * 0 /opt/secure-fl/scripts/backup-config.sh
```

## Troubleshooting

### 1. Common Issues

#### High Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check for memory leaks
valgrind --leak-check=full secure-fl server

# Monitor memory over time
while true; do
  echo "$(date): $(free -m | grep Mem:)"
  sleep 60
done
```

#### Performance Issues
```bash
# Check CPU usage
top -p $(pgrep -f secure-fl)

# Profile application
python -m cProfile -o profile_output secure-fl server

# Network diagnostics
netstat -tulpn | grep :8080
ss -tulpn | grep :8080
```

#### ZKP Tool Issues
```bash
# Verify ZKP tools
cairo-compile --version
circom --version
snarkjs help

# Test proof generation
secure-fl setup check
```

### 2. Log Analysis

#### Common Error Patterns
```bash
# Search for errors
grep -i error /var/log/secure-fl/server.log

# Check for timeouts
grep -i timeout /var/log/secure-fl/server.log

# Monitor client connections
grep "client.*connected" /var/log/secure-fl/server.log
```

#### Performance Analysis
```bash
# Analyze response times
awk '/request_duration/ {print $NF}' /var/log/secure-fl/server.log | \
  sort -n | tail -100

# Check aggregation times
grep "aggregation_complete" /var/log/secure-fl/server.log
```

### 3. Health Checks

#### Manual Health Checks
```bash
# Check server health
curl -f http://localhost:8080/health

# Check metrics endpoint
curl http://localhost:9090/metrics

# Check database connectivity
psql -h localhost -U secure_fl_user -d secure_fl_metrics -c "SELECT 1"
```

#### Automated Monitoring
```bash
#!/bin/bash
# health-check.sh

HEALTH_URL="http://localhost:8080/health"
ALERT_EMAIL="admin@yourcompany.com"

if ! curl -f -s $HEALTH_URL > /dev/null; then
    echo "Health check failed" | mail -s "Secure FL Alert" $ALERT_EMAIL
    exit 1
fi

echo "Health check passed"
```

## Maintenance

### 1. Regular Maintenance Tasks

#### Daily Tasks
- Monitor system health and performance
- Check log files for errors
- Verify backup completion
- Monitor client connections

#### Weekly Tasks
- Update system packages
- Clean up old log files
- Review security alerts
- Performance analysis

#### Monthly Tasks
- Security audit
- Capacity planning review
- Disaster recovery testing
- Update documentation

### 2. Update Procedures

#### Security Updates
```bash
# System updates
sudo apt update && sudo apt upgrade -y

# Python package updates
pip list --outdated
pip install -U secure-fl
```

#### Application Updates
```bash
# Backup before update
./backup-database.sh
./backup-config.sh

# Update application
pip install -U secure-fl

# Restart services
sudo systemctl restart secure-fl-server

# Verify update
secure-fl --version
```

### 3. Capacity Planning

#### Metrics to Monitor
- CPU utilization trends
- Memory usage growth
- Network bandwidth utilization
- Storage space consumption
- Client connection patterns

#### Scaling Triggers
- CPU usage > 70% for 1 hour
- Memory usage > 80% for 30 minutes
- Response time > 5 seconds
- Error rate > 1%

### 4. Security Maintenance

#### Certificate Renewal
```bash
# Check certificate expiration
openssl x509 -in /etc/ssl/certs/secure-fl.crt -noout -dates

# Automated renewal with Let's Encrypt
certbot renew --dry-run
```

#### Security Scanning
```bash
# Vulnerability scanning
npm audit
pip-audit

# Port scanning
nmap -sS -O localhost
```

## Support and Resources

### Documentation
- [API Documentation](https://secure-fl.readthedocs.io/)
- [Configuration Reference](https://github.com/krishantt/secure-fl/blob/main/docs/configuration.md)
- [Troubleshooting Guide](https://github.com/krishantt/secure-fl/blob/main/docs/troubleshooting.md)

### Community
- [GitHub Issues](https://github.com/krishantt/secure-fl/issues)
- [Discussion Forum](https://github.com/krishantt/secure-fl/discussions)
- [Slack Channel](https://secure-fl.slack.com)

### Professional Support
- Email: support@secure-fl.com
- Phone: +1-555-SECURE-FL
- Enterprise Support: enterprise@secure-fl.com

---

For additional assistance or enterprise support, please contact the Secure FL team.