# Multi-stage Dockerfile for Secure FL using uv
FROM node:18-slim AS node-stage

# Install ZKP tools
RUN npm install -g circom snarkjs

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_CACHE_DIR=/tmp/.uv-cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy Node.js and ZKP tools from node stage
COPY --from=node-stage /usr/local/bin/node /usr/local/bin/
COPY --from=node-stage /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=node-stage /usr/local/bin/npm /usr/local/bin/
COPY --from=node-stage /usr/local/bin/npx /usr/local/bin/
COPY --from=node-stage /usr/local/bin/circom /usr/local/bin/
COPY --from=node-stage /usr/local/bin/snarkjs /usr/local/bin/

# Create app user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy dependency files and source code
COPY --chown=app:app pyproject.toml uv.lock ./
COPY --chown=app:app README.md LICENSE ./
COPY --chown=app:app secure_fl/ ./secure_fl/

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy additional application files (only if they exist)
COPY --chown=app:app proofs/ ./proofs/
COPY --chown=app:app experiments/ ./experiments/
COPY --chown=app:app scripts/ ./scripts/

# Create missing directories that might be referenced
RUN mkdir -p data logs results temp

# Add uv and virtual environment to PATH
ENV PATH="/home/app/.local/bin:/home/app/.venv/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import secure_fl; print('Secure FL is healthy')" || exit 1

# Default command
CMD ["python", "-m", "secure_fl.cli", "demo"]

# Production stage
FROM base AS production

# Ensure production environment
ENV SECURE_FL_ENV=production

# Server stage for running FL server
FROM production AS server
EXPOSE 8080
CMD ["python", "-m", "secure_fl.cli", "server", "--host", "0.0.0.0", "--port", "8080"]

# Client stage for running FL client
FROM production AS client
CMD ["python", "-m", "secure_fl.cli", "client", "--client-id", "docker-client"]

# Development stage with all dependencies
FROM base AS development

USER root
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

USER app

# Install development dependencies
RUN uv sync --frozen --all-extras --dev

# Install poethepoet task runner
RUN uv pip install poethepoet

# Expose development port
EXPOSE 8080

# Development environment
ENV SECURE_FL_ENV=development

# Default command for development
CMD ["bash"]
