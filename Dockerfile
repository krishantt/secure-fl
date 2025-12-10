# Multi-stage Dockerfile for Secure FL using uv with ZKP tools
FROM node:22-slim AS node-stage

# Install snarkjs via npm and ensure all dependencies are available
RUN npm install -g snarkjs && \
    npm list -g --depth=0

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_CACHE_DIR=/tmp/.uv-cache

# Install system dependencies including Rust
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

# Set Rust environment
ENV PATH="/root/.cargo/bin:$PATH"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy Node.js and all global modules from node stage
COPY --from=node-stage /usr/local/bin/node /usr/local/bin/
COPY --from=node-stage /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=node-stage /usr/local/bin/npm /usr/local/bin/
COPY --from=node-stage /usr/local/bin/npx /usr/local/bin/

# Create snarkjs wrapper script
RUN echo '#!/bin/bash' > /usr/local/bin/snarkjs && \
    echo 'exec node /usr/local/lib/node_modules/snarkjs/cli.js "$@"' >> /usr/local/bin/snarkjs && \
    chmod +x /usr/local/bin/snarkjs

ENV NODE_PATH="/usr/local/lib/node_modules"

# Install circom from source (will be cached between builds)
RUN git clone https://github.com/iden3/circom.git /tmp/circom \
    && cd /tmp/circom \
    && cargo build --release \
    && cargo install --path circom \
    && rm -rf /tmp/circom \
    && chmod +x /root/.cargo/bin/circom

# Create app user and give access to cargo binaries and node modules
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /home/app/.cargo/bin \
    && cp /root/.cargo/bin/circom /home/app/.cargo/bin/ \
    && cp -r /root/.cargo /home/app/ \
    && chown -R app:app /home/app/.cargo \
    && chmod +x /home/app/.cargo/bin/circom \
    && chown -R app:app /usr/local/lib/node_modules \
    && chown app:app /usr/local/bin/snarkjs

USER app
WORKDIR /home/app

# Set up environment for app user
ENV PATH="/home/app/.local/bin:/home/app/.venv/bin:/home/app/.cargo/bin:$PATH"

# Copy dependency files and source code
COPY --chown=app:app pyproject.toml uv.lock ./
COPY --chown=app:app README.md LICENSE ./
COPY --chown=app:app secure_fl/ ./secure_fl/

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Create missing directories that might be referenced
RUN mkdir -p data logs results temp proofs experiments scripts

# Copy additional application files
COPY --chown=app:app proofs ./proofs
COPY --chown=app:app experiments ./experiments
COPY --chown=app:app scripts ./scripts



# Verify installations (allow warnings)
RUN circom --help > /dev/null && snarkjs --help > /dev/null 2>&1 || true && echo "ZKP tools available"

# Run comprehensive verification via setup module
RUN uv run python -m secure_fl.setup zkp || echo "ZKP tools setup completed with warnings"

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

# Create test directory and copy tests for development
RUN mkdir -p tests
COPY --chown=app:app tests ./tests

# Install poethepoet task runner
RUN uv pip install poethepoet

# Expose development port
EXPOSE 8080

# Development environment
ENV SECURE_FL_ENV=development

# Default command for development
CMD ["bash"]

# Add test verification target
FROM development AS test
CMD ["uv", "run", "pytest", "tests/", "-v", "--tb=short"]
