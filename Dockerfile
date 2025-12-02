# Multi-stage Dockerfile for Secure FL
FROM node:18-slim as node-stage

# Install ZKP tools
RUN npm install -g circom snarkjs

FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

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

# Copy requirements and install Python dependencies
COPY --chown=app:app pyproject.toml README.md LICENSE ./
COPY --chown=app:app secure_fl/ ./secure_fl/

# Install PDM and dependencies
RUN pip install --user pdm && \
    ~/.local/bin/pdm install --prod --no-dev

# Install Cairo for zk-STARKs
RUN pip install --user cairo-lang || echo "Cairo installation failed, continuing without it"

# Copy remaining application files
COPY --chown=app:app proofs/ ./proofs/
COPY --chown=app:app blockchain/ ./blockchain/
COPY --chown=app:app experiments/ ./experiments/
COPY --chown=app:app k8s/ ./k8s/
COPY --chown=app:app infra/ ./infra/

# Create necessary directories
RUN mkdir -p data logs results

# Add PDM and pip to PATH
ENV PATH="/home/app/.local/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import secure_fl; print('Secure FL is healthy')" || exit 1

# Default command
CMD ["secure-fl", "demo"]

# Production stage
FROM base as production

# Install production dependencies only
RUN ~/.local/bin/pdm install --prod --no-dev

# Server stage for running FL server
FROM production as server
EXPOSE 8080
CMD ["secure-fl", "server", "--host", "0.0.0.0", "--port", "8080"]

# Client stage for running FL client
FROM production as client
CMD ["secure-fl", "client", "--client-id", "docker-client"]

# Development stage with all dependencies
FROM base as development

USER root
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER app

# Install development dependencies
RUN ~/.local/bin/pdm install -d

# Install additional development tools
RUN pip install --user \
    jupyter \
    jupyterlab \
    ipywidgets

# Expose Jupyter port
EXPOSE 8888

CMD ["bash"]
