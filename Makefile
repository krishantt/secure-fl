.PHONY: help install install-dev install-prod update-deps test test-fast lint format type-check clean clean-all docker-build docker-demo docker-dev server client demo benchmark setup-zkp docs version

# Default target
help: ## Show this help message
	@echo "Secure FL - Zero-Knowledge Federated Learning"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install production dependencies
	uv sync --no-dev

install-dev: ## Install development dependencies
	uv sync --all-extras --dev

install-prod: ## Install production dependencies only
	uv sync --no-dev --frozen

update-deps: ## Update all dependencies
	uv lock --upgrade

# Development targets
test: ## Run all tests with coverage
	uv run pytest tests/ -v --cov=secure_fl --cov-report=html

test-fast: ## Run tests quickly (exit on first failure)
	uv run pytest tests/ -x -v

test-watch: ## Run tests in watch mode
	uv run pytest-watch tests/

lint: ## Check code with linter
	uv run ruff check .

format: ## Format code
	uv run ruff format .

type-check: ## Run type checking
	uv run mypy secure_fl/

lint-fix: ## Fix linting issues
	uv run ruff check --fix .

precommit: format lint-fix test-fast ## Run pre-commit checks

ci: lint test ## Run CI pipeline locally (type-check disabled temporarily)

# Application targets
server: ## Start FL server
	uv run python -m secure_fl.cli server

client: ## Start FL client
	uv run python -m secure_fl.cli client

demo: ## Run demo
	uv run python experiments/demo.py

benchmark: ## Run benchmarks
	uv run python experiments/benchmark.py

train: ## Run training experiment
	uv run python experiments/train.py

# Setup targets
setup-zkp: ## Setup ZKP tools
	uv run python -m secure_fl.setup zkp

setup-full: ## Full setup
	uv run python -m secure_fl.setup full

check-system: ## Check system requirements
	uv run python -m secure_fl.setup check

# Docker targets
docker-build: ## Build Docker images
	./scripts/docker.sh build

docker-demo: ## Run Docker demo
	./scripts/docker.sh demo

docker-dev: ## Start Docker development environment
	./scripts/docker.sh dev

docker-compose-up: ## Start services with docker-compose
	./scripts/docker.sh up

docker-compose-down: ## Stop docker-compose services
	./scripts/docker.sh down

docker-compose-logs: ## View docker-compose logs
	./scripts/docker.sh logs --follow

# Version management
version: ## Show current version
	uv run python scripts/version.py show

version-increment: ## Increment version
	uv run python scripts/version.py increment

version-publish: ## Publish version
	uv run python scripts/version.py publish

# Cleanup targets
clean: ## Clean temporary files
	uv run python -m secure_fl.setup clean

clean-all: ## Clean all generated files
	uv run python -m secure_fl.setup clean --all

clean-cache: ## Clean Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

docker-clean: ## Clean Docker images and containers
	./scripts/docker.sh clean

docker-test: ## Test Docker images
	./scripts/docker.sh test

docker-test-all: ## Test all Docker images
	./scripts/docker.sh test --all

docker-test-quick: ## Quick test of Docker images
	./scripts/docker.sh test --quick

# Build targets
build: ## Build package
	uv build

build-wheel: ## Build wheel only
	uv build --wheel

build-sdist: ## Build source distribution only
	uv build --sdist

# Development workflow shortcuts
dev-setup: install-dev setup-zkp ## Complete development setup
	@echo "Development environment ready!"

quick-test: format lint-fix test-fast ## Quick development test cycle

full-check: format lint type-check test ## Full code quality check

# Environment management
env-info: ## Show environment information
	@echo "Python version:"
	@python --version
	@echo "UV version:"
	@uv --version
	@echo "Virtual environment:"
	@echo $$VIRTUAL_ENV
	@echo "Installed packages:"
	@uv pip list

# Security
security-check: ## Run security checks
	uv run pip-audit

# Performance
profile: ## Profile the application
	uv run py-spy top --pid $$(pgrep -f "python -m secure_fl")

memory-profile: ## Memory profiling
	uv run mprof run python -m secure_fl.experiments.demo
	uv run mprof plot

# Experimental
experimental-setup: ## Setup experimental features
	uv sync --extra medical --dev

# Database/Results management
clean-results: ## Clean experiment results
	rm -rf results/temp_*
	rm -rf experiments/results/temp_*

backup-results: ## Backup experiment results
	tar -czf results/backup_$$(date +%Y%m%d_%H%M%S).tar.gz results/ experiments/results/

# Container management
container-shell: ## Open shell in running container
	docker exec -it $$(docker ps -q --filter ancestor=secure-fl:dev) bash

container-logs: ## View container logs
	docker logs -f $$(docker ps -q --filter ancestor=secure-fl:server)

# Integration testing
test-integration: ## Run integration tests
	uv run pytest tests/ -m integration -v

test-zkp: ## Run ZKP-specific tests
	uv run pytest tests/ -m zkp -v

test-slow: ## Run slow tests
	uv run pytest tests/ -m slow -v

# Quick development commands
dev: install-dev ## Alias for install-dev
prod: install-prod ## Alias for install-prod
fmt: format ## Alias for format
check: precommit ## Alias for precommit
up: docker-compose-up ## Alias for docker-compose-up
down: docker-compose-down ## Alias for docker-compose-down

# Docker shortcuts
docker: ## Show Docker help
	./scripts/docker.sh --help

docker-info: ## Show Docker system information
	./scripts/docker.sh info

docker-status: ## Show Docker service status
	./scripts/docker.sh status

docker-clean-all: ## Clean all Docker resources
	./scripts/docker.sh clean --all --force
