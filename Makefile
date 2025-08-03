# Makefile for SNN-Fusion Project
# Provides common development and deployment tasks

.PHONY: help install install-dev test test-unit test-integration test-api test-performance
.PHONY: lint format type-check security-check pre-commit-install
.PHONY: docs docs-serve clean clean-cache clean-all
.PHONY: docker-build docker-run docker-dev
.PHONY: train-example deploy-example benchmark
.PHONY: setup-dev setup-prod

# Default target
help:
	@echo "SNN-Fusion Development Commands"
	@echo "==============================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install           Install package in development mode"
	@echo "  install-dev       Install with development dependencies"
	@echo "  setup-dev         Complete development environment setup"
	@echo "  pre-commit-install Install pre-commit hooks"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test              Run all tests"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-api          Run API integration tests only"
	@echo "  test-performance  Run performance benchmarks"
	@echo "  test-coverage     Generate coverage report"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint              Run linting (flake8)"
	@echo "  format            Format code (black, isort)"
	@echo "  type-check        Run type checking (mypy)"
	@echo "  security-check    Run security checks (bandit)"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs              Build documentation"
	@echo "  docs-serve        Serve documentation locally"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build      Build Docker image"
	@echo "  docker-run        Run Docker container"
	@echo "  docker-dev        Run development container"
	@echo ""
	@echo "Example Commands:"
	@echo "  train-example     Run example training"
	@echo "  deploy-example    Run example deployment"
	@echo "  benchmark         Run performance benchmarks"
	@echo ""
	@echo "Cleanup Commands:"
	@echo "  clean             Clean build artifacts"
	@echo "  clean-cache       Clean cache files"
	@echo "  clean-all         Clean everything"

# Installation and setup
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,hardware,viz]"

setup-dev: install-dev pre-commit-install
	@echo "Development environment setup complete!"

pre-commit-install:
	pre-commit install

# Testing
test:
	pytest

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-api:
	pytest tests/integration/test_api.py -v -m api

test-performance:
	pytest tests/performance/ -v -m performance --benchmark-only

test-coverage:
	pytest --cov=snn_fusion --cov-report=html --cov-report=term-missing

test-gpu:
	pytest -m gpu --gpu-enabled

test-hardware:
	pytest -m hardware --hardware-enabled

# Code quality
lint:
	flake8 src/ tests/
	
format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/snn_fusion/

security-check:
	bandit -r src/

# Full quality check
quality-check: lint type-check security-check
	@echo "Code quality checks completed!"

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs && make serve

docs-clean:
	cd docs && make clean

# Docker commands
docker-build:
	docker build -t snn-fusion:latest .

docker-run:
	docker run --rm -it -p 8080:8080 snn-fusion:latest

docker-dev:
	docker-compose -f docker-compose.dev.yml up --build

# Example and benchmark commands
train-example:
	snn-fusion init-config --config-template basic --output-file config.yaml
	snn-fusion train --data-dir ./data/example --config config.yaml --epochs 5

deploy-example:
	snn-fusion deploy --model-path ./models/example.pt --hardware loihi2 --output-dir ./deployment

benchmark:
	python scripts/benchmark.py --config configs/benchmark.yaml

# Database commands
db-migrate:
	python -c "from snn_fusion.database import get_database; get_database().run_migrations()"

db-reset:
	rm -f snn_fusion.db
	$(MAKE) db-migrate

# Development server
dev-server:
	python -m snn_fusion.api.app

api-server:
	flask --app snn_fusion.api.app run --debug --port 8080

dashboard:
	snn-fusion dashboard --port 8080

# Data commands
download-data:
	python scripts/download_datasets.py --dataset all

process-data:
	python scripts/process_data.py --input-dir data/raw --output-dir data/processed

# Model commands
export-model:
	python scripts/export_model.py --model-path $(MODEL_PATH) --format onnx

optimize-model:
	python scripts/optimize_model.py --model-path $(MODEL_PATH) --hardware $(HARDWARE)

# Cleanup commands
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf outputs/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-cache:
	rm -rf cache/
	rm -rf .cache/
	rm -rf ~/.cache/snn_fusion/

clean-logs:
	rm -rf logs/
	rm -f *.log

clean-all: clean clean-cache clean-logs
	rm -rf data/processed/
	rm -rf models/checkpoints/
	docker system prune -f

# CI/CD commands
ci-test:
	pytest --cov=snn_fusion --cov-report=xml --junit-xml=test-results.xml

ci-build:
	python -m build

ci-deploy:
	twine upload dist/*

# Development utilities
notebook:
	jupyter lab --port 8888

profile:
	python -m cProfile -o profile.stats scripts/profile_training.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

memory-profile:
	python -m memory_profiler scripts/memory_profile.py

# Environment info
env-info:
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "PyTorch version: $$(python -c 'import torch; print(torch.__version__)')"
	@echo "CUDA available: $$(python -c 'import torch; print(torch.cuda.is_available())')"
	@echo "GPU count: $$(python -c 'import torch; print(torch.cuda.device_count())')"

# Check dependencies
check-deps:
	pip check
	pip list --outdated

# Update dependencies
update-deps:
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt

# Variables for common paths
DATA_DIR ?= ./data
MODEL_DIR ?= ./models
OUTPUT_DIR ?= ./outputs
CONFIG_FILE ?= config.yaml
HARDWARE ?= loihi2
MODEL_PATH ?= ./models/trained_model.pt

# Export variables for scripts
export DATA_DIR MODEL_DIR OUTPUT_DIR