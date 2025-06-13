.PHONY: help install test lint format clean data train serve docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean generated files"
	@echo "  data         Generate synthetic data"
	@echo "  train        Train model"
	@echo "  serve        Start API server"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"

install:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/ -v -m "not integration"

test-integration:
	pytest tests/ -v -m integration

lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf data/processed/*.csv
	rm -rf data/models/*.pt

generate-data:
	python -m src.data.generator

train-model:
	python -m src.models.training

serve:
	uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t nicu-kangaroo-ml:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/data:/app/data nicu-kangaroo-ml:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Development workflow
dev-setup: install generate-data
	@echo "Development environment ready!"

dev-train: generate-data train-model
	@echo "Model trained with latest data!"

dev-test: lint test
	@echo "All tests passed!"

# Production workflow
prod-build: test docker-build
	@echo "Production image built!"

prod-deploy: prod-build
	docker tag nicu-kangaroo-ml:latest nicu-kangaroo-ml:$(shell date +%Y%m%d-%H%M%S)
	@echo "Tagged for deployment!"