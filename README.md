# ML Engineering for Prod

PyTorch Application with Pydantic and FastAPI for deployment, plus PyTest and Pandera for tests. Demo for best practices with Infant Healthcare NICU classification model about infant readiness for kangaroo care.

## Quick Demo

### Setup & Run Steps

#### 1. Start the containers
```cmd
docker-compose up -d
```
Builds and starts the API container in background. Creates the NICU ML service on port 8000.

#### 2. Generate data
```cmd
docker-compose exec api make data
```
Generates 2000 synthetic NICU patient records with realistic correlations.

#### 3. Run tests
```cmd
docker-compose exec api make test
```
Runs full PyTest suite including model architecture tests, API endpoint tests, and data validation tests.

#### 4. Train model
```cmd
docker-compose exec api make train
```
- Trains PyTorch binary classifier with early stopping
- Saves model to `data/models/`

#### 5. Test the API health
```cmd
curl http://localhost:8000/health
```
Verifies API is running and model is loaded. Should return `{"status": "healthy", "model_loaded": true}`.

#### 6. Make a prediction
```cmd
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": {\"heart_rate\": 120, \"oxygen_saturation\": 96.5, \"respiratory_rate\": 35, \"weight_grams\": 2100, \"temperature_celsius\": 36.8}}"
```
Sends infant vitals to get kangaroo care readiness prediction. Returns probability, confidence level, and feature importance.

#### 7. Stop containers
```cmd
docker-compose down
```
Stops and removes containers. Data persists in `data/` directory.

## Production Workflow

### 1. Run all tests first
```cmd
docker-compose up -d
docker-compose exec api make data
docker-compose exec api pytest tests/ -v
```
Tests include:
- **PyTest**: Model architecture, gradients, API endpoints, data generation
- **Pandera**: Automatic validation during data generation/loading
- **Pydantic**: Automatic validation on every API request/response

### 2. Train model only if tests pass
```cmd
docker-compose exec api make train
```

### 3. Run integration tests
```cmd
docker-compose exec api pytest tests/ -m integration -v
```

### 4. Verify API serving
```cmd
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": {\"heart_rate\": 120, \"oxygen_saturation\": 96.5, \"respiratory_rate\": 35, \"weight_grams\": 2100, \"temperature_celsius\": 36.8}}"
```

### 5. Build production image
```cmd
docker-compose exec api make prod-build
```

## Project Overview

ML system predicting NICU infant readiness for kangaroo care based on:
- Heart Rate (optimal: 110-130 bpm)
- Oxygen Saturation (optimal: ≥95%)
- Respiratory Rate (optimal: 30-40 breaths/min)
- Weight (optimal: >1500g)
- Temperature (optimal: 36.5-37.5°C)

## Key Features

### Testing (PyTest)
- Unit tests for forward/backward passes
- API endpoint tests
- Data validation tests
- Integration tests
- ~90% coverage

### Data Validation (Pandera)
- Runtime schema validation
- Feature correlation checks
- Overfitting detection
- Convergence monitoring

### API (FastAPI + Pydantic)
- Type-safe request/response models
- Automatic documentation
- Batch predictions
- Model explainability

### MLOps Best Practices
- Reproducible training
- Model versioning
- Structured logging
- Docker deployment
- Makefile automation

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `POST /explain` - Model explainability
- `GET /model/info` - Model metadata

## Development

```bash
make help           # Show all commands
make dev-setup      # Install & generate data
make dev-train      # Train model
make dev-test       # Run linting & tests
make clean          # Clean generated files
```

## Project Structure

```
├── src/
│   ├── data/       # Data generation & validation (Pandera)
│   ├── models/     # PyTorch model & training
│   ├── serving/    # FastAPI + Pydantic schemas
│   └── utils/      # Config & logging
├── tests/          # PyTest suite
├── configs/        # Training configuration
├── scripts/        # Helper scripts
└── data/           # Data directories
```

## Performance

- Accuracy: ~92%
- Inference: <10ms per prediction
- Throughput: 1000+ requests/second
- Model size: <1MB

## License

MIT