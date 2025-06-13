"""PyTest unit tests for FastAPI serving endpoints."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import torch
from unittest.mock import Mock, patch

from src.serving.api import app
from src.models.architecture import NICUReadinessClassifier, ModelWithExplainability


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create mock model for testing."""
    base_model = NICUReadinessClassifier()
    return ModelWithExplainability(base_model)


@pytest.fixture
def valid_features():
    """Valid feature set for testing."""
    return {
        "heart_rate": 120.0,
        "oxygen_saturation": 96.5,
        "respiratory_rate": 35.0,
        "weight_grams": 2100.0,
        "temperature_celsius": 36.8
    }


@pytest.fixture
def invalid_features():
    """Invalid feature set for testing."""
    return {
        "heart_rate": 250.0,  # Too high
        "oxygen_saturation": 120.0,  # > 100%
        "respiratory_rate": -10.0,  # Negative
        "weight_grams": 100.0,  # Too low
        "temperature_celsius": 45.0  # Too high
    }


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_model_loaded(self, client, mock_model):
        """Test health check when model is loaded."""
        with patch('src.serving.api.MODEL', mock_model):
            with patch('src.serving.api.MODEL_INFO', {'version': '1.0.0'}):
                response = client.get("/health")
                
                assert response.status_code == 200
                data = response.json()
                assert data['status'] == 'healthy'
                assert data['model_loaded'] is True
                assert data['model_version'] == '1.0.0'
    
    def test_health_check_model_not_loaded(self, client):
        """Test health check when model is not loaded."""
        with patch('src.serving.api.MODEL', None):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'unhealthy'
            assert data['model_loaded'] is False


class TestPredictionEndpoint:
    """Test single prediction endpoint."""
    
    def test_predict_invalid_features(self, client, mock_model, invalid_features):
        """Test prediction with invalid features."""
        with patch('src.serving.api.MODEL', mock_model):
            request_data = {"features": invalid_features}
            
            response = client.post("/predict", json=request_data)
            
            assert response.status_code == 422  # Validation error
    
    def test_predict_model_not_loaded(self, client, valid_features):
        """Test prediction when model is not loaded."""
        with patch('src.serving.api.MODEL', None):
            request_data = {"features": valid_features}
            
            response = client.post("/predict", json=request_data)
            
            assert response.status_code == 503
            assert 'Model not loaded' in response.json()['detail']
    
    @pytest.mark.parametrize("missing_field", [
        "heart_rate", "oxygen_saturation", "respiratory_rate", 
        "weight_grams", "temperature_celsius"
    ])
    def test_predict_missing_fields(self, client, mock_model, valid_features, missing_field):
        """Test prediction with missing required fields."""
        with patch('src.serving.api.MODEL', mock_model):
            features = valid_features.copy()
            del features[missing_field]
            request_data = {"features": features}
            
            response = client.post("/predict", json=request_data)
            
            assert response.status_code == 422


class TestBatchPredictionEndpoint:
    """Test batch prediction endpoint."""
    
    def test_batch_predict_empty(self, client, mock_model):
        """Test batch prediction with empty list."""
        with patch('src.serving.api.MODEL', mock_model):
            request_data = {"instances": []}
            
            response = client.post("/predict/batch", json=request_data)
            
            assert response.status_code == 422  # Validation error
    
    def test_batch_predict_too_many(self, client, mock_model, valid_features):
        """Test batch prediction with too many instances."""
        with patch('src.serving.api.MODEL', mock_model):
            request_data = {
                "instances": [valid_features] * 1001  # Over limit
            }
            
            response = client.post("/predict/batch", json=request_data)
            
            assert response.status_code == 422