"""PyTest unit tests for FastAPI serving endpoints."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import torch
from unittest.mock import Mock, patch

from src.serving.api import app, MODEL, MODEL_INFO
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
    
    def test_predict_success(self, client, mock_model, valid_features):
        """Test successful prediction."""
        with patch('src.serving.api.MODEL', mock_model):
            with patch('src.serving.api.MODEL_INFO', {'version': '1.0.0'}):
                request_data = {
                    "features": valid_features,
                    "patient_id": "TEST-001"
                }
                
                response = client.post("/predict", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert 'suitable_for_kangaroo_care' in data
                assert 'probability' in data
                assert 0 <= data['probability'] <= 1
                assert data['confidence'] in ['high', 'medium', 'low']
                assert 'feature_importance' in data
                assert len(data['feature_importance']) == 5
    
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
    
    def test_batch_predict_success(self, client, mock_model, valid_features):
        """Test successful batch prediction."""
        with patch('src.serving.api.MODEL', mock_model):
            with patch('src.serving.api.MODEL_INFO', {'version': '1.0.0'}):
                request_data = {
                    "instances": [valid_features, valid_features, valid_features]
                }
                
                response = client.post("/predict/batch", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert data['total_count'] == 3
                assert len(data['predictions']) == 3
                assert 'suitable_count' in data
                assert 'average_probability' in data
                assert data['processing_time_ms'] > 0
    
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


class TestExplainabilityEndpoint:
    """Test explainability endpoint."""
    
    def test_explain_success(self, client, mock_model, valid_features):
        """Test successful explanation request."""
        with patch('src.serving.api.MODEL', mock_model):
            with patch('src.serving.api.MODEL_INFO', {'version': '1.0.0'}):
                request_data = {
                    "features": valid_features,
                    "explanation_type": "feature_importance"
                }
                
                response = client.post("/explain", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert 'prediction' in data
                assert 'feature_importance' in data
                assert 'interpretation' in data
                assert data['explanation_type'] == 'feature_importance'


class TestModelInfoEndpoint:
    """Test model info endpoint."""
    
    def test_model_info_success(self, client, mock_model):
        """Test successful model info retrieval."""
        model_info = {
            'version': '1.0.0',
            'training_date': datetime.utcnow().isoformat(),
            'metrics': {
                'accuracy': 0.92,
                'precision': 0.89,
                'recall': 0.94,
                'f1_score': 0.91,
                'auc_roc': 0.95
            },
            'hyperparameters': {
                'hidden_dims': [64, 32, 16],
                'dropout_rate': 0.3
            }
        }
        
        with patch('src.serving.api.MODEL', mock_model):
            with patch('src.serving.api.MODEL_INFO', model_info):
                response = client.get("/model/info")
                
                assert response.status_code == 200
                data = response.json()
                assert data['model_version'] == '1.0.0'
                assert data['model_name'] == 'NICUReadinessClassifier'
                assert 'metrics' in data
                assert data['metrics']['accuracy'] == 0.92


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API workflows."""
    
    def test_prediction_workflow(self, client, mock_model, valid_features):
        """Test complete prediction workflow."""
        with patch('src.serving.api.MODEL', mock_model):
            with patch('src.serving.api.MODEL_INFO', {'version': '1.0.0'}):
                # Check health
                health_response = client.get("/health")
                assert health_response.status_code == 200
                
                # Make prediction
                predict_response = client.post("/predict", json={
                    "features": valid_features,
                    "patient_id": "INT-TEST-001"
                })
                assert predict_response.status_code == 200
    
    def test_concurrent_predictions(self, client, mock_model, valid_features):
        """Test API handles concurrent requests."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        with patch('src.serving.api.MODEL', mock_model):
            with patch('src.serving.api.MODEL_INFO', {'version': '1.0.0'}):
                def make_prediction():
                    return client.post("/predict", json={"features": valid_features})
                
                # Make 10 concurrent requests
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(make_prediction) for _ in range(10)]
                    responses = [f.result() for f in futures]
                
                assert all(r.status_code == 200 for r in responses)