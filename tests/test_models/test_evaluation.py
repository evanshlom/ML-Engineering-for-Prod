"""Tests for model evaluation."""

import pytest
import torch
import numpy as np
from sklearn.metrics import accuracy_score

from src.models.evaluation import evaluate_model
from src.models.architecture import NICUReadinessClassifier


class TestModelEvaluation:
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model."""
        model = NICUReadinessClassifier()
        model.eval()
        return model
        
    @pytest.fixture
    def test_loader(self):
        """Create test data loader."""
        X = torch.randn(100, 5)
        y = torch.randint(0, 2, (100, 1)).float()
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(dataset, batch_size=32)
        
    def test_evaluate_model(self, trained_model, test_loader):
        """Test comprehensive model evaluation."""
        results = evaluate_model(trained_model, test_loader)
        
        assert 'classification_report' in results
        assert 'confusion_matrix' in results
        assert results['confusion_matrix'].shape == (2, 2)
        
    def test_evaluate_model_metrics(self, trained_model, test_loader):
        """Test evaluation produces valid metrics."""
        results = evaluate_model(trained_model, test_loader)
        
        # Check confusion matrix values
        cm = results['confusion_matrix']
        assert np.all(cm >= 0)
        assert cm.sum() == 100  # Total samples