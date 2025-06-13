"""Tests for training pipeline."""

import pytest
import torch
import pandas as pd
from unittest.mock import Mock, patch

from src.models.training import (
    EarlyStopping, prepare_data, create_dataloaders,
    train_epoch, evaluate
)
from src.models.architecture import NICUReadinessClassifier


class TestEarlyStopping:
    
    def test_early_stopping_trigger(self):
        """Test early stopping triggers correctly."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        
        # Decreasing loss - should not trigger
        assert early_stopping(0.5) == False
        assert early_stopping(0.4) == False
        assert early_stopping(0.3) == False
        
        # Plateauing loss - should trigger after patience
        assert early_stopping(0.301) == False  # counter = 1
        assert early_stopping(0.302) == False  # counter = 2
        assert early_stopping(0.301) == True   # counter = 3, triggers now
        assert early_stopping.early_stop == True


class TestDataPreparation:
    
    def test_prepare_data(self):
        """Test data preparation to tensors."""
        df = pd.DataFrame({
            'heart_rate': [120.0, 130.0],
            'oxygen_saturation': [96.5, 97.0],
            'respiratory_rate': [35.0, 38.0],
            'weight_grams': [2100.0, 2200.0],
            'temperature_celsius': [36.8, 36.9],
            'suitable_for_kangaroo_care': [True, False]
        })
        
        X, y = prepare_data(df)
        
        assert X.shape == (2, 5)
        assert y.shape == (2, 1)
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        
    def test_create_dataloaders(self):
        """Test dataloader creation."""
        train_df = pd.DataFrame({
            'heart_rate': [120.0] * 100,
            'oxygen_saturation': [96.5] * 100,
            'respiratory_rate': [35.0] * 100,
            'weight_grams': [2100.0] * 100,
            'temperature_celsius': [36.8] * 100,
            'suitable_for_kangaroo_care': [True] * 100
        })
        
        val_df = train_df.copy()
        
        train_loader, val_loader = create_dataloaders(
            train_df, val_df, batch_size=32
        )
        
        assert len(train_loader) == 4  # 100 samples / 32 batch size
        assert len(val_loader) == 4


class TestTrainingFunctions:
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        return NICUReadinessClassifier()
        
    def test_train_epoch(self, mock_model):
        """Test single training epoch."""
        # Create simple dataset
        X = torch.randn(64, 5)
        y = torch.randint(0, 2, (64, 1)).float()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(mock_model.parameters())
        device = torch.device('cpu')
        
        loss, accuracy = train_epoch(
            mock_model, loader, criterion, optimizer, device
        )
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        
    def test_evaluate(self, mock_model):
        """Test model evaluation."""
        # Create simple dataset
        X = torch.randn(64, 5)
        y = torch.randint(0, 2, (64, 1)).float()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        criterion = torch.nn.BCEWithLogitsLoss()
        device = torch.device('cpu')
        
        loss, accuracy, metrics = evaluate(
            mock_model, loader, criterion, device
        )
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc_roc' in metrics