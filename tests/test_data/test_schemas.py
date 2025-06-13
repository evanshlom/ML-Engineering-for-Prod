"""Tests for Pandera schemas."""

import pytest
import pandas as pd
import pandera as pa

from src.data.schemas import (
    NICUFeatureSchema, NICULabelSchema, NICUDatasetSchema,
    ModelMetricsSchema, TrainingMetricsSchema,
    validate_overfitting, validate_convergence
)


class TestDataSchemas:
    
    def test_valid_features(self):
        """Test schema accepts valid features."""
        valid_data = pd.DataFrame([{
            'heart_rate': 120.0,
            'oxygen_saturation': 96.5,
            'respiratory_rate': 35.0,
            'weight_grams': 2100.0,
            'temperature_celsius': 36.8
        }])
        
        # Should not raise
        validated = NICUFeatureSchema.validate(valid_data)
        assert len(validated) == 1
        
    def test_invalid_features(self):
        """Test schema rejects invalid features."""
        invalid_data = pd.DataFrame([{
            'heart_rate': 250.0,  # Too high
            'oxygen_saturation': 96.5,
            'respiratory_rate': 35.0,
            'weight_grams': 2100.0,
            'temperature_celsius': 36.8
        }])
        
        with pytest.raises(pa.errors.SchemaError):
            NICUFeatureSchema.validate(invalid_data)
            
    def test_complete_dataset_schema(self):
        """Test complete dataset schema validation."""
        data = pd.DataFrame([{
            'heart_rate': 120.0,
            'oxygen_saturation': 96.5,
            'respiratory_rate': 35.0,
            'weight_grams': 2100.0,
            'temperature_celsius': 36.8,
            'suitable_for_kangaroo_care': True,
            'patient_id': 'TEST-001'
        }])
        
        validated = NICUDatasetSchema.validate(data)
        assert len(validated) == 1


class TestModelValidationSchemas:
    
    def test_model_metrics_schema(self):
        """Test model metrics validation."""
        metrics = pd.DataFrame([{
            'accuracy': 0.92,
            'precision': 0.89,
            'recall': 0.94,
            'f1_score': 0.91,
            'auc_roc': 0.95
        }])
        
        validated = ModelMetricsSchema.validate(metrics)
        assert validated['accuracy'].iloc[0] == 0.92
        
    def test_training_metrics_schema(self):
        """Test training metrics validation."""
        metrics = pd.DataFrame([{
            'epoch': 0,
            'train_loss': 0.5,
            'val_loss': 0.6,
            'train_accuracy': 0.8,
            'val_accuracy': 0.75,
            'learning_rate': 0.001
        }])
        
        validated = TrainingMetricsSchema.validate(metrics)
        assert len(validated) == 1


class TestValidationFunctions:
    
    def test_validate_overfitting_pass(self):
        """Test overfitting check passes with good metrics."""
        metrics = pd.DataFrame([
            {'train_accuracy': 0.90, 'val_accuracy': 0.88, 'train_loss': 0.1, 'val_loss': 0.1, 'epoch': 0, 'learning_rate': 0.001},
            {'train_accuracy': 0.91, 'val_accuracy': 0.89, 'train_loss': 0.1, 'val_loss': 0.1, 'epoch': 1, 'learning_rate': 0.001},
            {'train_accuracy': 0.92, 'val_accuracy': 0.90, 'train_loss': 0.1, 'val_loss': 0.1, 'epoch': 2, 'learning_rate': 0.001},
            {'train_accuracy': 0.93, 'val_accuracy': 0.91, 'train_loss': 0.1, 'val_loss': 0.1, 'epoch': 3, 'learning_rate': 0.001},
            {'train_accuracy': 0.94, 'val_accuracy': 0.92, 'train_loss': 0.1, 'val_loss': 0.1, 'epoch': 4, 'learning_rate': 0.001}
        ])
        
        assert validate_overfitting(metrics, threshold=0.1) == True
        
    def test_validate_overfitting_fail(self):
        """Test overfitting check fails with large gap."""
        metrics = pd.DataFrame([
            {'train_accuracy': 0.95, 'val_accuracy': 0.80, 'train_loss': 0.1, 'val_loss': 0.1, 'epoch': 0, 'learning_rate': 0.001},
            {'train_accuracy': 0.96, 'val_accuracy': 0.79, 'train_loss': 0.1, 'val_loss': 0.1, 'epoch': 1, 'learning_rate': 0.001},
            {'train_accuracy': 0.97, 'val_accuracy': 0.78, 'train_loss': 0.1, 'val_loss': 0.1, 'epoch': 2, 'learning_rate': 0.001},
            {'train_accuracy': 0.98, 'val_accuracy': 0.77, 'train_loss': 0.1, 'val_loss': 0.1, 'epoch': 3, 'learning_rate': 0.001},
            {'train_accuracy': 0.99, 'val_accuracy': 0.76, 'train_loss': 0.1, 'val_loss': 0.1, 'epoch': 4, 'learning_rate': 0.001}
        ])
        
        assert validate_overfitting(metrics, threshold=0.1) == False