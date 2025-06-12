"""PyTest unit tests for model architecture."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List

from src.models.architecture import (
    NICUReadinessClassifier, 
    ModelWithExplainability,
    create_model
)


class TestNICUReadinessClassifier:
    """Test suite for the main classifier."""
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(32, 5)  # batch_size=32, features=5
    
    @pytest.fixture
    def single_input(self):
        """Create single sample input."""
        return torch.tensor([[120.0, 95.0, 35.0, 2000.0, 36.5]])
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return NICUReadinessClassifier(
            input_dim=5,
            hidden_dims=[64, 32, 16],
            dropout_rate=0.3
        )
    
    def test_model_initialization(self):
        """Test model can be initialized with various configurations."""
        # Default initialization
        model1 = NICUReadinessClassifier()
        assert model1.input_dim == 5
        assert model1.hidden_dims == [64, 32, 16]
        
        # Custom initialization
        model2 = NICUReadinessClassifier(
            input_dim=5,
            hidden_dims=[128, 64],
            dropout_rate=0.5,
            use_batch_norm=False,
            activation='leaky_relu'
        )
        assert len(model2.layers) == 2
        assert isinstance(model2.activation, nn.LeakyReLU)
    
    def test_forward_pass_shape(self, model, sample_input):
        """Test forward pass produces correct output shape."""
        output = model(sample_input)
        assert output.shape == (32, 1)
        assert output.dtype == torch.float32
    
    def test_forward_pass_single_sample(self, model, single_input):
        """Test forward pass with single sample."""
        output = model(single_input)
        assert output.shape == (1, 1)
    
    def test_predict_proba(self, model, sample_input):
        """Test probability prediction method."""
        probs = model.predict_proba(sample_input)
        assert probs.shape == (32, 1)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
    
    def test_predict_binary(self, model, sample_input):
        """Test binary prediction method."""
        predictions = model.predict(sample_input, threshold=0.5)
        assert predictions.shape == (32, 1)
        assert torch.all((predictions == 0) | (predictions == 1))
    
    def test_gradient_flow(self, model, sample_input):
        """Test gradients flow properly through the network."""
        model.train()
        output = model(sample_input)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist and are non-zero
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.all(param.grad == 0), f"Zero gradients for {name}"
    
    def test_skip_connection(self, model, sample_input):
        """Test skip connection is working."""
        # Get intermediate activations
        x = sample_input
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if model.use_batch_norm and i < len(model.batch_norms):
                x = model.batch_norms[i](x)
            x = model.activation(x)
            x = model.dropouts[i](x)
        
        # Check skip connection dimensions
        skip_out = model.skip_connection(sample_input)
        assert skip_out.shape == x.shape
    
    @pytest.mark.parametrize("batch_size", [1, 16, 64, 128])
    def test_variable_batch_sizes(self, model, batch_size):
        """Test model handles various batch sizes."""
        input_tensor = torch.randn(batch_size, 5)
        output = model(input_tensor)
        assert output.shape == (batch_size, 1)
    
    def test_dropout_behavior(self, model, sample_input):
        """Test dropout behaves correctly in train/eval modes."""
        model.train()
        outputs_train = [model(sample_input) for _ in range(10)]
        
        model.eval()
        outputs_eval = [model(sample_input) for _ in range(10)]
        
        # Training outputs should vary due to dropout
        train_std = torch.stack(outputs_train).std(dim=0).mean()
        eval_std = torch.stack(outputs_eval).std(dim=0).mean()
        
        assert train_std > eval_std  # More variation in training
    
    def test_weight_initialization(self):
        """Test weights are properly initialized."""
        model = NICUReadinessClassifier()
        
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                # Check weights are not all zeros or ones
                assert not torch.all(param == 0)
                assert not torch.all(param == 1)
                
                # Check reasonable initialization scale
                assert param.std() > 0.01 and param.std() < 1.0


class TestModelWithExplainability:
    """Test suite for explainability wrapper."""
    
    @pytest.fixture
    def base_model(self):
        """Create base model."""
        return NICUReadinessClassifier()
    
    @pytest.fixture
    def explainable_model(self, base_model):
        """Create explainable model."""
        return ModelWithExplainability(base_model)
    
    @pytest.fixture
    def sample_features(self):
        """Create realistic sample features."""
        return torch.tensor([[
            120.0,  # heart_rate
            95.0,   # oxygen_saturation
            35.0,   # respiratory_rate
            2000.0, # weight_grams
            36.5    # temperature_celsius
        ]])
    
    def test_feature_importance(self, explainable_model, sample_features):
        """Test feature importance calculation."""
        importance = explainable_model.get_feature_importance(sample_features)
        
        # Check output format
        assert isinstance(importance, dict)
        assert len(importance) == 5
        assert all(feature in importance for feature in [
            'heart_rate', 'oxygen_saturation', 'respiratory_rate',
            'weight_grams', 'temperature_celsius'
        ])
        
        # Check normalization
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 1e-5
        
        # Check all values are non-negative
        assert all(val >= 0 for val in importance.values())
    
    def test_decision_boundary_sample(self, explainable_model, sample_features):
        """Test decision boundary visualization."""
        # Test varying heart rate
        values, predictions = explainable_model.get_decision_boundary_sample(
            sample_features, 
            feature_idx=0,  # heart_rate
            n_points=50
        )
        
        assert len(values) == 50
        assert len(predictions) == 50
        assert values.min() >= 50.0 and values.max() <= 200.0
        assert all(0 <= p <= 1 for p in predictions)
    
    def test_forward_compatibility(self, explainable_model, sample_features):
        """Test that wrapper maintains forward pass compatibility."""
        base_model = explainable_model.base_model
        
        # Compare outputs
        wrapper_output = explainable_model(sample_features)
        base_output = base_model(sample_features)
        
        assert torch.allclose(wrapper_output, base_output)


class TestModelFactory:
    """Test model creation from config."""
    
    def test_create_model_default(self):
        """Test creating model with default config."""
        config = {}
        model = create_model(config)
        
        assert isinstance(model, NICUReadinessClassifier)
        assert model.input_dim == 5
    
    def test_create_model_custom(self):
        """Test creating model with custom config."""
        config = {
            'input_dim': 5,
            'hidden_dims': [128, 64, 32],
            'dropout_rate': 0.5,
            'use_batch_norm': False,
            'activation': 'elu'
        }
        model = create_model(config)
        
        assert len(model.layers) == 3
        assert isinstance(model.activation, nn.ELU)
        assert len(model.batch_norms) == 0  # No batch norm


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for end-to-end model behavior."""
    
    def test_training_step(self):
        """Test a complete training step."""
        model = NICUReadinessClassifier()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        # Generate batch
        inputs = torch.randn(16, 5)
        targets = torch.randint(0, 2, (16, 1)).float()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_model_device_transfer(self):
        """Test model can be transferred between devices."""
        model = NICUReadinessClassifier()
        input_tensor = torch.randn(8, 5)
        
        # Test CPU
        model.cpu()
        output_cpu = model(input_tensor)
        assert output_cpu.device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model.cuda()
            input_cuda = input_tensor.cuda()
            output_cuda = model(input_cuda)
            assert output_cuda.device.type == 'cuda'