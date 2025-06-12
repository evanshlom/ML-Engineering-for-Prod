"""PyTorch model architecture for NICU kangaroo care readiness prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class NICUReadinessClassifier(nn.Module):
    """Binary classifier for kangaroo care readiness prediction.
    
    Architecture designed with:
    - Batch normalization for stable training
    - Dropout for regularization
    - Skip connections for gradient flow
    - Flexible hidden layer sizes
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: List[int] = [64, 32, 16],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = "relu"
    ):
        """Initialize the classifier.
        
        Args:
            input_dim: Number of input features (default 5: HR, O2, RR, Weight, Temp)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'elu')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Select activation function
        self.activation = self._get_activation(activation)
        
        # Build layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Skip connection from input to final hidden layer
        self.skip_connection = nn.Linear(input_dim, hidden_dims[-1])
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, 1)
        """
        # Store input for skip connection
        identity = x
        
        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropouts[i](x)
            
            # Add skip connection at the last hidden layer
            if i == len(self.layers) - 1:
                skip = self.skip_connection(identity)
                x = x + skip
        
        # Output layer (no activation - using BCEWithLogitsLoss)
        output = self.output_layer(x)
        
        return output
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Probabilities of positive class
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions.
        
        Args:
            x: Input tensor
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0 or 1)
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).float()


class ModelWithExplainability(nn.Module):
    """Wrapper for the classifier with built-in interpretability features."""
    
    def __init__(self, base_model: NICUReadinessClassifier):
        super().__init__()
        self.base_model = base_model
        self.feature_names = [
            'heart_rate',
            'oxygen_saturation', 
            'respiratory_rate',
            'weight_grams',
            'temperature_celsius'
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
    
    def get_feature_importance(self, x: torch.Tensor) -> Dict[str, float]:
        """Calculate feature importance using gradient-based method.
        
        Args:
            x: Input tensor (single sample)
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        x = x.clone().requires_grad_(True)
        
        # Get model output
        output = self.base_model(x)
        
        # Calculate gradients
        self.base_model.zero_grad()
        output.backward()
        
        # Get absolute gradients as importance
        importances = x.grad.abs().squeeze().cpu().numpy()
        
        # Normalize
        importances = importances / importances.sum()
        
        return dict(zip(self.feature_names, importances))
    
    def get_decision_boundary_sample(
        self, 
        x: torch.Tensor, 
        feature_idx: int, 
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get decision boundary by varying one feature.
        
        Args:
            x: Base input sample
            feature_idx: Index of feature to vary
            n_points: Number of points to sample
            
        Returns:
            Tuple of (feature_values, predictions)
        """
        with torch.no_grad():
            # Create range of values for the feature
            feature_min = {
                0: 50.0,   # heart_rate
                1: 70.0,   # oxygen_saturation
                2: 20.0,   # respiratory_rate
                3: 500.0,  # weight_grams
                4: 35.0    # temperature_celsius
            }
            feature_max = {
                0: 200.0,  # heart_rate
                1: 100.0,  # oxygen_saturation
                2: 80.0,   # respiratory_rate
                3: 5000.0, # weight_grams
                4: 38.5    # temperature_celsius
            }
            
            values = np.linspace(
                feature_min[feature_idx],
                feature_max[feature_idx],
                n_points
            )
            
            predictions = []
            for val in values:
                x_copy = x.clone()
                x_copy[0, feature_idx] = val
                pred = self.base_model.predict_proba(x_copy).item()
                predictions.append(pred)
            
            return values, np.array(predictions)


def create_model(config: Dict) -> NICUReadinessClassifier:
    """Factory function to create model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    return NICUReadinessClassifier(
        input_dim=config.get('input_dim', 5),
        hidden_dims=config.get('hidden_dims', [64, 32, 16]),
        dropout_rate=config.get('dropout_rate', 0.3),
        use_batch_norm=config.get('use_batch_norm', True),
        activation=config.get('activation', 'relu')
    )