"""Tests for inference utilities."""

import pytest
import numpy as np

from src.serving.inference import preprocess_features, postprocess_prediction


class TestInference:
    
    def test_preprocess_features(self):
        """Test feature preprocessing."""
        features = {
            'heart_rate': 120.0,
            'oxygen_saturation': 96.5,
            'respiratory_rate': 35.0,
            'weight_grams': 2100.0,
            'temperature_celsius': 36.8
        }
        
        array = preprocess_features(features)
        
        assert array.shape == (5,)
        assert array[0] == 120.0
        assert array[1] == 96.5
        
    def test_postprocess_prediction_high_confidence(self):
        """Test postprocessing with high confidence."""
        result = postprocess_prediction(0.95)
        
        assert result['prediction'] == 1
        assert result['probability'] == 0.95
        assert result['confidence'] == 'high'
        
    def test_postprocess_prediction_low_confidence(self):
        """Test postprocessing with low confidence."""
        result = postprocess_prediction(0.45)
        
        assert result['prediction'] == 0
        assert result['probability'] == 0.45
        assert result['confidence'] == 'low'