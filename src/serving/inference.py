"""Inference utilities."""

import torch
import numpy as np
from typing import Dict, List


def preprocess_features(features_dict: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to numpy array in correct order."""
    feature_order = [
        'heart_rate',
        'oxygen_saturation', 
        'respiratory_rate',
        'weight_grams',
        'temperature_celsius'
    ]
    
    return np.array([features_dict[f] for f in feature_order])


def postprocess_prediction(
    probability: float,
    threshold: float = 0.5
) -> Dict[str, any]:
    """Process raw model output into structured response."""
    prediction = int(probability >= threshold)
    
    # Determine confidence level
    if probability > 0.85 or probability < 0.15:
        confidence = 'high'
    elif probability > 0.7 or probability < 0.3:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return {
        'prediction': prediction,
        'probability': probability,
        'confidence': confidence
    }