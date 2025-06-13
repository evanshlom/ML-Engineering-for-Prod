"""Configuration utilities."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.environ.get('CONFIG_PATH', 'configs/training_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Mock model info for API demo - use strings for dates
    config['model_version'] = '1.0.0'
    config['training_date'] = '2024-01-01T00:00:00'
    config['metrics'] = {
        'accuracy': 0.92,
        'precision': 0.89,
        'recall': 0.94,
        'f1_score': 0.91,
        'auc_roc': 0.95,
        'threshold': 0.5,
        'evaluation_date': '2024-01-01T00:00:00',
        'test_set_size': 400
    }
    
    return config