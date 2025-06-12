"""Data preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import Tuple


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize features to 0-1 range."""
    feature_cols = ['heart_rate', 'oxygen_saturation', 'respiratory_rate', 
                   'weight_grams', 'temperature_celsius']
    
    df_normalized = df.copy()
    
    # Define normalization ranges
    ranges = {
        'heart_rate': (50, 200),
        'oxygen_saturation': (70, 100),
        'respiratory_rate': (20, 80),
        'weight_grams': (500, 5000),
        'temperature_celsius': (35, 38.5)
    }
    
    for col in feature_cols:
        min_val, max_val = ranges[col]
        df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_normalized