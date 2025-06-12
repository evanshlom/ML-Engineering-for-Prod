"""Generate synthetic NICU data for demo purposes."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime, timedelta
import random


def generate_nicu_data(
    n_samples: int = 1000,
    test_split: float = 0.2,
    val_split: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic NICU patient data with realistic correlations.
    
    Args:
        n_samples: Total number of samples
        test_split: Fraction for test set
        val_split: Fraction for validation set
        random_state: Random seed
    
    Returns:
        train_df, val_df, test_df
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Generate base features with correlations
    data = []
    
    for i in range(n_samples):
        # Weight as base feature (500-5000g)
        weight = np.random.normal(2200, 800)
        weight = np.clip(weight, 500, 5000)
        
        # Temperature correlates with weight (heavier babies regulate better)
        temp_base = 36.5 + (weight - 1500) / 3000 * 0.5
        temperature = np.random.normal(temp_base, 0.3)
        temperature = np.clip(temperature, 35.0, 38.5)
        
        # Oxygen saturation (higher weight = better saturation)
        o2_base = 88 + (weight - 500) / 4500 * 10
        oxygen_saturation = np.random.normal(o2_base, 3)
        oxygen_saturation = np.clip(oxygen_saturation, 70, 100)
        
        # Heart rate (inversely related to oxygen saturation)
        hr_base = 150 - (oxygen_saturation - 85) * 1.5
        heart_rate = np.random.normal(hr_base, 15)
        heart_rate = np.clip(heart_rate, 50, 200)
        
        # Respiratory rate (correlates with oxygen saturation)
        rr_base = 50 - (oxygen_saturation - 85) * 0.8
        respiratory_rate = np.random.normal(rr_base, 8)
        respiratory_rate = np.clip(respiratory_rate, 20, 80)
        
        # Determine suitability based on thresholds and stability
        suitable = (
            (weight > 1500) and
            (temperature >= 36.0 and temperature <= 37.5) and
            (oxygen_saturation >= 92) and
            (heart_rate >= 100 and heart_rate <= 140) and
            (respiratory_rate >= 25 and respiratory_rate <= 45)
        )
        
        # Add some noise to labels (90% accuracy)
        if np.random.random() < 0.1:
            suitable = not suitable
        
        # Add timestamp
        base_time = datetime.now() - timedelta(days=365)
        timestamp = base_time + timedelta(
            days=random.randint(0, 365),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        data.append({
            'patient_id': f'NICU-{i:05d}',
            'timestamp': timestamp,
            'heart_rate': round(heart_rate, 1),
            'oxygen_saturation': round(oxygen_saturation, 1),
            'respiratory_rate': round(respiratory_rate, 1),
            'weight_grams': round(weight, 1),
            'temperature_celsius': round(temperature, 2),
            'suitable_for_kangaroo_care': suitable
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split data
    n_test = int(n_samples * test_split)
    n_val = int(n_samples * val_split)
    
    test_df = df[:n_test]
    val_df = df[n_test:n_test + n_val]
    train_df = df[n_test + n_val:]
    
    return train_df, val_df, test_df


def add_data_quality_issues(
    df: pd.DataFrame,
    missing_rate: float = 0.02,
    outlier_rate: float = 0.01
) -> pd.DataFrame:
    """Add realistic data quality issues for testing robustness.
    
    Args:
        df: Input DataFrame
        missing_rate: Fraction of values to make missing
        outlier_rate: Fraction of outliers to introduce
    
    Returns:
        DataFrame with quality issues
    """
    df_copy = df.copy()
    
    # Add missing values
    for col in ['heart_rate', 'oxygen_saturation', 'respiratory_rate', 'temperature_celsius']:
        mask = np.random.random(len(df_copy)) < missing_rate
        df_copy.loc[mask, col] = np.nan
    
    # Add outliers
    n_outliers = int(len(df_copy) * outlier_rate)
    outlier_indices = np.random.choice(len(df_copy), n_outliers, replace=False)
    
    for idx in outlier_indices:
        # Randomly select a feature to make outlier
        feature = np.random.choice(['heart_rate', 'oxygen_saturation', 'respiratory_rate'])
        
        if feature == 'heart_rate':
            df_copy.loc[idx, feature] = np.random.choice([45, 210])
        elif feature == 'oxygen_saturation':
            df_copy.loc[idx, feature] = np.random.choice([65, 102])
        else:  # respiratory_rate
            df_copy.loc[idx, feature] = np.random.choice([15, 85])
    
    return df_copy


def generate_edge_cases() -> pd.DataFrame:
    """Generate edge cases for testing."""
    edge_cases = [
        # Borderline suitable cases
        {
            'patient_id': 'EDGE-001',
            'heart_rate': 100.0,  # Lower boundary
            'oxygen_saturation': 92.0,  # Lower boundary
            'respiratory_rate': 25.0,  # Lower boundary
            'weight_grams': 1500.0,  # Lower boundary
            'temperature_celsius': 36.0,  # Lower boundary
            'suitable_for_kangaroo_care': True
        },
        # Borderline unsuitable cases
        {
            'patient_id': 'EDGE-002',
            'heart_rate': 99.0,  # Just below suitable
            'oxygen_saturation': 91.5,  # Just below suitable
            'respiratory_rate': 24.0,  # Just below suitable
            'weight_grams': 1499.0,  # Just below suitable
            'temperature_celsius': 35.9,  # Just below suitable
            'suitable_for_kangaroo_care': False
        },
        # Extreme values but still valid
        {
            'patient_id': 'EDGE-003',
            'heart_rate': 190.0,  # Very high
            'oxygen_saturation': 100.0,  # Perfect
            'respiratory_rate': 70.0,  # Very high
            'weight_grams': 4500.0,  # Very heavy
            'temperature_celsius': 38.0,  # High fever
            'suitable_for_kangaroo_care': False
        },
        # Mixed signals
        {
            'patient_id': 'EDGE-004',
            'heart_rate': 120.0,  # Good
            'oxygen_saturation': 98.0,  # Excellent
            'respiratory_rate': 35.0,  # Good
            'weight_grams': 800.0,  # Too low
            'temperature_celsius': 36.5,  # Good
            'suitable_for_kangaroo_care': False
        }
    ]
    
    return pd.DataFrame(edge_cases)


if __name__ == "__main__":
    # Generate datasets
    train_df, val_df, test_df = generate_nicu_data(
        n_samples=2000,
        test_split=0.2,
        val_split=0.1
    )
    
    # Save to CSV
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    # Generate edge cases
    edge_df = generate_edge_cases()
    edge_df.to_csv('data/processed/edge_cases.csv', index=False)
    
    print(f"Generated datasets:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print(f"  Edge cases: {len(edge_df)} samples")
    
    # Show class distribution
    print(f"\nClass distribution:")
    print(f"  Train: {train_df['suitable_for_kangaroo_care'].value_counts().to_dict()}")
    print(f"  Val: {val_df['suitable_for_kangaroo_care'].value_counts().to_dict()}")
    print(f"  Test: {test_df['suitable_for_kangaroo_care'].value_counts().to_dict()}")