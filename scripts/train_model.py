#!/usr/bin/env python
"""Train NICU model."""

import sys
sys.path.append('.')

from src.models.training import train_model
from src.data.generator import generate_nicu_data
import yaml
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    # Load config
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if data exists, generate if not
    data_path = Path('data/processed')
    if not (data_path / 'train.csv').exists():
        print("Generating training data...")
        train_df, val_df, test_df = generate_nicu_data(n_samples=2000)
        train_df.to_csv(data_path / 'train.csv', index=False)
        val_df.to_csv(data_path / 'val.csv', index=False)
        test_df.to_csv(data_path / 'test.csv', index=False)
    
    # Load data
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Train model
    model, metadata = train_model(config, train_df, val_df, test_df)
    print(f"Training completed. Metrics: {metadata['metrics']}")