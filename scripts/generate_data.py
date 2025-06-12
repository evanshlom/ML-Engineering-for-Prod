#!/usr/bin/env python
"""Generate synthetic NICU data."""

import sys
sys.path.append('.')

from src.data.generator import generate_nicu_data, generate_edge_cases

if __name__ == "__main__":
    # Generate main datasets
    train_df, val_df, test_df = generate_nicu_data(
        n_samples=2000,
        test_split=0.2,
        val_split=0.1
    )
    
    # Save datasets
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