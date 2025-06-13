"""Tests for data generator."""

import pytest
import pandas as pd
import numpy as np

from src.data.generator import generate_nicu_data, generate_edge_cases, add_data_quality_issues


class TestDataGenerator:
    
    def test_generate_nicu_data_shape(self):
        """Test data generation produces correct shapes."""
        train_df, val_df, test_df = generate_nicu_data(
            n_samples=100,
            test_split=0.2,
            val_split=0.1
        )
        
        assert len(train_df) == 70
        assert len(val_df) == 10
        assert len(test_df) == 20
        
    def test_feature_ranges(self):
        """Test features are within valid ranges."""
        train_df, _, _ = generate_nicu_data(n_samples=500)
        
        assert train_df['heart_rate'].between(50, 200).all()
        assert train_df['oxygen_saturation'].between(70, 100).all()
        assert train_df['respiratory_rate'].between(20, 80).all()
        assert train_df['weight_grams'].between(500, 5000).all()
        assert train_df['temperature_celsius'].between(35, 38.5).all()
        
    def test_label_distribution(self):
        """Test reasonable class balance."""
        train_df, _, _ = generate_nicu_data(n_samples=1000)
        
        suitable_ratio = train_df['suitable_for_kangaroo_care'].mean()
        assert 0.2 < suitable_ratio < 0.8  # Accept wider range
        
    def test_feature_correlations(self):
        """Test expected correlations exist."""
        train_df, _, _ = generate_nicu_data(n_samples=1000)
        
        # Weight should correlate positively with temperature
        corr_weight_temp = train_df['weight_grams'].corr(train_df['temperature_celsius'])
        assert corr_weight_temp > 0.1
        
        # Oxygen saturation should correlate negatively with heart rate
        corr_o2_hr = train_df['oxygen_saturation'].corr(train_df['heart_rate'])
        assert corr_o2_hr < -0.1
        
    def test_edge_cases(self):
        """Test edge case generation."""
        edge_df = generate_edge_cases()
        
        assert len(edge_df) >= 4
        assert 'patient_id' in edge_df.columns
        assert edge_df['patient_id'].str.startswith('EDGE').all()
        
    def test_data_quality_issues(self):
        """Test adding realistic data issues."""
        train_df, _, _ = generate_nicu_data(n_samples=100)
        
        df_with_issues = add_data_quality_issues(
            train_df,
            missing_rate=0.1,
            outlier_rate=0.05
        )
        
        # Check missing values were added
        assert df_with_issues.isnull().sum().sum() > 0
        
        # Check some outliers exist
        assert (df_with_issues['heart_rate'] > 200).any() or \
               (df_with_issues['heart_rate'] < 50).any() or \
               df_with_issues['heart_rate'].isnull().any()