"""Shared test fixtures."""

import pytest
import pandas as pd
import torch
from pathlib import Path

from src.models.architecture import NICUReadinessClassifier
from src.data.generator import generate_nicu_data


@pytest.fixture
def sample_data():
    """Generate small sample dataset."""
    train_df, val_df, test_df = generate_nicu_data(n_samples=100)
    return train_df, val_df, test_df


@pytest.fixture
def model():
    """Create model instance."""
    return NICUReadinessClassifier()


@pytest.fixture
def valid_input():
    """Valid model input tensor."""
    return torch.randn(1, 5)