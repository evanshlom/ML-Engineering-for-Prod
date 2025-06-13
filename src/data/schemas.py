"""Pandera schemas for data validation in NICU kangaroo care prediction."""

import pandera as pa
from pandera.api.pandas.model import DataFrameModel
from pandera.typing import DataFrame
import pandas as pd
from typing import Optional


class NICUFeatureSchema(DataFrameModel):
    """Schema for validating NICU infant features."""
    
    heart_rate: float = pa.Field(
        ge=50.0, le=200.0,
        description="Heart rate in beats per minute",
        nullable=False
    )
    oxygen_saturation: float = pa.Field(
        ge=70.0, le=100.0,
        description="Oxygen saturation percentage",
        nullable=False
    )
    respiratory_rate: float = pa.Field(
        ge=20.0, le=80.0,
        description="Respiratory rate in breaths per minute",
        nullable=False
    )
    weight_grams: float = pa.Field(
        ge=500.0, le=5000.0,
        description="Infant weight in grams",
        nullable=False
    )
    temperature_celsius: float = pa.Field(
        ge=35.0, le=38.5,
        description="Body temperature in Celsius",
        nullable=False
    )
    
    class Config:
        name = "NICUFeatures"
        strict = True
        coerce = True


class NICULabelSchema(DataFrameModel):
    """Schema for kangaroo care readiness labels."""
    
    suitable_for_kangaroo_care: bool = pa.Field(
        description="Binary label indicating kangaroo care readiness",
        nullable=False
    )
    
    class Config:
        name = "NICULabels"
        strict = True


class NICUDatasetSchema(DataFrameModel):
    """Complete dataset schema combining features and labels."""
    
    # Features
    heart_rate: float = pa.Field(ge=50.0, le=200.0)
    oxygen_saturation: float = pa.Field(ge=70.0, le=100.0)
    respiratory_rate: float = pa.Field(ge=20.0, le=80.0)
    weight_grams: float = pa.Field(ge=500.0, le=5000.0)
    temperature_celsius: float = pa.Field(ge=35.0, le=38.5)
    
    # Label
    suitable_for_kangaroo_care: bool = pa.Field()
    
    # Optional metadata
    patient_id: Optional[str] = pa.Field(nullable=True)
    timestamp: Optional[pd.Timestamp] = pa.Field(nullable=True)
    
    class Config:
        name = "NICUDataset"
        strict = True
        coerce = True


# Custom checks for data quality
@pa.check_types
def validate_feature_correlations(df: DataFrame[NICUFeatureSchema]) -> DataFrame[NICUFeatureSchema]:
    """Validate expected correlations between features."""
    
    # Check that low oxygen saturation tends to correlate with higher heart rate
    low_oxygen_mask = df['oxygen_saturation'] < 90
    if low_oxygen_mask.any():
        avg_hr_low_oxygen = df.loc[low_oxygen_mask, 'heart_rate'].mean()
        avg_hr_normal_oxygen = df.loc[~low_oxygen_mask, 'heart_rate'].mean()
        
        if avg_hr_low_oxygen < avg_hr_normal_oxygen:
            import warnings
            warnings.warn(
                "Unexpected correlation: Low oxygen saturation should typically "
                "correlate with higher heart rate"
            )
    
    return df


@pa.check_types
def validate_stability_metrics(df: DataFrame[NICUDatasetSchema]) -> DataFrame[NICUDatasetSchema]:
    """Validate that suitable infants have more stable vital signs."""
    
    suitable_mask = df['suitable_for_kangaroo_care']
    
    # Calculate coefficient of variation for vital signs
    for feature in ['heart_rate', 'respiratory_rate', 'temperature_celsius']:
        cv_suitable = df.loc[suitable_mask, feature].std() / df.loc[suitable_mask, feature].mean()
        cv_unsuitable = df.loc[~suitable_mask, feature].std() / df.loc[~suitable_mask, feature].mean()
        
        if cv_suitable > cv_unsuitable * 1.5:
            import warnings
            warnings.warn(
                f"Warning: {feature} variability is higher in suitable infants, "
                "which may indicate data quality issues"
            )
    
    return df


# Model validation schemas
class ModelMetricsSchema(DataFrameModel):
    """Schema for model evaluation metrics."""
    
    accuracy: float = pa.Field(ge=0.0, le=1.0)
    precision: float = pa.Field(ge=0.0, le=1.0)
    recall: float = pa.Field(ge=0.0, le=1.0)
    f1_score: float = pa.Field(ge=0.0, le=1.0)
    auc_roc: float = pa.Field(ge=0.0, le=1.0)
    
    class Config:
        name = "ModelMetrics"
        strict = True


class TrainingMetricsSchema(DataFrameModel):
    """Schema for training progress metrics."""
    
    epoch: int = pa.Field(ge=0)
    train_loss: float = pa.Field(ge=0.0)
    val_loss: float = pa.Field(ge=0.0)
    train_accuracy: float = pa.Field(ge=0.0, le=1.0)
    val_accuracy: float = pa.Field(ge=0.0, le=1.0)
    learning_rate: float = pa.Field(ge=0.0)
    
    class Config:
        name = "TrainingMetrics"
        strict = True


# Validation functions for model behavior
def validate_overfitting(metrics_df: DataFrame[TrainingMetricsSchema], threshold: float = 0.1) -> bool:
    """Check if model is overfitting based on train/val gap."""
    
    last_epochs = metrics_df.tail(5)
    avg_train_acc = last_epochs['train_accuracy'].mean()
    avg_val_acc = last_epochs['val_accuracy'].mean()
    
    gap = avg_train_acc - avg_val_acc
    
    if gap > threshold:
        import warnings
        warnings.warn(
            f"Potential overfitting detected: Train-Val accuracy gap = {gap:.3f} > {threshold}"
        )
        return False
    
    return True


def validate_convergence(metrics_df: DataFrame[TrainingMetricsSchema], window: int = 5) -> bool:
    """Check if model has converged based on loss plateau."""
    
    if len(metrics_df) < window * 2:
        return False
    
    recent_losses = metrics_df['val_loss'].tail(window).values
    previous_losses = metrics_df['val_loss'].tail(window * 2).head(window).values
    
    # Check if loss has plateaued
    loss_change = abs(recent_losses.mean() - previous_losses.mean())
    relative_change = loss_change / previous_losses.mean()
    
    converged = relative_change < 0.01  # Less than 1% change
    
    if converged:
        print(f"Model appears to have converged (relative change: {relative_change:.4f})")
    
    return converged