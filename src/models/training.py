"""Training pipeline for NICU kangaroo care model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import structlog
from pathlib import Path
import yaml
import json
from datetime import datetime

from .architecture import NICUReadinessClassifier
from ..data.schemas import TrainingMetricsSchema, validate_overfitting, validate_convergence

logger = structlog.get_logger()


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def prepare_data(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert DataFrame to tensors."""
    feature_cols = ['heart_rate', 'oxygen_saturation', 'respiratory_rate', 
                   'weight_grams', 'temperature_celsius']
    
    X = torch.FloatTensor(df[feature_cols].values)
    y = torch.FloatTensor(df['suitable_for_kangaroo_care'].values.reshape(-1, 1))
    
    return X, y


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders."""
    X_train, y_train = prepare_data(train_df)
    X_val, y_val = prepare_data(val_df)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predictions = (probs >= 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_targets, all_predictions),
        'precision': precision_score(all_targets, all_predictions, zero_division=0),
        'recall': recall_score(all_targets, all_predictions, zero_division=0),
        'f1_score': f1_score(all_targets, all_predictions, zero_division=0),
        'auc_roc': roc_auc_score(all_targets, all_probs)
    }
    
    return avg_loss, metrics['accuracy'], metrics


def train_model(
    config: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    save_dir: str = 'data/models'
) -> Tuple[NICUReadinessClassifier, Dict]:
    """Complete training pipeline."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = NICUReadinessClassifier(**config['model']).to(device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, 
        batch_size=config['training']['batch_size']
    )
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0001)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training'].get('early_stopping_patience', 15)
    )
    
    # Training metrics storage
    training_history = []
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_metrics = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        training_history.append(epoch_metrics)
        
        logger.info(
            f"Epoch {epoch+1}/{config['training']['epochs']} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_metrics = val_metrics.copy()
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Validate training quality
    history_df = pd.DataFrame(training_history)
    TrainingMetricsSchema.validate(history_df)
    
    overfitting_ok = validate_overfitting(history_df)
    converged = validate_convergence(history_df)
    
    logger.info(f"Training quality - Overfitting check: {overfitting_ok}, Converged: {converged}")
    
    # Test set evaluation
    if test_df is not None:
        test_loader = DataLoader(
            TensorDataset(*prepare_data(test_df)),
            batch_size=config['training']['batch_size'],
            shuffle=False
        )
        test_loss, test_acc, test_metrics = evaluate(
            model, test_loader, criterion, device
        )
        logger.info(f"Test set metrics: {test_metrics}")
    else:
        test_metrics = None
    
    # Save model and metadata
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = save_path / f"nicu_model_{timestamp}.pt"
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_metrics': best_metrics,
        'test_metrics': test_metrics,
        'training_history': training_history,
        'timestamp': timestamp
    }, model_path)
    
    # Save metadata
    metadata = {
        'model_version': f"1.0.{timestamp}",
        'training_date': datetime.now().isoformat(),
        'metrics': test_metrics if test_metrics else best_metrics,
        'hyperparameters': config['model'],
        'training_config': config['training'],
        'model_path': str(model_path)
    }
    
    with open(save_path / f"model_metadata_{timestamp}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    
    return model, metadata


if __name__ == "__main__":
    # Load config
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Train model
    model, metadata = train_model(config, train_df, val_df, test_df)
    
    print(f"Training completed. Best metrics: {metadata['metrics']}")