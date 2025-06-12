"""Model evaluation utilities."""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def evaluate_model(model, test_loader, device='cpu'):
    """Comprehensive model evaluation."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=['Not Ready', 'Ready'])
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'classification_report': report,
        'confusion_matrix': cm
    }