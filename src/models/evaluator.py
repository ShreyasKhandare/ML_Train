import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from scipy.stats import ks_2samp
from datetime import datetime
from typing import Dict, Any, Tuple

class ModelEvaluator:
    """
    Evaluates model performance and detects drift.
    
    Why evaluation matters?
    - Training accuracy ≠ real-world performance
    - Need multiple metrics (accuracy alone lies)
    - Drift detection = continuous monitoring
    - Production requirement: track everything
    """
    
    def __init__(self, model):
        """Initialize evaluator with trained model."""
        self.model = model
        self.metrics = {}
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dict with accuracy, precision, recall, F1, AUC
        
        Why multiple metrics?
        - Accuracy: overall correctness (but class imbalance confuses it)
        - Precision: of predicted positives, how many are correct? (false alarm rate)
        - Recall: of actual positives, how many did we find? (missed cases)
        - F1: harmonic mean of precision/recall (single number)
        - AUC: area under ROC curve (threshold-independent, best for ranking)
        
        Example:
        - Fraud detection: recall matters (catch fraud)
        - Spam filter: precision matters (avoid false positives)
        - This project: balanced (F1 + AUC)
        """
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'timestamp': datetime.now(),
            'samples_tested': len(y_test)
        }
        
        return self.metrics
    
    def detect_drift(
        self,
        X_train_baseline: np.ndarray,
        X_new_data: np.ndarray,
        p_value_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect data drift using KS test.
        
        Args:
            X_train_baseline: Original training data (reference distribution)
            X_new_data: New incoming data (test distribution)
            p_value_threshold: Statistical significance level
        
        Returns:
            Dict with drift detection results
        
        Why KS test?
        - Kolmogorov-Smirnov test compares two distributions
        - p-value < 0.05 → distributions significantly different
        - Fast: O(n) time complexity
        - No parameters to tune
        - Standard in MLOps
        
        Why drift detection?
        - Models trained on old distribution
        - Real world changes (user behavior, seasonality, bugs)
        - Without detection: silent degradation
        - With detection: alert + retrain
        
        Example:
        - Train on 2024 data (mean age 35)
        - 2026 data arrives (mean age 42)
        - KS test detects difference → alert
        """
        drift_results = {
            'timestamp': datetime.now(),
            'tests': {},
            'drift_detected': False,
            'action': 'continue'
        }
        
        # Check each feature
        for feature_idx in range(X_train_baseline.shape[1]):
            baseline_feature = X_train_baseline[:, feature_idx]
            new_feature = X_new_data[:, feature_idx]
            
            # KS test
            ks_statistic, p_value = ks_2samp(baseline_feature, new_feature)
            
            drift_results['tests'][f'feature_{feature_idx}'] = {
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'drift': p_value < p_value_threshold
            }
            
            # If any feature drifts significantly
            if p_value < p_value_threshold:
                drift_results['drift_detected'] = True
        
        # Decision
        if drift_results['drift_detected']:
            drift_results['action'] = 'retrain'
            drift_results['reason'] = f"Significant drift detected in {len([t for t in drift_results['tests'].values() if t['drift']])} features"
        
        return drift_results
    
    def detect_prediction_drift(
        self,
        y_pred_baseline: np.ndarray,
        y_pred_new: np.ndarray,
        p_value_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect drift in model predictions.
        
        Args:
            y_pred_baseline: Predictions on original test set
            y_pred_new: Predictions on new data
            p_value_threshold: Statistical threshold
        
        Returns:
            Drift detection results for predictions
        
        Why predict drift?
        - Different than data drift
        - Example: model predicts 90% class 0, suddenly 20%
        - Indicates either data or concept drift
        - Quicker signal than data drift alone
        """
        ks_stat, p_value = ks_2samp(y_pred_baseline, y_pred_new)
        
        return {
            'prediction_ks_statistic': float(ks_stat),
            'prediction_p_value': float(p_value),
            'prediction_drift_detected': p_value < p_value_threshold,
            'baseline_mean_prediction': float(y_pred_baseline.mean()),
            'new_mean_prediction': float(y_pred_new.mean())
        }