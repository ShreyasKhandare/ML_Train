import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import Dict, Tuple, Any

class ModelTrainer:
    """
    Trains XGBoost classifier.
    
    Why XGBoost?
    - Industry standard for tabular data (used by 95% of Kaggle winners)
    - Handles non-linear relationships
    - Feature importance (interpretability)
    - Fast (unlike neural networks)
    - Robust to outliers (unlike linear models)
    
    Why not alternatives?
    - Random Forest: Slower, less accurate
    - Linear Regression: Can't capture non-linearity
    - Neural Networks: Overkill, harder to tune, slower
    - Logistic Regression: Too simple for complex patterns
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize trainer with hyperparameters.
        
        Args:
            n_estimators: Number of boosting rounds (trees)
            max_depth: Maximum tree depth (prevents overfitting)
            learning_rate: Shrinkage rate (smaller = more robust)
            random_state: Seed for reproducibility
        
        Why these hyperparameters?
        - n_estimators=100: Reasonable default (100-500 is typical)
        - max_depth=5: Shallow trees reduce overfitting
        - learning_rate=0.1: Conservative learning (avoid wild swings)
        - random_state: Reproducible results (critical for science)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = None
        self.train_date = None
        self.hyperparameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate
        }
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train XGBoost model.
        
        Args:
            X: Feature matrix (preprocessed)
            y: Target vector
            test_size: Fraction for test set
            verbose: Print training info
        
        Returns:
            Dict with model, train/test splits, metadata
        
        Why return everything?
        - Caller has full context (train/test, dates, model)
        - Flexibility: can log, evaluate, save independently
        - No hidden state in trainer
        """
        # Split data: critical to do BEFORE preprocessing
        # (Otherwise test information leaks into training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y  # Keep class balance in both sets
        )
        
        if verbose:
            print(f"Training set: {X_train.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")
            print(f"Class distribution: {np.bincount(y_train)}")
        
        # Initialize and train model
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',  # Binary classification metric
            n_jobs=-1  # Use all CPU cores
        )
        
        # Train
        self.model.fit(X_train, y_train)
        self.train_date = datetime.now()
        
        if verbose:
            print(f"Model trained at {self.train_date}")
        
        return {
            'model': self.model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_date': self.train_date,
            'hyperparameters': self.hyperparameters
        }
    
    def save(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Why save?
        - Production: model needs to be deployed
        - Experimentation: compare different models
        - Version control: track model history
        - Can't store in git (too large)
        """
        if self.model is None:
            raise ValueError("No model trained yet. Call train() first.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load(filepath: str) -> xgb.XGBClassifier:
        """
        Load trained model from disk.
        
        Returns:
            Loaded XGBoost model ready for prediction
        """
        return joblib.load(filepath)