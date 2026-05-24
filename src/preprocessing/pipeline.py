import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from typing import Tuple

class PreprocessingPipeline:
    """
    Feature engineering: scaling + encoding.
    
    Why a preprocessing class?
    - In production: fit on train, save, apply to production data
    - Prevents data leakage: never fit on test data
    - Reproducible: same preprocessing pipeline always
    - Reusable: deploy preprocessor independently of model
    """
    
    def __init__(
        self,
        numeric_features: list = None,
        categorical_features: list = None
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            numeric_features: List of numeric column names
            categorical_features: List of categorical column names
        
        Why separate numeric vs categorical?
        - Numeric: StandardScaler (mean=0, std=1)
        - Categorical: OneHotEncoder (convert to binary features)
        - Different transformations for different data types
        """
        # Default features for our project
        self.numeric_features = numeric_features or ['feature_1', 'feature_2']
        self.categorical_features = categorical_features or ['feature_3']
        
        # Build transformer pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                # Scale numeric features to mean=0, std=1
                ('num_scaler', StandardScaler(), self.numeric_features),
                
                # Encode categorical features to binary (one-hot)
                ('cat_encoder', OneHotEncoder(
                    drop='first',  # Drop first category (avoid multicollinearity)
                    sparse_output=False,  # Return dense array (faster)
                    handle_unknown='ignore'  # Unknown categories → zeros
                ), self.categorical_features)
            ],
            remainder='drop'  # Drop unknown columns
        )
    
    def fit(self, X: pd.DataFrame) -> 'PreprocessingPipeline':
        """
        Fit preprocessor on training data.
        
        CRITICAL: Always fit only on training data, never on test!
        
        Why?
        - Scaler computes mean/std from training data
        - If you fit on test: test distribution leaks into training
        - Model learns test distribution → overfits
        - In production: scaler always uses training statistics
        """
        self.preprocessor.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply preprocessing to new data.
        
        Why separate fit and transform?
        - Fit on train (learns statistics)
        - Transform on train (same statistics)
        - Transform on test (same statistics as train)
        - This prevents data leakage
        """
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Only use for training data (where fit_transform is appropriate).
        """
        return self.preprocessor.fit_transform(X)
    
    def get_feature_names(self) -> list:
        """
        Get names of output features after preprocessing.
        
        Why?
        - After encoding, we have new column names
        - Good for debugging: 'feature_3_A', 'feature_3_B', etc.
        - Essential for feature importance in models
        """
        try:
            names = self.preprocessor.get_feature_names_out().tolist()
            return names
        except:
            # Fallback if names not available
            n_numeric = len(self.numeric_features)
            n_categorical = len(self.categorical_features)
            # Rough estimate of output features
            return [f"feature_{i}" for i in range(n_numeric + n_categorical + 1)]
    
    def save(self, filepath: str) -> None:
        """
        Save preprocessor to disk for production use.
        
        Why save?
        - In production: new data arrives
        - Load preprocessor, apply same transformation
        - Ensures consistency: no manual transform bugs
        - Standard MLOps practice
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, filepath)
    
    @staticmethod
    def load(filepath: str) -> 'PreprocessingPipeline':
        """
        Load preprocessor from disk.
        
        Returns:
            Loaded preprocessor ready to transform new data
        """
        preprocessor = joblib.load(filepath)
        # Wrap in our class
        pipeline = PreprocessingPipeline()
        pipeline.preprocessor = preprocessor
        return pipeline