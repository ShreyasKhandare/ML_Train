import pandas as pd
import numpy as np
from typing import Dict, List, Any

class DataValidator:
    """
    Validates data quality before processing.
    
    Why validation?
    - Most of ML failures are due to bad data, not bad models
    - Silent failures: model trains but on garbage data
    - Validation catches issues early (fail fast principle)
    - Production requirement: data quality gates before training
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize validator with dataframe."""
        self.df = df
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def check_nulls(self, critical_cols: List[str] = None) -> None:
        """
        Check for missing values.
        
        Why nulls matter?
        - XGBoost can't handle NaN
        - Silent NaN handling → wrong predictions
        - Target column nulls = unlabeled data (useless)
        """
        null_counts = self.df.isnull().sum()
        
        if null_counts.sum() == 0:
            return
        
        null_cols = null_counts[null_counts > 0]
        msg = f"Missing values: {dict(null_cols)}"
        
        # Critical columns = fail immediately
        if critical_cols:
            if any(col in null_cols.index for col in critical_cols):
                self.errors.append(msg)
                return
        
        # Non-critical = warning
        self.warnings.append(msg)
    
    def check_duplicates(self) -> None:
        """
        Check for duplicate rows.
        
        Duplicates matter coz:
        - Exact duplicates = data quality issue
        - In train/test → data leakage (overfitting)
        - In production → indicates ingestion bug
        """
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            self.warnings.append(
                f"Found {dup_count} duplicate rows (may indicate data pipeline issue)"
            )
    
    def check_schema(self, expected_dtypes: Dict[str, str]) -> None:
        """
        Validate that columns have expected types.
        
        Args:
            expected_dtypes: {'col_name': 'int64', 'target': 'object', ...}
        
        Why schema validation?
        - Type mismatches → silent errors in preprocessing
        - String column treated as numeric → crashes downstream
        - Categorical column as numeric → wrong results
        - Production: schema contracts between teams
        """
        for col, expected_type in expected_dtypes.items():
            if col not in self.df.columns:
                self.errors.append(f"Missing column: {col}")
            elif str(self.df[col].dtype) != expected_type:
                self.warnings.append(
                    f"Column {col}: expected {expected_type}, got {self.df[col].dtype}"
                )
    
    def check_feature_distributions(self) -> None:
        """
        Quick check: do features have reasonable distributions?
        
        Why this?
        - All zeros = constant feature (no information)
        - All identical = data loading bug
        - Extreme outliers → data quality issue
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].std() == 0:
                self.warnings.append(
                    f"Column {col}: zero variance (constant value, no information)"
                )
            
            # Check for extreme skew
            if len(self.df[col].unique()) < 10 and col != 'target':
                self.warnings.append(
                    f"Column {col}: very few unique values ({len(self.df[col].unique())})"
                )
    
    def validate(self) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Returns:
            {'status': 'valid'|'invalid', 'errors': [...], 'warnings': [...]}
        
        Why return dict?
        - Caller can decide what to do (fail, warn, log)
        - Separation of concerns: validation logic ≠ error handling logic
        """
        # Run all checks
        self.check_nulls(critical_cols=['target'])
        self.check_duplicates()
        self.check_feature_distributions()
        
        # Fail if critical errors
        if self.errors:
            return {
                'status': 'invalid',
                'errors': self.errors,
                'warnings': self.warnings,
                'reason': 'Critical validation errors found'
            }
        
        return {
            'status': 'valid',
            'errors': self.errors,
            'warnings': self.warnings,
            'rows': len(self.df),
            'columns': list(self.df.columns)
        }