import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Optional

class DataLoader:
    """
    Loads data from multiple sources (CSV files and simulated APIs).
    
    Design:
    - Abstraction layer separates data loading logic from pipeline
    - Easier to swap sources (CSV → PostgreSQL → Kafka) without changing workflow
    - Single source of truth for data ingestion
    """
    
    def __init__(self, raw_data_dir: str = "data/raw"):
        """Initialize loader with raw data directory."""
        self.raw_dir = Path(raw_data_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load CSV file from raw data directory.
        
        Args:
            filename: Name of CSV file (e.g., 'data.csv')
        
        Returns:
            DataFrame with data
        
        W do this to
        - Centralized CSV loading prevents magic strings scattered everywhere
        - Raises error if file missing (fail fast, not silently)
        - Standard in production (data → raw/ → processed/)
        """
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found at {path}. "
                f"Place your CSV in {self.raw_dir}/"
            )
        
        df = pd.read_csv(path)
        return df
    
    def simulate_api_data(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        """
        Simulate real-time API data ingestion.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Seed for reproducibility
        
        Returns:
            DataFrame with features and target
        
        Why simulate API data?
        - Real pipelines fetch from APIs, databases, message queues
        - Shows you understand data ingestion patterns
        - Controllable for testing (no external service dependency)
        - In production: would call actual API with error handling, retries
        """
        np.random.seed(random_state)
        
        # Generate realistic features
        feature_1 = np.random.normal(loc=100, scale=15, size=n_samples)  # Distribution 1
        feature_2 = np.random.normal(loc=50, scale=10, size=n_samples)   # Distribution 2
        feature_3 = np.random.choice(['A', 'B', 'C'], size=n_samples)    # Categorical
        
        # Generate target (slight correlation with features for realism)
        target = (
            (feature_1 > 100).astype(int) + 
            (feature_2 > 50).astype(int) + 
            (feature_3 == 'A').astype(int)
        )
        target = (target >= 2).astype(int)
        
        df = pd.DataFrame({
            'feature_1': feature_1,
            'feature_2': feature_2,
            'feature_3': feature_3,
            'target': target
        })
        
        return df
    
    def get_data(
        self, 
        source: Literal['csv', 'api'] = 'csv',
        filename: Optional[str] = None,
        n_samples: int = 200
    ) -> pd.DataFrame:
        """
        Load data from specified source.
        
        Args:
            source: 'csv' or 'api'
            filename: CSV filename (required if source='csv')
            n_samples: Samples to generate (if source='api')
        
        Returns:
            DataFrame
        
        Design: a unified interface? as
        - Pipeline calls get_data() once, doesn't care about source
        - Easy to switch sources in config without changing pipeline code
        - Standard pattern in production ML (DataLoader abstraction)
        """
        if source == 'csv':
            if filename is None:
                raise ValueError("filename required for CSV source")
            return self.load_csv(filename)
        
        elif source == 'api':
            return self.simulate_api_data(n_samples=n_samples)
        
        else:
            raise ValueError(f"Unknown source: {source}")