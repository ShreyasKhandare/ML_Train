"""
ML_Train: Production ML Pipeline Orchestration

This script runs the complete ML training pipeline:
1. Load data (CSV or API)
2. Validate data quality
3. Preprocess features
4. Train XGBoost model
5. Evaluate on test set
6. Detect data drift
7. Log results to database

Run: python main.py
"""

import sys
import os
from pathlib import Path

# Create necessary directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

# Prefect configuration - use LOCAL server (no cloud connection)
os.environ['PREFECT_API_URL'] = 'http://localhost:4200/api'
os.environ['PREFECT_TELEMETRY_ENABLED'] = 'false'
os.environ['PREFECT_LOGGING_LEVEL'] = 'INFO'

# Import pipeline
from src.orchestration.workflow import ml_pipeline

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ML_Train: Production ML Pipeline Orchestration")
    print("=" * 70 + "\n")
    
    try:
        # Run Prefect flow (locally)
        result = ml_pipeline()
        
        print("\n" + "=" * 70)
        print(f"✓ Pipeline completed!")
        print(f"  Run ID: {result['run_id']}")
        print(f"  Status: {result['status']}")
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print("=" * 70 + "\n")
        
        sys.exit(0)
    
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Pipeline failed!")
        print(f"  Error: {str(e)}")
        print("=" * 70 + "\n")
        
        sys.exit(1)
