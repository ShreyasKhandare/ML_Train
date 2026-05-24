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
from pathlib import Path

# Create necessary directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

# Import and run pipeline
from src.orchestration.workflow import ml_pipeline

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ML_Train: Production ML Pipeline Orchestration")
    print("=" * 70 + "\n")
    
    try:
        # Run Prefect flow
        result = ml_pipeline()
        
        print("\n" + "=" * 70)
        print(f"✓ Pipeline completed!")
        print(f"  Run ID: {result['run_id']}")
        print(f"  Status: {result['status']}")
        print(f"  Metrics: {result['metrics']}")
        print("=" * 70 + "\n")
        
        sys.exit(0)
    
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Pipeline failed!")
        print(f"  Error: {str(e)}")
        print("=" * 70 + "\n")
        
        sys.exit(1)