import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prefect import flow, task
from prefect.logging import get_run_logger
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
import sqlite3
import json

# Import our modules
from data.loader import DataLoader
from data.validator import DataValidator
from preprocessing.pipeline import PreprocessingPipeline
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator

# ============================================================================
# TASKS (individual units of work)
# ============================================================================

@task(name="Load Data", retries=2)
def load_data_task():
    """
    Task 1: Load data from CSV or API.
    
    Why @task decorator?
    - Prefect tracks this as a separate unit
    - Automatic retries on failure (retries=2)
    - Logs execution time
    - Can be individually tested
    
    Why retries=2?
    - Network blip → retry automatically
    - Avoid false negatives (transient failures)
    - Production standard
    """
    logger = get_run_logger()
    loader = DataLoader()
    
    # Try CSV first (preferred), fallback to API
    try:
        data = loader.load_csv("data.csv")
        logger.info(f"✓ Loaded data from CSV: {len(data)} rows, {len(data.columns)} columns")
    except FileNotFoundError:
        logger.warning("CSV not found, using simulated API data")
        data = loader.simulate_api_data(n_samples=200)
        logger.info(f"✓ Simulated API data: {len(data)} rows")
    
    return data

@task(name="Validate Data")
def validate_task(data: pd.DataFrame):
    """
    Task 2: Validate data quality.
    
    Why separate task?
    - Can skip if in hurry (bad practice but possible)
    - Can retry independently
    - Clear responsibility: validation
    - Logging at task level
    """
    logger = get_run_logger()
    
    validator = DataValidator(data)
    result = validator.validate()
    
    if result['status'] == 'invalid':
        logger.error(f"❌ Validation failed: {result['errors']}")
        raise ValueError(f"Data validation failed: {result['errors']}")
    
    logger.info(f"✓ Data validation passed")
    if result['warnings']:
        logger.warning(f"Warnings: {result['warnings']}")
    
    return data

@task(name="Preprocess Data")
def preprocess_task(data: pd.DataFrame):
    """
    Task 3: Scale and encode features.
    
    Why separate?
    - Preprocessing is independent operation
    - Can test preprocessing without training
    - Easy to modify scaling later
    """
    logger = get_run_logger()
    
    # Split features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Create and fit preprocessing
    pipeline = PreprocessingPipeline()
    X_processed = pipeline.fit_transform(X)
    
    # Save for later use (in production)
    pipeline.save("models/preprocessor.pkl")
    logger.info(f"✓ Preprocessing complete: {X_processed.shape}")
    
    return X_processed, y

@task(name="Train Model")
def train_task(X: np.ndarray, y: np.ndarray):
    """
    Task 4: Train XGBoost model.
    
    Why separate task?
    - Longest running step (can timeout/fail)
    - Want to log training separately
    - Can restart without preprocessing again
    """
    logger = get_run_logger()
    
    trainer = ModelTrainer(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1
    )
    
    result = trainer.train(X, y, test_size=0.2, verbose=False)
    
    # Save model
    trainer.save("models/model.pkl")
    logger.info(f"✓ Model trained and saved")
    
    return result

@task(name="Evaluate Model")
def evaluate_task(training_result: dict):
    """
    Task 5: Compute metrics on test set.
    
    Why separate task?
    - Need test performance before drift detection
    - Metrics drive decision to retrain
    """
    logger = get_run_logger()
    
    model = training_result['model']
    X_test = training_result['X_test']
    y_test = training_result['y_test']
    
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(X_test, y_test)
    
    logger.info(f"✓ Model evaluation:")
    logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  - Precision: {metrics['precision']:.4f}")
    logger.info(f"  - Recall: {metrics['recall']:.4f}")
    logger.info(f"  - F1: {metrics['f1']:.4f}")
    logger.info(f"  - AUC: {metrics['auc']:.4f}")
    
    return {
        'metrics': metrics,
        'evaluator': evaluator,
        'X_test': X_test,
        'y_test': y_test,
        'X_train': training_result['X_train'],
        'y_train': training_result['y_train'],
        'model': model
    }

@task(name="Check Data Drift")
def drift_check_task(eval_result: dict):
    """
    Task 6: Detect data drift.
    
    Why separate task?
    - Decision point: should we retrain?
    - Drift detection has its own logic
    - Can be enabled/disabled independently
    """
    logger = get_run_logger()
    
    evaluator = eval_result['evaluator']
    X_train = eval_result['X_train']
    X_test = eval_result['X_test']
    
    drift_result = evaluator.detect_drift(X_train, X_test)
    
    if drift_result['drift_detected']:
        logger.warning(f"⚠️ Data drift detected!")
        logger.warning(f"Reason: {drift_result.get('reason', 'Unknown')}")
        logger.warning(f"Action: {drift_result['action']}")
    else:
        logger.info(f"✓ No significant drift detected")
    
    return {
        'drift': drift_result,
        'metrics': eval_result['metrics'],
        'model': eval_result['model']
    }

@task(name="Log Results to Database")
def log_results_task(drift_result: dict):
    """
    Task 7: Store results in database.
    
    Why persist results?
    - Track model performance over time
    - Audit trail: what changed?
    - Detect degradation patterns
    - Production requirement
    """
    logger = get_run_logger()
    
    run_id = str(uuid.uuid4())[:8]  # Short ID
    timestamp = datetime.now()
    
    # Create database if not exists
    Path("logs").mkdir(exist_ok=True)
    conn = sqlite3.connect("logs/pipeline.db")
    c = conn.cursor()
    
    # Create tables if not exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        timestamp TEXT,
        status TEXT,
        drift_detected INTEGER
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS metrics (
        run_id TEXT,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1 REAL,
        auc REAL,
        timestamp TEXT,
        FOREIGN KEY(run_id) REFERENCES runs(run_id)
    )
    ''')
    
    # Insert data
    c.execute(
        'INSERT INTO runs VALUES (?, ?, ?, ?)',
        (run_id, timestamp.isoformat(), 'success', 
         int(drift_result['drift']['drift_detected']))
    )
    
    metrics = drift_result['metrics']
    c.execute(
        '''INSERT INTO metrics 
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (run_id, metrics['accuracy'], metrics['precision'],
         metrics['recall'], metrics['f1'], metrics['auc'], timestamp.isoformat())
    )
    
    conn.commit()
    conn.close()
    
    logger.info(f"✓ Results logged to database. Run ID: {run_id}")
    
    return run_id

# ============================================================================
# FLOW (chains tasks together)
# ============================================================================

@flow(name="ML_Train_Pipeline", log_prints=True)
def ml_pipeline():
    """
    Main pipeline: orchestrates all tasks.
    
    Why @flow decorator?
    - Defines the workflow orchestration
    - Handles task dependencies automatically
    - Logs everything to Prefect
    - Provides web UI (if using Prefect Cloud)
    
    Task dependency graph:
    Load → Validate → Preprocess → Train → Evaluate → DriftCheck → Log
    (Sequential: each task waits for previous)
    """
    logger = get_run_logger()
    
    logger.info("=" * 60)
    logger.info("Starting ML_Train Pipeline")
    logger.info("=" * 60)
    
    try:
        # Execute tasks in sequence
        data = load_data_task()
        data = validate_task(data)
        X, y = preprocess_task(data)
        training_result = train_task(X, y)
        eval_result = evaluate_task(training_result)
        drift_result = drift_check_task(eval_result)
        run_id = log_results_task(drift_result)
        
        logger.info("=" * 60)
        logger.info(f"✓ Pipeline completed successfully! Run: {run_id}")
        logger.info("=" * 60)
        
        return {
            'run_id': run_id,
            'status': 'success',
            'metrics': drift_result['metrics']
        }
    
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise