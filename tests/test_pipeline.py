"""
Unit tests for ML_Train pipeline components.

Why test?
- Catch bugs early
- Safe refactoring
- Documentation of expected behavior
- Production standard
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "data.csv"
        test_data = pd.DataFrame({
            'feature_1': [100.5, 102.3, 98.7],
            'feature_2': [50.2, 51.5, 49.8],
            'feature_3': ['A', 'B', 'A'],
            'target': [1, 1, 0]
        })
        test_data.to_csv(csv_path, index=False)
        yield tmpdir, csv_path

# ============================================================================
# DATA LOADER TESTS
# ============================================================================

def test_data_loader_api():
    """Test API data generation."""
    loader = DataLoader()
    data = loader.simulate_api_data(n_samples=50)
    
    assert len(data) == 50, "Should generate 50 samples"
    assert 'target' in data.columns, "Should have target column"
    assert data.shape[1] == 4, "Should have 4 columns"

def test_data_loader_get_data():
    """Test unified get_data interface."""
    loader = DataLoader()
    data = loader.get_data(source='api', n_samples=100)

    assert len(data) == 100
    assert data['target'].isin([0, 1]).all(), "Target should be binary"

def test_data_loader_csv(temp_csv_file):
    """Test CSV loading from temporary file."""
    tmpdir, csv_path = temp_csv_file
    loader = DataLoader(raw_data_dir=tmpdir)
    data = loader.load_csv("data.csv")

    assert len(data) == 3, "Should load 3 rows"
    assert 'target' in data.columns, "Should have target column"
    assert set(data.columns) == {'feature_1', 'feature_2', 'feature_3', 'target'}

# ============================================================================
# VALIDATOR TESTS
# ============================================================================

def test_validator_nulls():
    """Test null detection."""
    df = pd.DataFrame({
        'col1': [1, 2, None],
        'col2': ['a', 'b', 'c'],
        'target': [0, 1, 0]
    })
    
    validator = DataValidator(df)
    result = validator.validate()
    
    # Should warn about null in col1
    assert result['status'] == 'valid'  # Only warning, not error
    assert len(result['warnings']) > 0, "Should have warnings"

def test_validator_valid_data():
    """Test valid data passes."""
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
        'target': [0, 1, 0]
    })
    
    validator = DataValidator(df)
    result = validator.validate()
    
    assert result['status'] == 'valid'
    assert len(result['errors']) == 0

def test_validator_critical_nulls():
    """Test that null in target fails."""
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
        'target': [0, None, 0]
    })
    
    validator = DataValidator(df)
    result = validator.validate()
    
    # Should fail (target has nulls)
    assert result['status'] == 'invalid'
    assert len(result['errors']) > 0

def test_validator_duplicates():
    """Test duplicate row detection."""
    df = pd.DataFrame({
        'col1': [1, 1, 2],
        'col2': ['a', 'a', 'b'],
        'target': [0, 0, 1]
    })
    
    validator = DataValidator(df)
    result = validator.validate()
    
    assert 'duplicate' in str(result['warnings']).lower()

# ============================================================================
# PREPROCESSING TESTS
# ============================================================================

def test_preprocessing_fit_transform():
    """Test preprocessing fit and transform."""
    X = pd.DataFrame({
        'feature_1': [1, 2, 3, 4],
        'feature_2': [5, 6, 7, 8],
        'feature_3': ['A', 'B', 'A', 'B']
    })
    
    pipeline = PreprocessingPipeline()
    X_transformed = pipeline.fit_transform(X)
    
    assert X_transformed.shape[0] == 4, "Should preserve row count"
    assert X_transformed.shape[1] >= 3, "Should have at least 2 numeric + 1 encoded categorical"
    
    # Should be numpy array
    assert isinstance(X_transformed, np.ndarray)

def test_preprocessing_separate_fit_transform():
    """Test fit/transform separation (prevent data leakage)."""
    X_train = pd.DataFrame({
        'feature_1': [1, 2, 3],
        'feature_2': [5, 6, 7],
        'feature_3': ['A', 'B', 'A']
    })
    
    X_test = pd.DataFrame({
        'feature_1': [2, 3, 4],
        'feature_2': [6, 7, 8],
        'feature_3': ['B', 'A', 'B']
    })
    
    pipeline = PreprocessingPipeline()
    
    # Fit on train
    X_train_transformed = pipeline.fit(X_train).transform(X_train)
    
    # Transform test (with same scaling)
    X_test_transformed = pipeline.transform(X_test)
    
    assert X_train_transformed.shape[0] == 3
    assert X_test_transformed.shape[0] == 3
    
    # Both should have same number of features
    assert X_train_transformed.shape[1] == X_test_transformed.shape[1]

def test_preprocessing_save_load():
    """Test preprocessing save and load."""
    X = pd.DataFrame({
        'feature_1': [1, 2, 3],
        'feature_2': [5, 6, 7],
        'feature_3': ['A', 'B', 'A']
    })
    
    pipeline = PreprocessingPipeline()
    X_transformed = pipeline.fit_transform(X)
    
    # Save
    pipeline.save("models/test_preprocessor.pkl")
    
    # Load
    loaded_pipeline = PreprocessingPipeline.load("models/test_preprocessor.pkl")
    X_loaded = loaded_pipeline.transform(X)
    
    # Should produce same transformation
    np.testing.assert_array_almost_equal(X_transformed, X_loaded)

# ============================================================================
# TRAINER TESTS
# ============================================================================

def test_trainer_train():
    """Test model training."""
    # Create simple dataset
    df = pd.DataFrame({
        'feature_1': np.random.randn(50),
        'feature_2': np.random.randn(50),
        'feature_3': np.random.choice(['A', 'B'], 50)
    })
    y = np.random.choice([0, 1], 50)
    
    # Preprocess
    pipeline = PreprocessingPipeline()
    X_processed = pipeline.fit_transform(df)
    
    # Train
    trainer = ModelTrainer(n_estimators=10)
    result = trainer.train(X_processed, y, test_size=0.25, verbose=False)
    
    assert result['model'] is not None
    assert len(result['X_train']) > 0
    assert len(result['X_test']) > 0
    assert result['X_train'].shape[0] + result['X_test'].shape[0] == len(X_processed)

def test_trainer_save_load():
    """Test model save and load."""
    # Create simple dataset
    X = np.random.randn(50, 3)
    y = np.random.choice([0, 1], 50)
    
    # Train
    trainer = ModelTrainer(n_estimators=10)
    result = trainer.train(X, y, test_size=0.2, verbose=False)
    model = result['model']
    
    # Save
    trainer.save("models/test_model.pkl")
    
    # Load
    loaded_model = ModelTrainer.load("models/test_model.pkl")
    
    # Should make same predictions
    pred_original = model.predict(result['X_test'])
    pred_loaded = loaded_model.predict(result['X_test'])
    
    np.testing.assert_array_equal(pred_original, pred_loaded)

# ============================================================================
# EVALUATOR TESTS
# ============================================================================

def test_evaluator_metrics():
    """Test evaluation metrics computation."""
    # Create simple dataset
    df = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.choice(['A', 'B'], 100)
    })
    y = np.random.choice([0, 1], 100)
    
    # Preprocess
    pipeline = PreprocessingPipeline()
    X = pipeline.fit_transform(df)
    
    # Train
    trainer = ModelTrainer(n_estimators=10)
    result = trainer.train(X, y, test_size=0.2, verbose=False)
    model = result['model']
    X_test = result['X_test']
    y_test = result['y_test']
    
    # Evaluate
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(X_test, y_test)
    
    assert 'accuracy' in metrics
    assert 'f1' in metrics
    assert 'auc' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert 0 <= metrics['auc'] <= 1

def test_drift_detection():
    """Test drift detection."""
    X_train = np.random.normal(0, 1, (100, 3))
    X_new = np.random.normal(2, 1, (100, 3))  # Shifted mean
    
    # Create dummy model
    from sklearn.linear_model import LogisticRegression
    y = np.random.choice([0, 1], 100)
    model = LogisticRegression()
    model.fit(X_train, y)
    
    evaluator = ModelEvaluator(model)
    drift_result = evaluator.detect_drift(X_train, X_new)
    
    # Should detect drift (distributions are different)
    assert 'drift_detected' in drift_result
    assert 'action' in drift_result

def test_drift_no_drift():
    """Test that identical data does not trigger drift."""
    rng = np.random.RandomState(42)
    X_train = rng.normal(0, 1, (100, 3))
    X_new = X_train.copy()  # Same samples → KS test should not flag drift
    
    # Create dummy model
    from sklearn.linear_model import LogisticRegression
    y = np.random.choice([0, 1], 100)
    model = LogisticRegression()
    model.fit(X_train, y)
    
    evaluator = ModelEvaluator(model)
    drift_result = evaluator.detect_drift(X_train, X_new)
    
    # Should NOT detect drift (same distribution)
    assert drift_result['drift_detected'] == False
    assert drift_result['action'] == 'continue'

# ============================================================================
# INTEGRATION TEST
# ============================================================================

def test_full_pipeline():
    """Test complete pipeline flow."""
    # Load data
    loader = DataLoader()
    data = loader.simulate_api_data(n_samples=100)
    
    # Validate
    validator = DataValidator(data)
    assert validator.validate()['status'] == 'valid'
    
    # Preprocess
    X = data.drop('target', axis=1)
    y = data['target']
    
    pipeline = PreprocessingPipeline()
    X_processed = pipeline.fit_transform(X)
    
    # Train
    trainer = ModelTrainer(n_estimators=10)
    result = trainer.train(X_processed, y, test_size=0.2, verbose=False)
    
    # Evaluate
    evaluator = ModelEvaluator(result['model'])
    metrics = evaluator.evaluate(result['X_test'], result['y_test'])
    
    # Drift check
    drift = evaluator.detect_drift(result['X_train'], result['X_test'])
    
    assert metrics is not None
    assert drift is not None
    assert 'accuracy' in metrics
    assert 'drift_detected' in drift

if __name__ == "__main__":
    pytest.main([__file__, "-v"])