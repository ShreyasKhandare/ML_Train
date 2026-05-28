"""
ML_Train Flask API - Minimal, production-ready version
No Prefect overhead - direct ML pipeline execution
"""

import os
import json
import traceback
from flask import Flask, jsonify, request
from datetime import datetime
from pathlib import Path
import uuid

# Create necessary directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# Import ML components (no Prefect)
try:
    from src.data.loader import DataLoader
    from src.data.validator import DataValidator
    from src.preprocessing.pipeline import PreprocessingPipeline
    from src.models.trainer import ModelTrainer
    from src.models.evaluator import ModelEvaluator
    ML_AVAILABLE = True
except Exception as e:
    print(f"Warning: ML components unavailable: {e}")
    ML_AVAILABLE = False

@app.route('/', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'ML_Train Pipeline API',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/status', methods=['GET'])
def status():
    """Get service status."""
    return jsonify({
        'service': 'ML_Train Pipeline API',
        'status': 'running',
        'ml_available': ML_AVAILABLE,
        'endpoints': {
            'GET /': 'Health check',
            'GET /status': 'Service status',
            'POST /run': 'Execute ML pipeline'
        },
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/run', methods=['POST'])
def run_pipeline():
    """Execute ML pipeline directly (no Prefect overhead)."""
    if not ML_AVAILABLE:
        return jsonify({
            'status': 'error',
            'error': 'ML components not available',
            'timestamp': datetime.now().isoformat()
        }), 503

    try:
        print("\n" + "="*70)
        print("ML_Train Pipeline - Direct Execution")
        print("="*70 + "\n")

        run_id = str(uuid.uuid4())[:8]

        # Step 1: Load data
        print("📥 Loading data...")
        loader = DataLoader()
        try:
            data = loader.load_csv("data.csv")
            print(f"   ✓ Loaded from CSV: {len(data)} rows")
        except FileNotFoundError:
            print("   ℹ CSV not found, using simulated API data")
            data = loader.simulate_api_data(n_samples=200)
            print(f"   ✓ Generated API data: {len(data)} rows")

        # Step 2: Validate
        print("✔️  Validating data...")
        validator = DataValidator(data)
        validation = validator.validate()
        if validation['status'] == 'invalid':
            raise ValueError(f"Validation failed: {validation['errors']}")
        print(f"   ✓ Validation passed")

        # Step 3: Preprocess
        print("🔧 Preprocessing features...")
        X = data.drop('target', axis=1)
        y = data['target']
        pipeline = PreprocessingPipeline()
        X_processed = pipeline.fit_transform(X)
        pipeline.save("models/preprocessor.pkl")
        print(f"   ✓ Preprocessing done: {X_processed.shape}")

        # Step 4: Train
        print("🤖 Training model...")
        trainer = ModelTrainer(n_estimators=100, max_depth=5, learning_rate=0.1)
        result = trainer.train(X_processed, y, test_size=0.2, verbose=False)
        trainer.save("models/model.pkl")
        print(f"   ✓ Model trained")

        # Step 5: Evaluate
        print("📊 Evaluating model...")
        model = result['model']
        X_test = result['X_test']
        y_test = result['y_test']
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X_test, y_test)
        print(f"   ✓ Accuracy: {metrics['accuracy']:.4f}")

        # Step 6: Drift check
        print("🔍 Checking for data drift...")
        X_train = result['X_train']
        drift_result = evaluator.detect_drift(X_train, X_test)
        if drift_result['drift_detected']:
            print(f"   ⚠️  Drift detected")
        else:
            print(f"   ✓ No drift detected")

        print("\n" + "="*70)
        print(f"✅ Pipeline completed! Run: {run_id}")
        print("="*70 + "\n")

        return jsonify({
            'status': 'success',
            'run_id': run_id,
            'metrics': metrics,
            'drift_detected': drift_result['drift_detected'],
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ Pipeline failed: {str(e)}")
        print("="*70 + "\n")

        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found. Use GET /status for available endpoints'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting ML_Train API on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
