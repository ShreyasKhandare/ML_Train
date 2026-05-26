"""
Flask API wrapper for ML_Train pipeline.
Runs the ML pipeline on demand via HTTP endpoints.
"""

import os
import json
import traceback
from flask import Flask, jsonify, request
from datetime import datetime
import sys
from pathlib import Path

# Create necessary directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

from src.orchestration.workflow import ml_pipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'ML_Train Pipeline API',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/run', methods=['POST'])
def run_pipeline():
    """Run the ML pipeline and return results."""
    try:
        print("\n" + "=" * 70)
        print("ML_Train: Production ML Pipeline Orchestration")
        print("=" * 70 + "\n")

        result = ml_pipeline()

        return jsonify({
            'status': 'success',
            'run_id': result['run_id'],
            'pipeline_status': result['status'],
            'metrics': result['metrics'],
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Pipeline failed!")
        print(f"  Error: {str(e)}")
        print(traceback.format_exc())
        print("=" * 70 + "\n")

        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """Get pipeline status info."""
    return jsonify({
        'service': 'ML_Train Pipeline API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            '/': 'Health check',
            '/run': 'Execute ML pipeline (POST)',
            '/status': 'Get service status (GET)'
        },
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
