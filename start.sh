#!/bin/bash
# Fallback startup script for Render deployment

# Try gunicorn first (production)
if command -v gunicorn &> /dev/null; then
    echo "Starting with gunicorn..."
    gunicorn --workers 1 --bind 0.0.0.0:${PORT:-5000} app:app
else
    echo "Gunicorn not found, starting with Flask development server..."
    python -m flask --app app run --host=0.0.0.0 --port=${PORT:-5000}
fi
