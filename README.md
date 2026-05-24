# ML_Train: Production ML Pipeline Orchestration

End-to-end ML training pipeline demonstrating production-grade engineering practices.

## Overview

Complete ML pipeline with:
- Data ingestion (CSV + API simulation)
- Data validation and quality checks
- Feature preprocessing and scaling
- XGBoost model training
- Multi-metric evaluation
- Drift detection (production monitoring)
- Prefect workflow orchestration
- SQLite logging and audit trails

## Tech Stack

- **Orchestration:** Prefect 3.0
- **ML Model:** XGBoost
- **Data Processing:** pandas, scikit-learn, numpy
- **Testing:** pytest
- **Deployment:** Docker → Render

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run Pipeline

```bash
python main.py
```

## Project Structure