# ML_Train: Production ML Pipeline Orchestration

End-to-end ML training pipeline demonstrating production-grade engineering practices.

[![Tests Passing](https://github.com/ShreyasKhandare/ML_Train/actions/workflows/test.yml/badge.svg)](https://github.com/ShreyasKhandare/ML_Train/actions)


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

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip or conda

### 1. Clone & Setup

```bash
git clone https://github.com/ShreyasKhandare/ML_Train.git
cd ML_Train

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Pipeline

```bash
python main.py
```

**Expected output:**

### 3. Run Tests

```bash
pytest tests/ -v
```

**Expected:** 16 tests pass

### 4. Check Results

Pipeline stores results in SQLite database:

```bash
python << 'EOF'
import sqlite3

conn = sqlite3.connect('logs/pipeline.db')
c = conn.cursor()

c.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT 1;")
run = c.fetchone()
print(f"Latest run: {run[0]}, Status: {run[2]}, Drift: {run[3]}")

c.execute("SELECT * FROM metrics WHERE run_id = ?;", (run[0],))
metrics = c.fetchone()
print(f"Accuracy: {metrics[1]:.4f}, AUC: {metrics[5]:.4f}")

conn.close()
EOF
```

---

## 🔍 Why This Architecture?

### 1. **Modular Design**

Each component is independent:
- `loader.py` → can swap CSV for PostgreSQL
- `validator.py` → can add custom checks
- `trainer.py` → can swap XGBoost for LightGBM
- `evaluator.py` → can add new metrics

### 2. **Drift Detection (Production-Critical)**

Most teams don't monitor drift → silent model degradation.

**What we do:**
- KS test on each feature (p-value < 0.05 = drift)
- Detects distribution shifts automatically
- Logs action: `continue` or `retrain`

**Why KS test?**
- Fast: O(n) complexity
- No hyperparameters
- Statistical rigor (p-values)
- Production standard

### 3. **Data Leakage Prevention**

```python
# ❌ WRONG: Fit scaler on all data
scaler.fit(X_all)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# ✅ RIGHT: Fit on train, apply to test
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # Uses train statistics
```

We implement the ✅ approach.

### 4. **Orchestration with Prefect**

Without orchestration:
```bash
python load_data.py
python preprocess.py
python train.py
python evaluate.py
# If step 2 fails, no retry, no logging
```

With Prefect:
```python
@flow
def pipeline():
    data = load_data_task()      # Automatic retry on failure
    data = validate_task(data)    # Logged separately
    X, y = preprocess_task(data)  # Can skip/restart
    model = train_task(X, y)      # Long task gets monitored
    metrics = evaluate_task(...)  # Results persist
```

---

## 📊 Metrics & Performance

On test set (with sample synthetic data):

| Metric | Value |
|--------|-------|
| Accuracy | 0.53-0.68 |
| Precision | 0.48-0.67 |
| Recall | 0.56-0.67 |
| F1 | 0.52-0.67 |
| AUC | 0.54-0.73 |

*(Varies based on random splits; synthetic data has no signal)*

---

## 🧪 Testing

### 16 Unit Tests

```bash
pytest tests/ -v
```

**Coverage:**
- Data loader (CSV, API, unified interface)
- Data validator (nulls, duplicates, schema)
- Preprocessing (fit/transform, save/load, no leakage)
- Model trainer (training, save/load)
- Model evaluator (metrics, drift detection)
- Integration test (full pipeline)

### Integration Test

Tests complete pipeline:
1. Load → Validate → Preprocess → Train → Evaluate → Drift → Log

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -f docker/Dockerfile -t ml-train:latest .
```

### Run Container

```bash
docker run ml-train:latest
```

**Image size:** ~1.2GB (optimized with slim base)

---

## ☁️ Deploy to Render

### 1. Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/ML_Train.git
git branch -M main
git push -u origin main
```

### 2. Create Render Account

- Go to [render.com](https://render.com)
- Sign up with GitHub

### 3. Create Web Service

1. Click **New** → **Web Service**
2. Connect your GitHub repo
3. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python main.py`
   - **Instance Type:** Free (512MB RAM)
   - **Environment:** Python 3.11

4. Click **Deploy**

### 4. View Logs

Render will:
- Install dependencies
- Run `python main.py`
- Execute pipeline
- Log results to `logs/pipeline.db`

**Free tier limitations:**
- 512 MB RAM (sufficient for this pipeline)
- Spins down after 15 min inactivity
- For always-on: upgrade to Hobby ($7/month)

---

## 📈 Production Considerations

This pipeline demonstrates:

✅ **Error Handling** - Retries, fail-fast, clear errors  
✅ **Logging** - Every step logged with timestamps  
✅ **Modularity** - Components are independent  
✅ **Testability** - Unit tests + integration tests  
✅ **Reproducibility** - Same data → same results  
✅ **Monitoring** - Drift detection, metric tracking  
✅ **Audit Trail** - SQLite records all runs  
✅ **Version Control** - Clean git history  

### To Productionize Further

- [ ] Replace SQLite → PostgreSQL (concurrent writes)
- [ ] Add MLflow model registry (version control)
- [ ] Implement webhook alerts on drift (Slack, PagerDuty)
- [ ] Add RBAC and authentication (user isolation)
- [ ] Scale to Kubernetes (multi-node deployment)
- [ ] Add feature store (data lineage)
- [ ] Implement A/B testing (shadow models)
- [ ] Add SLA monitoring (uptime, latency)

---

## 🔑 Key Design Decisions

### Why XGBoost?

| Model | Pros | Cons |
|-------|------|------|
| **XGBoost** | Fast, interpretable, handles non-linearity | Single tree-based |
| Random Forest | Stable, parallel | Slower, less accurate |
| Neural Networks | Universal approximator | Overkill, hard to tune, slow |
| Logistic Regression | Fast, interpretable | Too simple for complex data |

**We chose:** XGBoost (best default for tabular data)

### Why Prefect 2.19.0?

| Tool | Pros | Cons |
|------|------|------|
| **Prefect** | Python-first, decorators, simple | Smaller community |
| Airflow | Mature, enterprise | XML/YAML, steep learning curve |
| Dagster | Type-safe, assets | Complex, overkill for small teams |
| Manual scripts | Simple | No retry, no logging, no visibility |

**We chose:** Prefect (best for small teams, fast iteration)

### Why KS Test for Drift?

| Method | Pros | Cons |
|--------|------|------|
| **KS Test** | Fast, no tuning, statistical | Univariate only |
| PSI | Good for binned data | Requires binning decisions |
| Wasserstein | Theoretically sound | Slower, overkill |
| Monitoring dashboard | Visual | Requires human attention |

**We chose:** KS Test (production standard)

---

## 📚 Learning Resources

- **Prefect:** https://docs.prefect.io
- **XGBoost:** https://xgboost.readthedocs.io
- **scikit-learn:** https://scikit-learn.org
- **ML Monitoring:** https://www.jeremyjordan.me/ml-model-monitoring/
- **Data Drift:** https://mlopslab.org/model-drift-detection/

---

## 🤝 Contributing

This is a portfolio project. For improvements:

1. Fork the repo
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

---

## 📄 License

MIT - Use however you want

---

## 👤 Author

**Shreyas Khandare**

- Email: khandareshreyas1@gmail.com
- GitHub: [ShreyasKhandare](https://github.com/ShreyasKhandare)
- LinkedIn: [shreyas-khandare](https://linkedin.com/in/shreyas-khandare)
- Portfolio: [portfolio-website-eta-tawny-30.vercel.app](https://portfolio-website-eta-tawny-30.vercel.app)

---

## 🎓 What This Project Shows

**To Hiring Managers:**

This isn't just a model training project. It demonstrates:

1. **Production ML Thinking** - Drift detection, monitoring, audit trails
2. **Software Engineering** - Modular design, testing, version control
3. **DevOps** - Docker, CI/CD, cloud deployment
4. **Communication** - Clear documentation, code comments
5. **Problem Solving** - Why each choice was made

**That's what separates junior from senior engineers.**

---

Last Updated: May 2026  
Status: Production-Ready ✅
