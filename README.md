# Athena

**Automated EDA, ML readiness scoring, and data diagnostics, running locally in your browser.**

[![PyPI](https://img.shields.io/pypi/v/athena-eda)](https://pypi.org/project/athena-eda/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

```bash
pip install athena-eda
athena run
```

Upload a CSV and get a diagnostic report covering data quality, leakage risks, a baseline model, and a model comparison. Then download a Python preprocessing script tailored to your dataset. No code required.

**Your data stays on your machine.** Athena runs as a local server on `127.0.0.1`, and nothing is uploaded anywhere.

---

## Quick start

1. `pip install athena-eda` (Python 3.10+)
2. `athena run`. Your browser opens at `http://127.0.0.1:8000`.
3. Upload a CSV and pick a mode:
   - **Supervised**: you have a target column to predict. Choose it, and Athena treats it as classification or regression.
   - **Unsupervised**: no target. Athena looks for cluster structure and anomalies instead.
4. Choose an [analysis profile](#analysis-profiles) and run.
5. Explore the report, then **↓ SCRIPT.PY** for a preprocessing script or export the full report as HTML.

To check for drift between two datasets, use **Compare** on the upload page and give it a train and a test CSV.

---

## What's in the report

| Tab | What you get |
|---|---|
| **Overview** | ML Readiness Score, data structure score, critical alerts, penalties and recommendations |
| **Features** | Column classification, skewness, numeric and categorical summaries, free-text columns, feature-to-target correlation |
| **Quality** | Missing values, outliers, redundant feature pairs |
| **ML Diagnostics** | Supervised: cross-validation scores, leakage risks, stability, learning curves, Model Arena. Unsupervised: clustering and anomaly detection |
| **Time Series** | Shown when a datetime column is detected: stationarity, decomposition, ACF/PACF, forecasting baselines |
| **NLP Analysis** | Shown when free-text columns are detected: vocabulary, text quality, vectorization advice |

## Features

### Core EDA
- Column classification: numeric, categorical, identifier, entity key, free text
- Missing value analysis at row and column level
- Outlier and skew analysis with log-transform recommendations
- Redundant feature pairs from the correlation matrix

### ML readiness (supervised)
- **Leakage detection**: features almost perfectly correlated with the target, categories that perfectly separate it, identifier columns, and duplicate rows
- **Baseline probe**: 3-fold cross-validated LightGBM with learning curves and over/underfitting flags. Early stopping uses a holdout from the training data, so the reported score isn't tuned on the data it's measured on.
- **Stability**: retrains on several subsamples to check whether results hold up
- **Model Arena**: Logistic Regression, Random Forest and LightGBM for classification; Ridge, Lasso, Random Forest and LightGBM for regression. All are compared on the same folds with preprocessing fitted inside each fold, then ranked by balanced accuracy or RMSE.

### Unsupervised diagnostics
- **Clustering**: silhouette scores for k = 2 to 5, plus stability across subsamples
- **Anomaly detection**: Isolation Forest with a score distribution

### Time series
- Automatic datetime and target detection
- ADF and KPSS stationarity tests with differencing advice
- Seasonal decomposition, ACF/PACF, missing-timestamp coverage
- Baselines: Naive, Moving Average, Exponential Smoothing, Auto-ARIMA, ranked by RMSE

### NLP
- Free-text column detection, vocabulary and lexical diversity
- Empty, duplicate and URL ratios
- TF-IDF vs. embeddings recommendation, per-class top words

### Drift detection
- Per-column drift scores (KS test for numeric, distribution shift for categorical)
- Categories that appear only in the test set, and changes in missing rates

### Outputs
- **Preprocessing script**, a ready-to-run Python script based on your report
- **Time-series pipeline script**, used when a time series is detected: loading, stationarity, decomposition, train/test split and baselines
- **NLP preprocessing script**, used when free-text columns are detected
- **HTML report**, the full report with charts, for sharing

### About the ML Readiness Score

The score combines data health, trainability and leakage risk into one 0–100 number. It's a **heuristic**: it adds up rule-based penalties and isn't calibrated against real model outcomes. Use it to spot problems, and read the penalties and recommendations behind it rather than treating the number as a verdict.

Results are reproducible: the same CSV gives the same report every time.

---

## Analysis profiles

Profiles change the thresholds Athena uses to flag issues. Leakage checks are the same in every profile.

| Profile | Flags missing values above | High cardinality above | Redundant if correlation above | Skew flagged above | Anomaly rate flagged above |
|---|---|---|---|---|---|
| Standard | 30% | 50 categories | 0.98 | 1.0 | 15% |
| Finance | 5% | 100 categories | 0.95 | 0.5 | 5% |
| Healthcare | 20% | 50 categories | 0.98 | 1.0 | 10% |
| NLP | 50% | 200 categories | 0.99 | 2.0 | 20% |

---

## CLI

```bash
athena run                  # start Athena and open the browser
athena run --port 9000      # use a different port
athena run --no-browser     # don't open a browser tab
athena info                 # system and package info
athena doctor               # check the installation
athena version              # installed version
athena update               # upgrade to the latest release
```

## Using Athena from AI agents (MCP)

Athena ships an [MCP](https://modelcontextprotocol.io) server so agents like Claude Code can run its checks on a CSV directly.

```bash
pip install "athena-eda[mcp]"
claude mcp add athena -- athena-mcp      # or: athena mcp
```

| Tool | Input | Returns |
|---|---|---|
| `describe_dataset` | `csv_path` | Columns, dtypes, inferred roles, missing %, row count |
| `check_leakage` | `csv_path`, `target_col` | Leakage score, risk level, suspicious features |
| `run_baselines` | `csv_path`, `target_col` | Task type, CV metric, top models, winner |
| `check_drift` | `reference_csv`, `current_csv` | Overall drift score, top drifted columns, new categories |

Every tool takes an optional `max_rows` to sample large CSVs, and returns compact JSON (or `{"error": ...}`).

To teach the agent when and in what order to use the tools, install the bundled skill:

```bash
athena install-skill      # copies SKILL.md to ~/.claude/skills/athena/
```

Then ask something like *"Is data/churn.csv ready for modelling? Compare it with data/churn_new.csv."*

## Limits

- CSV files only, up to 100 MB
- The first 200,000 rows are analyzed
- The Model Arena trains on at most 50,000 rows

---

## Development

```bash
git clone https://github.com/SaurabhLingam/Athena.git
cd Athena
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

Run the backend and the frontend dev server in two terminals:

```bash
uvicorn backend.main:app --reload --port 8000
```

```bash
cd frontend
npm install
npm run dev                    # http://localhost:3000
```

Run the tests:

```bash
python -m pytest
```

### Building a release

```bash
cd frontend && npm run build && cd ..   # writes the UI into backend/static/
python -m build                          # builds dist/athena_eda-<version>*
python -m twine upload dist/athena_eda-<version>*
```

Bump `version` in `pyproject.toml` first. The frontend must be rebuilt for UI changes to reach the package.

---

## Stack

**Frontend**: React, TypeScript, Vite, Recharts

**Backend**: FastAPI, pandas, scikit-learn, LightGBM, statsmodels, pmdarima

## License

[MIT](LICENSE)
