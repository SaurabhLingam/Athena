# Athena

**Automated EDA, ML readiness scoring, and data diagnostics — locally in your browser.**

```bash
pip install athena-eda
athena run
```

---

## What it does

Athena analyzes tabular datasets and surfaces everything that matters before you start modeling. Upload a CSV, pick a mode, and get a structured diagnostic report in under two minutes — no code required.

It covers the full pre-modeling workflow: data quality, feature analysis, ML readiness scoring, leakage detection, time series diagnostics, and multi-model comparison. When you're done, download a ready-to-run Python preprocessing script tailored to your dataset.

---

## Features

### Core EDA
- Column classification — numeric, categorical, identifiers, free text, entity keys
- Missing value analysis with row-level and column-level breakdown
- Outlier and skew analysis — per-feature skewness, kurtosis, log-transform recommendations
- Feature redundancy detection — correlation matrix analysis for highly redundant pairs

### ML Readiness
- **ML Readiness Score** — composite score across data health, trainability, and leakage risk
- **Leakage Detection** — flags identifier columns and suspiciously correlated features
- **Baseline Probe** — 3-fold cross-validated LightGBM baseline with learning curves and overfitting detection
- **Stability Analysis** — subsample stability checks across multiple runs
- **Clustering Diagnostics** — silhouette scoring across k=2 to 5 for unlabeled datasets
- **Anomaly Detection** — Isolation Forest with score distribution visualization

### Model Arena
- Auto-trains Random Forest, LightGBM, Logistic Regression (classification) or Ridge, Lasso, Random Forest, LightGBM (regression)
- Ranks all models by balanced accuracy or RMSE with training time and variance
- Only runs on labeled datasets with a specified target column

### Time Series
- Auto-detects datetime columns and forecast targets
- Stationarity testing — ADF and KPSS with differencing recommendations
- Seasonal decomposition with auto-detected period
- ACF and PACF with confidence bands
- Missing timestamp detection and coverage analysis
- Baseline models — Naive, Moving Average, Exponential Smoothing, Auto-ARIMA ranked by RMSE

### NLP
- Free-text column detection with vocabulary and lexical diversity analysis
- Text quality metrics — empty ratio, duplicate ratio, URL ratio
- Vectorization recommendations — TF-IDF vs embeddings
- Per-class word analysis for labeled datasets

### Drift Detection
- Upload train and test splits to get per-column KS-statistic drift scores
- New category detection and missing rate delta

### Outputs
- **Preprocessing script** — one-click export of a production-ready Python script tailored to your EDA results
- **HTML report** — full diagnostic report with charts, tables, and recommendations
- **Time series script** — pipeline script with load, stationarity check, decomposition, train/test split, and all baseline models

---

## Analysis Profiles

| Profile | Use case |
|---|---|
| Standard | General purpose defaults |
| Finance | Strict leakage detection, low anomaly tolerance |
| Healthcare | High missing value tolerance, strict cardinality thresholds |
| NLP / Text | Relaxed cardinality thresholds for text-heavy datasets |

---

## Installation

```bash
pip install athena-eda
```

Requires Python 3.9+. All dependencies including `statsmodels` and `pmdarima` for time series are included in the base install.

### Running locally

```bash
athena run
```

Opens your browser at `http://127.0.0.1:8000` automatically.

### Other commands

```bash
athena info      # show system and package info
athena doctor    # check all dependencies
athena version   # show current version
athena update    # update to latest version
```

---

## Web version

A hosted version is available at [athena-alpha-hazel.vercel.app](https://athena-alpha-hazel.vercel.app) — note this runs on a server with memory limits so large datasets may fail. The local package has no such restriction.

---

## Stack

**Frontend** — React, TypeScript, Vite, Recharts

**Backend** — FastAPI, Python, LightGBM, scikit-learn, statsmodels, pmdarima, pandas

---

## License

MIT
