# Athena — ML Diagnostics Platform

Upload a CSV. Get a full diagnostic report. Download a preprocessing script.

Athena is a machine learning readiness tool that analyzes tabular datasets and surfaces what matters before you start modeling — leakage risks, class imbalance, outliers, feature redundancy, NLP readiness, and more.

---

## How It Works

1. Upload a CSV — up to 200k rows, labeled or unlabeled
2. Select Supervised or Unsupervised mode. For supervised, specify the target column
3. Choose an analysis profile: Standard, Finance, Healthcare, or NLP
4. Athena runs a full diagnostic pass and returns a structured report
5. Explore results across five tabs: Overview, Features, Quality, ML Diagnostics, and NLP
6. Download a ready-to-run Python preprocessing script tailored to your dataset, or export the full report as HTML

---

## Features

- **ML Readiness Score** — composite score across data health, trainability, and leakage risk
- **Leakage Detection** — flags identifier columns and suspiciously correlated features
- **Baseline Probe** — 3-fold cross-validated LightGBM baseline with learning curves
- **Outlier and Skew Analysis** — per-feature skewness, kurtosis, log-transform recommendations
- **NLP Readiness** — detects free-text columns, vocabulary analysis, embedding recommendations
- **Drift Comparison** — upload train and test splits to get per-column distribution drift scores
- **Feature Redundancy** — correlation matrix analysis for highly redundant feature pairs
- **Analysis Profiles** — Standard, Finance, Healthcare, NLP
- **Preprocessing Script** — one-click export of a production-ready Python preprocessing script

---

## Stack

- **Frontend** — React, TypeScript, Vite, Recharts
- **Backend** — FastAPI, Python, LightGBM, scikit-learn, pandas
