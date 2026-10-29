---
name: athena
description: Check a tabular CSV dataset before modelling — describe its columns, detect target leakage, get cross-validated baseline scores, and measure drift between two datasets — using the Athena MCP tools. Use when the user asks whether a dataset is ready for ML, which column to predict, whether a feature leaks the target, what a realistic baseline score is, or whether train/test or train/production data have drifted.
---

# Athena dataset checks

Athena exposes four MCP tools. All take local CSV paths and return compact JSON summaries.
Every tool accepts `max_rows`: pass it (e.g. `50000`) for CSVs over ~100k rows to sample instead of loading everything. If a result contains `"error"`, fix the input (path, column name) and retry; don't guess results.

## Order

1. **`describe_dataset(csv_path)`** — always first.
   - Use `columns[].name` to get the exact target column name. If the user didn't name a target, propose one (a column whose name or role suggests a label, e.g. `target`, `label`, `churn`, `price`) and confirm before continuing.
   - Note `role: identifiers` columns (IDs) and high `missing_pct` columns. They'll come up again in later steps.
2. **`check_leakage(csv_path, target_col)`** — before any modelling.
   - `risk: high` (score ≥ 40): a feature almost certainly encodes the target. Baseline scores will be inflated and meaningless until `suspicious_features` are removed. Tell the user which columns and why (`issues`), and ask before continuing.
   - `risk: medium`: usually heavy duplicate rows or one suspicious feature. Report it and continue, but caveat the baseline numbers.
   - Identifier columns appear in `suspicious_features` with no score penalty. They're a "drop or engineer" recommendation, not leakage.
3. **`run_baselines(csv_path, target_col)`** — realistic performance floor.
   - `metric` is `balanced_accuracy` (classification, higher is better; 0.5 = chance for binary) or `rmse` (regression, lower is better, in target units).
   - Report the `winner`, its score ± `std`, and the gap to the simplest model. A small gap means the signal is mostly linear or simple.
   - If the result has `leakage_warning` (balanced accuracy > 0.98 or R² > 0.999), treat it as leakage even when `check_leakage` was clean. The leakage check scores one column at a time, so it misses features that only combine into the target (e.g. `casual + registered = cnt`). Find them, for example by asking which columns sum to the target, and tell the user before reporting any score.
   - `status: skipped` means the target isn't a usable classification or regression target.
4. **`check_drift(reference_csv, current_csv)`** — only when there's a second dataset (train vs test, last month vs this month, train vs production).
   - `overall_drift_score` is 0–100. Below 25 is negligible, 25–50 moderate, above 50 significant.
   - `top_drifted` lists per-column `drift_score`. Numeric columns include KS statistic, p-value and both means, so describe the direction of the shift.
   - `new_categories` are values seen only in the current data. A model will treat them as unseen.
   - Identifier columns (ID-like names, or near-unique integer/string columns) are skipped. If a column the user cares about is missing from the output, that's why.

## Reporting

Lead with blockers (high leakage, severe drift), then the baseline number, then minor issues. Quote column names exactly and keep it short. The tools are already summaries, so don't paste raw JSON.
