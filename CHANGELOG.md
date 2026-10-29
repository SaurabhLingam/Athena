# Changelog

## 0.3.0 — 2026-10-29

### Added
- MCP server for AI agents (`pip install "athena-eda[mcp]"`, run with `athena-mcp` or `athena mcp`). Tools: `describe_dataset`, `check_leakage`, `run_baselines`, `check_drift`. Each takes CSV paths, accepts an optional `max_rows` to sample large files, and returns a compact JSON summary.
- Bundled Claude Code skill and `athena install-skill` command, which copies it to `~/.claude/skills/athena/`.

### Fixed
- Near-unique float columns (prices, measurements) were treated as identifiers and silently skipped by drift detection and leakage checks. Floats are now treated as IDs only when their name looks like one (`id`, `*_id`, `uuid`, `key`, …).
- The leakage check missed near-copies of the target in numeric columns with few distinct values, because those columns were routed to the categorical path. These columns are now also scored by |correlation| (regression, cut-off 0.95) or single-feature AUC (classification, cut-off 0.995, so strong real predictors aren't flagged).
- `run_baselines` returns a `leakage_warning` when the best score is near-perfect (balanced accuracy > 0.98 or R² > 0.999). This catches columns that only combine into the target, which the per-column leakage check can't see.
- Column names containing `num`, `no` or `code` (e.g. `education_num`) are treated as identifiers only when the column is also nearly unique.
- Date columns are detected by dtype or by parsing a sample of values. They're no longer treated as categories, and drift reports each side's date range instead of a 100% drift score.
- Whether a column holds whole numbers is now decided by dtype, so whole-number float columns (e.g. seconds, population counts) are numeric features instead of entity keys or categories.
- Model Arena classifiers use balanced class weights, so baselines on imbalanced data are no longer dominated by the majority class (BRFSS diabetes: balanced accuracy 0.575 → 0.754).
- Drift scored statistical significance instead of shift size: any column with p < 0.05 got +20 points, so on large files tiny shifts counted as drift (creditcard `Amount`, KS 0.024, scored 22.4). Numeric drift is now the KS statistic and categorical drift the total variation distance (both ×100). New categories count through their share of rows instead of +10 each. A column drifts above 10 (was 25 including the bonus). `ks_pvalue` is still reported.
- Tool messages are ASCII-only (e.g. `R^2 ~ 1.00000`), so consoles with a non-UTF-8 code page can print them.

### Changed
- `describe_dataset` lists the first 30 columns as compact `name|dtype|role|n_unique|missing_pct` strings, plus a count and the names (`hidden_columns`) of the rest. `check_drift`'s `top_drifted` omits columns with a drift score of 0.
- The MCP server loads pandas, scikit-learn and LightGBM on first use, cutting startup from about 5.8 s to 2.6 s.

### Known issues
- Column roles depend on the dtype pandas infers. A float column whose values are all whole numbers is read as integers when the file stores them without decimals, and is then classed as an entity key or category instead of a numeric feature. Example: creditcard `Time` read from only its first 50k rows becomes integer-typed. `max_rows` isn't affected, because it reads the whole file before sampling.
