"""
MCP server exposing Athena's checks as agent tools.

Install with `pip install "athena-eda[mcp]"` and run `athena-mcp` (stdio).
Every tool takes CSV paths and returns a compact, JSON-safe summary.
"""
import functools
import math
from pathlib import Path

# pandas, scikit-learn and LightGBM load inside the tools, so the server
# answers initialize/list_tools quickly; the first tool call pays the import cost.

SEED = 42
TOP_N = 5
MAX_DESCRIBE_COLS = 30
NEAR_CONSTANT_SHARE = 0.99
NEAR_PERFECT_R2 = 0.999
NEAR_PERFECT_BALANCED_ACC = 0.98


def _clean(obj):
    """Make results JSON-safe: numpy scalars -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if type(obj).__module__ == "numpy" and hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _load(csv_path: str, max_rows: int | None) -> tuple["pd.DataFrame", dict]:
    import pandas as pd
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise ValueError(f"CSV not found: {csv_path}")
    # Read NA-like tokens as text so _replace_placeholders can count them per column
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    df, placeholders = _replace_placeholders(df)
    meta = {"rows_total": len(df), "sampled": False}
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED).reset_index(drop=True)
        meta.update(sampled=True, rows_used=max_rows)
    df.attrs["placeholders"] = placeholders  # counts are over the full file
    return df, meta


PLACEHOLDER_TOKENS = {"?", "-", "--", "na", "n/a", "nan", "null", "none", "#n/a", "<na>"}
NUMERIC_SENTINELS = (-999, -9999, 999999)


def _replace_placeholders(df):
    """Turn placeholder strings and far-out numeric sentinels into NaN.
    Returns (df, {column: {placeholder: count}})."""
    import pandas as pd
    counts = {}
    for col in df.columns:
        s = df[col]
        found = {}
        if s.dtype == object:
            text = s.astype(str).str.strip()
            mask = s.notna() & (text.str.lower().isin(PLACEHOLDER_TOKENS) | (text == ""))
            if mask.any():
                for token, n in text[mask].replace("", "<blank>").value_counts().items():
                    found[str(token)] = int(n)
                s = s.mask(mask)
                # A numeric column that only looked like text because of placeholders
                converted = pd.to_numeric(s, errors="coerce")
                if converted.notna().sum() == s.notna().sum():
                    s = converted
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            is_sentinel = s.isin(NUMERIC_SENTINELS)
            others = s[~is_sentinel].dropna()
            if is_sentinel.any() and len(others):
                span = max(float(others.max() - others.min()), 1.0)
                lo, hi = float(others.min()), float(others.max())
                far = [v for v in NUMERIC_SENTINELS if (s == v).any() and (
                    v < lo - span or v > hi + span           # far outside the real range
                    or (v < 0 <= lo)                          # negative in a non-negative column
                    or (v > 0 and hi > 0 and v > 10 * hi))]   # an order of magnitude above it
                if far:
                    for v in far:
                        found[str(v)] = int((s == v).sum())
                    s = s.mask(s.isin(far))
        if found:
            df[col] = s
            counts[str(col)] = found
    return df, counts


def _require_target(df, target_col: str):
    if target_col not in df.columns:
        raise ValueError(
            f"target_col '{target_col}' not in columns: {list(df.columns)[:50]}"
        )


def _guard(fn):
    """Return {'error': ...} instead of raising so agents get a structured failure."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return _clean(fn(*args, **kwargs))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
    return wrapper


def _constant_columns(df) -> dict:
    """Columns with one value, or >= 99% of rows sharing one value (missing counts as a value)."""
    constant, near = [], []
    for col in df.columns:
        top_share = df[col].value_counts(dropna=False, normalize=True).iloc[0]
        if df[col].nunique(dropna=False) <= 1:
            constant.append(str(col))
        elif top_share >= NEAR_CONSTANT_SHARE:
            near.append(f"{col} ({top_share:.1%})")
    return {**({"constant_columns": constant} if constant else {}),
            **({"near_constant_columns": near} if near else {})}


@_guard
def describe_dataset(csv_path: str, max_rows: int | None = None) -> dict:
    """Columns, types, inferred roles, row count and missing values. Run first to pick a target column."""
    from backend.eda import EdaToolKit
    df, meta = _load(csv_path, max_rows)
    placeholders = df.attrs.get("placeholders", {})
    eda = EdaToolKit()
    basic = eda.basicdatainfo(df)
    views = eda.build_column_views(df)
    role_of = {}
    for role, short in (("identifiers", "id"), ("datetime_cols", "date"), ("numeric_features", "num"),
                        ("categorical_features", "cat"), ("entity_keys", "key"), ("free_text_cols", "text")):
        for c in views.get(role, []):
            role_of.setdefault(c, short)
    missing = df.isnull().mean()
    shown = df.columns[:MAX_DESCRIBE_COLS]
    # One compact string per column keeps wide files small once pretty-printed
    columns = [
        f"{c}|{df[c].dtype}|{role_of.get(c, 'other')}|{int(basic['unique_values'][c])}|{round(float(missing[c]) * 100, 2):g}"
        for c in shown
    ]
    return {
        **meta,
        "n_cols": basic["cols"],
        "column_format": "name|dtype|role|n_unique|missing_pct",
        "roles": "num=numeric, cat=categorical, id=identifier, key=entity key, date=date, text=free text",
        "columns": columns,
        **({"more_columns": len(df.columns) - len(shown),
            "hidden_columns": [str(c) for c in df.columns[len(shown):]]}
           if len(df.columns) > len(shown) else {}),
        "total_missing_pct": round(float(df.isnull().values.mean()) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "date_ranges": eda.date_ranges(df, views["datetime_cols"]),
        **({"placeholder_missing": placeholders} if placeholders else {}),
        **_constant_columns(df),
    }


@_guard
def check_leakage(csv_path: str, target_col: str, max_rows: int | None = None) -> dict:
    """Target-leakage check: near-perfect correlations, perfect separators, identifiers, duplicate rows."""
    from backend.eda import EdaToolKit
    from backend.ml import MLdiagnosticskit
    from backend.profiles import get_profile
    df, meta = _load(csv_path, max_rows)
    _require_target(df, target_col)
    cfg = get_profile("standard")
    eda = EdaToolKit(cfg=cfg)
    ml_kit = MLdiagnosticskit(cfg=cfg)
    views = eda.build_column_views(df, target_col)
    corr = eda.feature_correlation(df, target_col)
    report = ml_kit.run_leakage_checks(df, corr, views, target_col)
    score = report["leakage_score"]
    return {
        **meta,
        "leakage_score": score,
        "risk": "high" if score >= 40 else "medium" if score >= 10 else "low",
        "suspicious_features": sorted(report["suspicious_features"]),
        "issues": ml_kit.penalty_buckets.get("Leakage Checks", [])[:TOP_N],
        "recommendations": ml_kit.recommendation_buckets.get("Leakage Checks", [])[:TOP_N],
    }


@_guard
def run_baselines(csv_path: str, target_col: str, max_rows: int | None = None, n_folds: int = 3) -> dict:
    """Cross-validated baseline models for the target; returns the top models and the winner."""
    import pandas as pd
    from backend.eda import EdaToolKit
    from backend.model_arena import ModelArenaKit
    from backend.profiles import get_profile
    df, meta = _load(csv_path, max_rows)
    _require_target(df, target_col)
    cfg = get_profile("standard")
    eda = EdaToolKit(cfg=cfg)
    task_type = (eda.target_insights(df, target_col) or {}).get("task_type")
    if task_type not in ("classification", "regression"):
        return {**meta, "status": "skipped", "reason": f"Unsupported task type: {task_type}"}
    views = eda.build_column_views(df, target_col)
    result = ModelArenaKit(cfg=cfg).run_arena(df, views, target_col, task_type, n_folds=n_folds)
    if result.get("status") != "ok":
        return {**meta, **result}
    best = result["ranking"][0] if result["ranking"] else {}
    warning = None
    if task_type == "classification":
        if (best.get("score") or 0) > NEAR_PERFECT_BALANCED_ACC:
            warning = (f"Near-perfect balanced accuracy ({best['score']:.4f}). Almost certainly leakage: "
                       "check for features derived from the target or that combine into it.")
    elif best.get("rmse") is not None:
        y = pd.to_numeric(df[target_col], errors="coerce")
        r2 = 1 - best["rmse"] ** 2 / y.var() if y.var() > 0 else None
        if r2 is not None and r2 > NEAR_PERFECT_R2:
            warning = (f"Near-perfect fit (R^2 ~ {r2:.5f}). Almost certainly leakage: "
                       "check for features that sum or combine into the target.")
    return {
        **({"leakage_warning": warning} if warning else {}),
        **meta,
        "status": "ok",
        "task_type": task_type,
        "metric": result["metric"],
        "metric_direction": "higher_is_better" if result["metric"] == "balanced_accuracy" else "lower_is_better",
        "n_samples": result["n_samples"],
        "n_features": result["n_features"],
        "winner": result["winner"],
        "top_models": result["ranking"][:TOP_N],
    }


@_guard
def check_drift(reference_csv: str, current_csv: str, max_rows: int | None = None) -> dict:
    """Distribution drift between a reference (e.g. train) and a current (e.g. test/prod) CSV."""
    from backend.drift import DriftAnalyzer, DRIFT_THRESHOLD
    ref, ref_meta = _load(reference_csv, max_rows)
    cur, cur_meta = _load(current_csv, max_rows)
    result = DriftAnalyzer().compare(ref, cur)
    drifted = sorted(result["column_drift"], key=lambda c: c["drift_score"], reverse=True)
    return {
        "reference": ref_meta,
        "current": cur_meta,
        "overall_drift_score": result["overall_drift_score"],
        "summary": result["summary"],
        "n_columns_compared": len(result["column_drift"]),
        "n_drifted": sum(c["drift_score"] > DRIFT_THRESHOLD for c in drifted),
        "top_drifted": [c for c in drifted if c["drift_score"] > 0][:TOP_N],
        "new_categories": {k: v[:TOP_N] for k, v in list(result["new_categories"].items())[:TOP_N]},
        "missing_rate_delta": result["missing_rate_delta"],
        "date_ranges": result["date_ranges"],
    }


TOOLS = [describe_dataset, check_leakage, run_baselines, check_drift]


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as MCPServer  # mcp 1.x
        except ImportError as e:
            raise SystemExit('MCP support not installed. Run: pip install "athena-eda[mcp]"') from e
    server = MCPServer("athena")
    for tool in TOOLS:
        server.tool()(tool)
    return server


def main():
    build_server().run()


if __name__ == "__main__":
    main()
