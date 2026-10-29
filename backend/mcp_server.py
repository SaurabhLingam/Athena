"""
MCP server exposing Athena's checks as agent tools.

Install with `pip install "athena-eda[mcp]"` and run `athena-mcp` (stdio).
Every tool takes CSV paths and returns a compact, JSON-safe summary.
"""
import functools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backend.eda import EdaToolKit
from backend.ml import MLdiagnosticskit
from backend.model_arena import ModelArenaKit
from backend.drift import DriftAnalyzer
from backend.profiles import get_profile

SEED = 42
TOP_N = 5


def _clean(obj):
    """Make results JSON-safe: numpy scalars -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _load(csv_path: str, max_rows: int | None) -> tuple[pd.DataFrame, dict]:
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise ValueError(f"CSV not found: {csv_path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    meta = {"rows_total": len(df), "sampled": False}
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED).reset_index(drop=True)
        meta.update(sampled=True, rows_used=max_rows)
    return df, meta


def _require_target(df: pd.DataFrame, target_col: str):
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


@_guard
def describe_dataset(csv_path: str, max_rows: int | None = None) -> dict:
    """Columns, types, inferred roles, row count and missing values. Run first to pick a target column."""
    df, meta = _load(csv_path, max_rows)
    eda = EdaToolKit()
    basic = eda.basicdatainfo(df)
    views = eda.build_column_views(df)
    role_of = {}
    for role in ("identifiers", "numeric_features", "categorical_features", "entity_keys", "free_text_cols"):
        for c in views.get(role, []):
            role_of.setdefault(c, role)
    missing = df.isnull().mean()
    columns = [
        {
            "name": c,
            "dtype": str(df[c].dtype),
            "role": role_of.get(c, "other"),
            "n_unique": int(basic["unique_values"][c]),
            "missing_pct": round(float(missing[c]) * 100, 2),
        }
        for c in df.columns
    ]
    return {
        **meta,
        "n_cols": basic["cols"],
        "columns": columns,
        "total_missing_pct": round(float(df.isnull().values.mean()) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }


@_guard
def check_leakage(csv_path: str, target_col: str, max_rows: int | None = None) -> dict:
    """Target-leakage check: near-perfect correlations, perfect separators, identifiers, duplicate rows."""
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
    return {
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
        "n_drifted": sum(c["drift_score"] > 25 for c in drifted),
        "top_drifted": drifted[:TOP_N],
        "new_categories": {k: v[:TOP_N] for k, v in list(result["new_categories"].items())[:TOP_N]},
        "missing_rate_delta": result["missing_rate_delta"],
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
