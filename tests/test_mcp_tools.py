import numpy as np
import pandas as pd
import pytest

from backend.mcp_server import describe_dataset, check_leakage, run_baselines, check_drift


@pytest.fixture
def labeled_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    # Rounded: Athena treats columns with >=95% unique values as ID-like.
    x1 = rng.normal(size=n).round(1)
    x2 = rng.normal(size=n).round(1)
    y = (x1 + 0.5 * x2 > 0).astype(int)
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "leak": y * 10 + rng.normal(scale=0.5, size=n).round(1),  # near-copy of target
        "color": rng.choice(["red", "blue", "green"], size=n),
        "target": y,
    })
    df.loc[:9, "x2"] = np.nan
    path = tmp_path / "labeled.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_describe_dataset(labeled_csv):
    out = describe_dataset(labeled_csv)
    assert out["rows_total"] == 300
    cols = {c["name"]: c for c in out["columns"]}
    assert set(cols) == {"x1", "x2", "leak", "color", "target"}
    assert cols["x2"]["missing_pct"] > 0


def test_describe_dataset_samples_large_csv(labeled_csv):
    out = describe_dataset(labeled_csv, max_rows=50)
    assert out["sampled"] is True and out["rows_used"] == 50


def test_check_leakage_flags_leaky_column(labeled_csv):
    out = check_leakage(labeled_csv, "target")
    assert "error" not in out
    assert "leak" in out["suspicious_features"]
    assert out["risk"] in ("medium", "high")


def test_run_baselines(labeled_csv):
    out = run_baselines(labeled_csv, "target")
    assert out["status"] == "ok"
    assert out["task_type"] == "classification"
    assert out["winner"] and len(out["top_models"]) <= 5


def test_check_drift_detects_shift(tmp_path):
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 500).round(1), "b": rng.normal(0, 1, 500).round(1)})
    cur = pd.DataFrame({"a": rng.normal(3, 1, 500).round(1), "b": rng.normal(0, 1, 500).round(1)})
    ref.to_csv(tmp_path / "ref.csv", index=False)
    cur.to_csv(tmp_path / "cur.csv", index=False)
    out = check_drift(str(tmp_path / "ref.csv"), str(tmp_path / "cur.csv"))
    assert out["top_drifted"][0]["column"] == "a"
    assert out["n_drifted"] >= 1


def test_errors_are_structured(labeled_csv):
    assert "error" in check_leakage(labeled_csv, "nope")
    assert "error" in describe_dataset("does/not/exist.csv")
