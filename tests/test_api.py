import numpy as np
import pandas as pd

from backend import main
from tests.conftest import signal_df, to_csv


def test_tiny_labeled_dataset_returns_report_not_500(analyze):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4], "y": [0, 1, 0]})
    r = analyze(df, target_col="y")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ml_diagnostics"]["status"] == "critical"
    assert body["model_arena"]["status"] == "skipped"


def test_missing_string_target_rows_are_dropped(analyze):
    df = signal_df()
    df["y"] = df["y"].map({0: "no", 1: "yes"}).astype(object)
    df.loc[::25, "y"] = np.nan
    r = analyze(df, target_col="y")
    assert r.status_code == 200, r.text
    ml = r.json()["ml_diagnostics"]
    assert ml["status"] == "ok", ml
    assert ml["baseline"]["mean_score"] > 0.6
    recs = " ".join(sum(ml["recommendations"].values(), []))
    assert "missing target" in recs


def test_leakage_failure_explains_why(analyze):
    rng = np.random.default_rng(1)
    n = 200
    y = rng.integers(0, 2, size=n)
    base = pd.DataFrame({
        "leak_a": (y * 10 + rng.uniform(0, 0.5, n)).round(2),
        "leak_b": (y * 5 + rng.uniform(0, 0.5, n)).round(2),
        "noise": rng.normal(size=n).round(2),
        "y": y,
    })
    df = pd.concat([base, base.iloc[:100]], ignore_index=True)  # 33% duplicate rows
    r = analyze(df, target_col="y")
    assert r.status_code == 200, r.text
    ml = r.json()["ml_diagnostics"]
    assert ml["status"] == "failed" and ml["stage"] == "leakage"
    leak_msgs = " ".join(ml["penalties"].get("Leakage Checks", []))
    assert "leak_a" in leak_msgs and "leak_b" in leak_msgs
    assert "duplicate" in leak_msgs


def test_heavy_duplicates_produce_a_message(analyze):
    base = signal_df(n=200)
    df = pd.concat([base, base.iloc[:100]], ignore_index=True)
    r = analyze(df, target_col="y")
    ml = r.json()["ml_diagnostics"]
    assert ml["status"] == "ok"
    assert any("duplicate" in m for m in ml["penalties"].get("Leakage Checks", []))


def test_report_is_deterministic(analyze):
    df = signal_df()

    def strip_timings(arena):
        for key in ("models", "ranking"):
            for m in arena.get(key, []):
                m.pop("time_s", None)
        return arena

    a, b = analyze(df, target_col="y").json(), analyze(df, target_col="y").json()
    assert a["ml_diagnostics"] == b["ml_diagnostics"]
    assert strip_timings(a["model_arena"]) == strip_timings(b["model_arena"])

    ua = analyze(df.drop(columns="y"), data_type="unlabeled").json()
    ub = analyze(df.drop(columns="y"), data_type="unlabeled").json()
    assert ua["ml_diagnostics"] == ub["ml_diagnostics"]


def test_arena_one_hot_encodes_categoricals_for_linear_models(analyze):
    # The class is decided by a category whose labels have no useful order,
    # so integer-coding it would cripple logistic regression.
    rng = np.random.default_rng(2)
    n = 400
    cats = np.array(list("ABCDEFGH"))
    positive = {"A", "D", "E", "H"}
    product = rng.choice(cats, size=n)
    df = pd.DataFrame({
        "product": product,
        "x1": rng.normal(size=n).round(2),
        "x2": rng.normal(size=n).round(2),
        "y": [int(p in positive) for p in product],
    })
    r = analyze(df, target_col="y")
    arena = r.json()["model_arena"]
    assert arena["status"] == "ok", arena
    by_name = {m["model"]: m for m in arena["models"]}
    assert all(m["error"] is None for m in arena["models"]), arena["models"]
    assert by_name["Logistic Regression"]["score"] > 0.95


def test_arena_regression_ranks_by_rmse(analyze):
    rng = np.random.default_rng(3)
    n = 300
    x1, x2, x3 = (rng.normal(size=n) for _ in range(3))
    df = pd.DataFrame({"x1": x1.round(2), "x2": x2.round(2), "x3": x3.round(2),
                       "y": (3 * x1 - 2 * x2 + rng.normal(scale=0.3, size=n)).round(3)})
    arena = analyze(df, target_col="y").json()["model_arena"]
    assert arena["status"] == "ok", arena
    rmses = [m["rmse"] for m in arena["ranking"]]
    assert rmses == sorted(rmses) and len(rmses) == 4


def test_script_returns_timeseries_pipeline(client):
    payload = {"eda": {}, "timeseries": {"detected": True, "datetime_col": "date", "target_col": "sales"}}
    r = client.post("/script", json=payload)
    assert r.status_code == 200
    assert "timeseries_pipeline.py" in r.headers["content-disposition"]


def test_cors_exposes_content_disposition(client):
    r = client.post("/script", json={"eda": {}}, headers={"Origin": "http://localhost:3000"})
    assert "content-disposition" in r.headers.get("access-control-expose-headers", "").lower()


def test_oversized_upload_rejected(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_FILE_SIZE", 10)
    r = client.post("/analyze", params={"data_type": "unlabeled"},
                    files={"file": ("data.csv", to_csv(signal_df(n=20)))})
    assert r.status_code == 413


def test_timeseries_detection_finds_dates_and_skips_text():
    from backend.timeseries import TimeSeriesKit
    n = 30
    df = pd.DataFrame({
        "iso": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
        "us": pd.date_range("2024-01-01", periods=n).strftime("%m/%d/%Y"),
        "native": pd.date_range("2024-01-01", periods=n),
        "answer": ["yes", "no"] * (n // 2),
        "sales": np.arange(n, dtype=float),
    })
    kit = TimeSeriesKit()
    assert set(kit.detect_datetime_cols(df)) == {"iso", "us", "native"}
    for col in ("iso", "us", "native"):
        assert len(kit.prepare_series(df, col, "sales")) == n
