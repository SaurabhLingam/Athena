import numpy as np
import pandas as pd
import pytest

from backend.mcp_server import describe_dataset, check_leakage, run_baselines, check_drift


@pytest.fixture
def labeled_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.5 * x2 > 0).astype(int)
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "leak": y * 10 + rng.normal(scale=0.01, size=n),  # near-copy of target
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
    cols = {c.split("|")[0]: c.split("|") for c in out["columns"]}
    assert set(cols) == {"x1", "x2", "leak", "color", "target"}
    assert float(cols["x2"][4]) > 0


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
    ref = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})
    cur = pd.DataFrame({"a": rng.normal(3, 1, 500), "b": rng.normal(0, 1, 500)})
    ref.to_csv(tmp_path / "ref.csv", index=False)
    cur.to_csv(tmp_path / "cur.csv", index=False)
    out = check_drift(str(tmp_path / "ref.csv"), str(tmp_path / "cur.csv"))
    assert out["top_drifted"][0]["column"] == "a"
    assert out["n_drifted"] >= 1


def test_errors_are_structured(labeled_csv):
    assert "error" in check_leakage(labeled_csv, "nope")
    assert "error" in describe_dataset("does/not/exist.csv")


def test_run_baselines_warns_when_columns_sum_to_target(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    a = rng.uniform(0, 100, n).round(2)
    b = rng.uniform(0, 300, n).round(2)
    df = pd.DataFrame({"a": a, "b": b, "noise": rng.normal(size=n), "total": a + b})
    path = tmp_path / "sum.csv"
    df.to_csv(path, index=False)
    out = run_baselines(str(path), "total")
    assert out["status"] == "ok"
    assert "leakage_warning" in out


def test_run_baselines_no_warning_on_honest_data(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    df = pd.DataFrame({"x": x, "y": x + rng.normal(size=400)})
    df.to_csv(tmp_path / "honest.csv", index=False)
    out = run_baselines(str(tmp_path / "honest.csv"), "y")
    assert out["status"] == "ok" and "leakage_warning" not in out


def test_describe_dataset_is_compact_for_wide_files(tmp_path):
    import json
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, 45)), columns=[f"V{i}" for i in range(45)])
    df.to_csv(tmp_path / "wide.csv", index=False)
    out = describe_dataset(str(tmp_path / "wide.csv"))
    assert len(out["columns"]) == 30 and out["more_columns"] == 15
    assert out["hidden_columns"] == [f"V{i}" for i in range(30, 45)]
    assert len(json.dumps(out, indent=2)) < 2000


def test_top_drifted_omits_zero_scores(tmp_path):
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 500), "same": np.zeros(500)})
    cur = ref.assign(a=rng.normal(3, 1, 500))
    ref.to_csv(tmp_path / "r.csv", index=False)
    cur.to_csv(tmp_path / "c.csv", index=False)
    out = check_drift(str(tmp_path / "r.csv"), str(tmp_path / "c.csv"))
    assert all(c["drift_score"] > 0 for c in out["top_drifted"])


def test_server_module_does_not_import_heavy_libraries():
    import subprocess
    import sys
    code = ("import sys, backend.mcp_server; "
            "print(sorted(m for m in ('pandas', 'sklearn', 'lightgbm') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_describe_dataset_exposes_capped_target_in_hidden_columns(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, 31)), columns=[f"V{i}" for i in range(31)])
    df["Class"] = rng.integers(0, 2, 200)
    df.to_csv(tmp_path / "wide_target_last.csv", index=False)
    out = describe_dataset(str(tmp_path / "wide_target_last.csv"))
    assert not any(c.startswith("Class|") for c in out["columns"])
    assert out["hidden_columns"][-1] == "Class"


def test_tool_outputs_are_ascii(tmp_path):
    import json
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 100, 400).round(2)
    b = rng.uniform(0, 300, 400).round(2)
    df = pd.DataFrame({"a": a, "b": b, "total": a + b})
    df.to_csv(tmp_path / "sum.csv", index=False)
    path = str(tmp_path / "sum.csv")
    outs = [describe_dataset(path), check_leakage(path, "total"), run_baselines(path, "total"),
            check_drift(path, path)]
    assert "leakage_warning" in outs[2]
    for out in outs:
        assert json.dumps(out, ensure_ascii=False).isascii(), out


def test_placeholders_and_sentinels_count_as_missing(tmp_path):
    n = 100
    rng = np.random.default_rng(0)
    age = rng.integers(20, 80, n).astype(object)
    age[:5] = "?"
    age[5:8] = "N/A"
    income = rng.normal(50_000, 10_000, n).round()
    income[:4] = -999          # sentinel far below real values
    score = rng.integers(-1000, 1000, n)
    score[:2] = -999           # inside the real range, so not a sentinel
    city = rng.choice(["Pune", "Delhi"], n).astype(object)
    city[:3] = "  "
    city[3:5] = "none"
    df = pd.DataFrame({"age": age, "income": income, "score": score, "city": city})
    df.to_csv(tmp_path / "ph.csv", index=False)
    out = describe_dataset(str(tmp_path / "ph.csv"))
    ph = out["placeholder_missing"]
    assert ph["age"] == {"?": 5, "N/A": 3}
    assert ph["income"] == {"-999": 4}
    assert "score" not in ph
    assert ph["city"] == {"<blank>": 3, "none": 2}
    cols = {c.split("|")[0]: c.split("|") for c in out["columns"]}
    assert cols["age"][1] == "float64" and cols["age"][2] == "num"   # parsed as numeric once cleaned
    assert float(cols["age"][4]) == 8.0


def test_free_text_and_constant_columns(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    words = ["great", "product", "arrived", "late", "but", "works", "well", "would", "buy", "again"]
    review = [" ".join(rng.choice(words, 8)) + f" order {i}" for i in range(n)]
    flag = np.zeros(n, dtype=int)
    flag[:2] = 1                                   # 99.3% one value
    df = pd.DataFrame({"review": review, "country": "IN", "flag": flag,
                       "x": rng.normal(size=n), "user_key": [f"u{i:05d}" for i in range(n)]})
    df.to_csv(tmp_path / "text.csv", index=False)
    out = describe_dataset(str(tmp_path / "text.csv"))
    roles = {c.split("|")[0]: c.split("|")[2] for c in out["columns"]}
    assert roles["review"] == "text"
    assert roles["user_key"] == "id"
    assert out["constant_columns"] == ["country"]
    assert out["near_constant_columns"] == ["flag (99.3%)"]


def test_describe_dataset_target_balance_and_redundant_pairs(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    city = rng.choice(["Pune", "Delhi", "Goa"], n)
    x = rng.normal(size=n)
    df = pd.DataFrame({
        "city": city,
        "city_code": pd.Series(city).map({"Pune": "PN", "Delhi": "DL", "Goa": "GA"}),
        "temp_c": x,
        "temp_f": x * 1.8 + 32,
        "other": rng.normal(size=n),
        "label": np.where(rng.random(n) < 0.1, "yes", "no"),
    })
    df.to_csv(tmp_path / "r.csv", index=False)
    out = describe_dataset(str(tmp_path / "r.csv"), target_col="label")
    assert out["target"]["type"] == "classification" and out["target"]["n_classes"] == 2
    assert 5 < out["target"]["minority_pct"] < 15
    pairs = out["redundant_pairs"]
    assert any("city" in p and "city_code" in p and "one-to-one" in p for p in pairs)
    assert any("temp_c" in p and "temp_f" in p for p in pairs)
    assert not any("other" in p for p in pairs)
    assert "error" in describe_dataset(str(tmp_path / "r.csv"), target_col="missing")
