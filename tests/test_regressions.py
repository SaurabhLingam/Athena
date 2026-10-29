import numpy as np
import pandas as pd
from typer.testing import CliRunner

from backend.cli import app
from backend.drift import DriftAnalyzer
from backend.eda import EdaToolKit
from backend.ml import MLdiagnosticskit


def _leakage(df, target_col="target"):
    eda = EdaToolKit()
    views = eda.build_column_views(df, target_col)
    corr = eda.feature_correlation(df, target_col)
    return MLdiagnosticskit().run_leakage_checks(df, corr, views, target_col)


def test_drifted_float_column_is_detected():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"price": rng.normal(100, 10, 500)})
    cur = pd.DataFrame({"price": rng.normal(130, 10, 500)})
    result = DriftAnalyzer().compare(ref, cur)
    cols = {c["column"]: c for c in result["column_drift"]}
    assert "price" in cols
    assert cols["price"]["drift_score"] > 25


def test_float_price_column_is_not_an_identifier():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"price": rng.normal(100, 10, 500), "target": rng.integers(0, 2, 500)})
    views = EdaToolKit().build_column_views(df, "target")
    assert "price" not in views["identifiers"]
    assert "price" in views["numeric_features"]


def test_low_cardinality_near_copy_of_target_is_flagged():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    y = (x + rng.normal(size=n) > 0).astype(int)
    df = pd.DataFrame({"x": x, "near_copy": y * 10 + rng.integers(0, 2, n), "target": y})
    report = _leakage(df)
    assert "near_copy" in report["suspicious_features"]
    assert report["leakage_score"] >= 40


def test_real_id_column_is_still_skipped():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"customer_id": range(500), "code": [f"c{i}" for i in range(500)],
                        "amount": rng.normal(0, 1, 500)})
    cur = ref.assign(customer_id=range(1000, 1500), amount=rng.normal(0, 1, 500))
    views = EdaToolKit().build_column_views(ref)
    assert {"customer_id", "code"} <= set(views["identifiers"])
    drifted = {c["column"] for c in DriftAnalyzer().compare(ref, cur)["column_drift"]}
    assert "customer_id" not in drifted and "code" not in drifted


def test_install_skill_copies_file(tmp_path):
    result = CliRunner().invoke(app, ["install-skill", "--dest", str(tmp_path)])
    assert result.exit_code == 0, result.output
    installed = tmp_path / "athena" / "SKILL.md"
    assert installed.is_file()
    assert installed.read_text(encoding="utf-8").startswith("---\nname: athena")


def test_strong_real_features_are_not_flagged_but_planted_leak_is():
    from sklearn.datasets import load_breast_cancer
    df = load_breast_cancer(as_frame=True).frame
    df["leak"] = df["target"] + np.random.default_rng(0).normal(scale=0.05, size=len(df))
    flagged = set(_leakage(df)["suspicious_features"])
    assert "leak" in flagged
    assert not flagged & {"worst perimeter", "worst radius", "worst area", "worst concave points", "mean concave points"}


def test_weak_id_name_parts_need_near_uniqueness():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "education_num": rng.integers(1, 17, n),
        "order_no": rng.permutation(n) + 10_000,
        "target": rng.integers(0, 2, n),
    })
    views = EdaToolKit().build_column_views(df, "target")
    assert "education_num" not in views["identifiers"]
    assert "order_no" in views["identifiers"]
