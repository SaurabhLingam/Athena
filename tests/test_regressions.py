import numpy as np
import pandas as pd
from typer.testing import CliRunner

from backend.cli import app
from backend.drift import DriftAnalyzer, DRIFT_THRESHOLD
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


def test_moderate_real_features_are_not_flagged_but_planted_leak_is():
    # Binary targets use single-feature AUC tiers (>= 0.90), so breast cancer's AUC ~0.97
    # "worst" features are flagged as possible/likely too; moderate ones must stay clean.
    from sklearn.datasets import load_breast_cancer
    df = load_breast_cancer(as_frame=True).frame
    df["leak"] = df["target"] + np.random.default_rng(0).normal(scale=0.05, size=len(df))
    flagged = set(_leakage(df)["suspicious_features"])
    assert "leak" in flagged
    assert not flagged & {"mean texture", "mean smoothness", "mean symmetry", "texture error"}


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


def test_date_columns_are_detected_and_left_out_of_drift():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=400, freq="D").strftime("%Y-%m-%d")
    df = pd.DataFrame({"day": dates, "code": rng.choice(["a", "b"], 400), "x": rng.normal(size=400)})
    views = EdaToolKit().build_column_views(df)
    assert views["datetime_cols"] == ["day"]
    assert "day" not in views["categorical_features"] + views["entity_keys"]
    result = DriftAnalyzer().compare(df.iloc[:200], df.iloc[200:])
    assert "day" not in {c["column"] for c in result["column_drift"]}
    assert "day" not in result["new_categories"]
    assert result["date_ranges"]["train"]["day"] == {"min": "2020-01-01", "max": "2020-07-18", "format": "%Y-%m-%d"}


def test_whole_number_floats_are_numeric_features():
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({
        "Time": np.sort(rng.integers(0, 170_000, n)).astype(float),  # seconds, mostly unique
        "Population": rng.integers(100, 5000, n).astype(float),      # counts, p5 > 100
        "target": rng.integers(0, 2, n),
    })
    views = EdaToolKit().build_column_views(df, "target")
    assert {"Time", "Population"} <= set(views["numeric_features"])


def test_arena_classifiers_handle_imbalance():
    from backend.model_arena import ModelArenaKit
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + rng.normal(scale=1.0, size=n) > 1.6).astype(int)  # ~13% positives
    df = pd.DataFrame(x, columns=["a", "b", "c"]).assign(target=y)
    views = EdaToolKit().build_column_views(df, "target")
    result = ModelArenaKit().run_arena(df, views, "target", "classification")
    scores = {r["model"]: r["score"] for r in result["ranking"]}
    # Unweighted models score ~0.61-0.63 here by leaning on the majority class
    assert all(s > 0.7 for s in scores.values()), scores


def test_drift_scores_effect_size_not_significance():
    rng = np.random.default_rng(0)
    n = 50_000
    ref = pd.DataFrame({"tiny": rng.normal(0, 1, n), "big": rng.normal(0, 1, n),
                        "cat": rng.choice(list("abc"), n)})
    cur = pd.DataFrame({"tiny": rng.normal(0.06, 1, n), "big": rng.normal(1.5, 1, n),
                        "cat": np.append(rng.choice(list("abc"), n - 5), list("vwxyz"))})
    cols = {c["column"]: c for c in DriftAnalyzer().compare(ref, cur)["column_drift"]}
    assert cols["tiny"]["ks_pvalue"] < 0.05       # significant at this sample size...
    assert cols["tiny"]["drift_score"] < DRIFT_THRESHOLD   # ...but a small shift
    assert cols["big"]["drift_score"] > DRIFT_THRESHOLD
    assert cols["cat"]["drift_score"] < DRIFT_THRESHOLD    # 5 rare new categories are not drift


def test_geographic_code_names_are_not_numeric():
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({
        "2020 Census Tract": rng.choice(np.arange(53033000100, 53033030000, 100), n).astype(float),
        "Legislative District": rng.integers(1, 50, n).astype(float),
        "zip": rng.choice([98101.0, 98052.0, 98004.0], n),
        "county_fips": rng.permutation(n) + 53000,   # near-unique -> ID
        "target": rng.normal(size=n),
    })
    views = EdaToolKit().build_column_views(df, "target")
    assert {"2020 Census Tract", "Legislative District", "zip"} <= set(views["categorical_features"])
    assert "county_fips" in views["identifiers"]
    assert not set(views["numeric_features"])


def test_single_feature_leaks_are_flagged():
    rng = np.random.default_rng(0)
    n = 600
    target = rng.choice(["low", "mid", "high", "very_high", "max", "none"], n)  # 6 classes
    code = pd.Series(target).map({"low": "L1", "mid": "M2", "high": "H3", "very_high": "V4",
                                  "max": "X5", "none": "N6"})
    # Post-outcome column: recorded after the outcome, agrees with it 99.5% of the time
    outcome_status = np.where(rng.random(n) < 0.995, target, "unknown")
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "target_code": code,
        "outcome_status": outcome_status,
        "target": target,
    })
    flagged = set(_leakage(df)["suspicious_features"])
    assert {"target_code", "outcome_status"} <= flagged
    assert "x" not in flagged


def test_integer_columns_are_numeric_unless_few_values_or_code_names():
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({
        "fnlwgt": rng.choice(rng.integers(10_000, 900_000, 700), n),  # ~50% unique, like Adult
        "capital_gain": rng.choice([0, 0, 0, 2174, 5178, 14084, 99999] + list(range(100, 160)), n),
        "hours": rng.integers(1, 99, n),
        "weekday": rng.integers(0, 7, n),               # 7 values -> categorical
        "store_code": rng.integers(100, 400, n),        # code by name
        "zipcode": rng.integers(10_000, 10_300, n),
        "target": rng.normal(size=n),
    })
    views = EdaToolKit().build_column_views(df, "target")
    assert {"fnlwgt", "capital_gain", "hours"} <= set(views["numeric_features"])
    assert "weekday" in views["categorical_features"]
    for col in ("store_code", "zipcode"):
        assert col in views["categorical_features"] + views["identifiers"]


def test_only_exact_duplicate_rows_count_as_leakage():
    rng = np.random.default_rng(0)
    # Same features, different targets: overlap, not leakage
    base = pd.DataFrame({"q1": rng.integers(0, 3, 300), "q2": rng.normal(size=300)})
    overlap = pd.concat([base.assign(target=0), base.assign(target=1)], ignore_index=True)
    report = MLdiagnosticskit()
    eda = EdaToolKit()
    out = report.run_leakage_checks(overlap, eda.feature_correlation(overlap, "target"),
                                    eda.build_column_views(overlap, "target"), "target")
    assert out["leakage_score"] == 0
    assert any("different target" in r for r in report.recommendation_buckets["Leakage Checks"])

    # Exact copies of whole records are scored
    rich = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500),
                         "target": rng.integers(0, 2, 500)})
    assert _leakage(pd.concat([rich] * 2, ignore_index=True))["leakage_score"] == 10


def test_dates_use_one_day_first_or_month_first_format_per_column():
    dayfirst = pd.Series(["03/04/2021", "25/04/2021", "01/05/2021", "12/12/2021"])
    parsed, fmt = EdaToolKit.parse_dates(dayfirst)
    assert fmt == "%d/%m/%Y"
    assert parsed.iloc[0] == pd.Timestamp("2021-04-03")    # not March 4th
    monthfirst = pd.Series(["03/04/2021", "04/25/2021", "05/01/2021"])
    parsed, fmt = EdaToolKit.parse_dates(monthfirst)
    assert fmt == "%m/%d/%Y" and parsed.iloc[0] == pd.Timestamp("2021-03-04")
    df = pd.DataFrame({"when": dayfirst.tolist() * 5, "x": range(20)})
    assert EdaToolKit().date_ranges(df, ["when"])["when"] == {
        "min": "2021-04-03", "max": "2021-12-12", "format": "%d/%m/%Y"}


def test_ambiguous_day_month_dates_report_both_ranges():
    df = pd.DataFrame({"when": ["03/04/2021", "01/02/2021", "12/11/2021"]})
    out = EdaToolKit().date_ranges(df, ["when"])["when"]
    assert out == {
        "format": "ambiguous",
        "month_first": {"min": "2021-01-02", "max": "2021-12-11"},
        "day_first": {"min": "2021-02-01", "max": "2021-11-12"},
    }
    iso = EdaToolKit().date_ranges(pd.DataFrame({"d": ["2021-03-04", "2021-01-02"]}), ["d"])["d"]
    assert iso["format"] == "%Y-%m-%d"


def test_single_feature_leaks_near_threshold_and_with_zero_blocks():
    rng = np.random.default_rng(1)
    n = 1000
    # (a) 0/1/2 column matching a 3-class target 98% of the time
    y3 = rng.integers(0, 3, n)
    agree = np.where(rng.random(n) < 0.98, y3, (y3 + 1) % 3)
    df_a = pd.DataFrame({"x": rng.normal(size=n), "agree": agree, "target": y3})
    assert "agree" in _leakage(df_a)["suspicious_features"]
    # (b) post-outcome: 0 for every negative row, 31-180 for every positive row
    y2 = rng.integers(0, 2, n)
    days = np.where(y2 == 1, rng.integers(31, 181, n), 0)
    df_b = pd.DataFrame({"x": rng.normal(size=n), "days_in_followup": days, "target": y2})
    report = _leakage(df_b)
    assert "days_in_followup" in report["suspicious_features"] and report["leakage_score"] >= 40
    assert MLdiagnosticskit._single_feature_score(df_b["days_in_followup"], df_b["target"])[1] == 1.0


def test_baselines_use_the_same_roles_as_describe_dataset():
    from backend.model_arena import ModelArenaKit
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "hours": rng.integers(1, 99, n),          # int, many values -> numeric
        "weekday": rng.integers(0, 7, n),         # int, <= 10 values -> categorical
        "store_code": rng.integers(100, 150, n),  # int code by name -> categorical
        "price": rng.normal(100, 10, n),
        "target": rng.integers(0, 2, n),
    })
    views = EdaToolKit().build_column_views(df, "target")
    _, _, num_cols, cat_cols = ModelArenaKit()._prepare(df, views, "target", "classification")
    assert set(num_cols) == {"hours", "price"} == set(views["numeric_features"])
    assert set(cat_cols) == {"weekday", "store_code"} == set(views["categorical_features"])


def test_duplicate_row_severity_scales_with_share():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000),
                         "target": rng.integers(0, 2, 1000)})

    def run(n_copies):
        df = pd.concat([base, base.iloc[:n_copies]], ignore_index=True)
        report = MLdiagnosticskit()
        eda = EdaToolKit()
        out = report.run_leakage_checks(df, eda.feature_correlation(df, "target"),
                                        eda.build_column_views(df, "target"), "target")
        return out["leakage_score"], report

    score, report = run(5)                 # 0.5%: information only
    assert score == 0 and any("Too few to matter" in r for r in report.recommendation_buckets["Leakage Checks"])
    assert run(30)[0] == 5                 # ~2.9%: low
    assert run(100)[0] == 10               # ~9.1%: medium


def _leak_probe(df, target):
    from backend.eda import EdaToolKit
    from backend.ml import MLdiagnosticskit
    eda, kit = EdaToolKit(), MLdiagnosticskit()
    views = eda.build_column_views(df, target)
    kit.run_leakage_checks(df, eda.feature_correlation(df, target), views, target)
    return " ".join(kit.penalty_buckets.get("Leakage Checks", []))


def test_binary_leak_probe_scores_with_auc():
    from backend.ml import MLdiagnosticskit
    rng = np.random.default_rng(1)
    n = 3000
    y = (rng.random(n) < 0.3).astype(int)
    strong = pd.Series(rng.normal(size=n) + 2.7 * y)   # raw AUC ~0.97
    honest = pd.Series(rng.normal(size=n) + 0.95 * y)  # raw AUC ~0.75
    metric, score = MLdiagnosticskit._single_feature_score(strong, pd.Series(y))
    assert metric == "AUC" and score >= 0.95
    assert MLdiagnosticskit._single_feature_score(honest, pd.Series(y))[1] < 0.80
    msgs = _leak_probe(pd.DataFrame({"strong": strong, "honest": honest, "y": y}), "y")
    assert "Likely leakage: strong" in msgs and "honest" not in msgs


def test_rare_positive_yes_no_leak_is_caught():
    rng = np.random.default_rng(2)
    n = 4000
    y = (rng.random(n) < 0.02).astype(int)
    flag = np.where(y == 1, "Y", "N")
    flip = rng.random(n) < 0.01
    flag = np.where(flip, np.where(flag == "Y", "N", "Y"), flag)
    msgs = _leak_probe(pd.DataFrame({"flag": flag, "x": rng.normal(size=n), "y": y}), "y")
    assert "flag" in msgs and "x " not in msgs


def test_feature_filled_only_for_one_class_is_caught():
    rng = np.random.default_rng(3)
    n = 1000
    y = (rng.random(n) < 0.3).astype(int)
    filled = np.where(y == 1, rng.normal(size=n), np.nan)
    msgs = _leak_probe(pd.DataFrame({"filled": filled, "x": rng.normal(size=n), "y": y}), "y")
    assert "Likely leakage: filled" in msgs


def test_multiclass_leak_probe_weights_rare_classes():
    from backend.ml import MLdiagnosticskit
    rng = np.random.default_rng(4)
    n = 3000
    y = rng.choice(list("abc"), n, p=[0.9, 0.08, 0.02])
    copy = np.where(rng.random(n) < 0.97, y, "a")
    metric, score = MLdiagnosticskit._single_feature_score(pd.Series(copy), pd.Series(y))
    assert metric == "balanced_accuracy" and score >= 0.95


def _labels(counts):
    return pd.Series([k for k, n in counts.items() for _ in range(n)])


def test_class_balance_levels():
    from backend.eda import class_balance
    cases = [
        ({"a": 50, "b": 50}, "balanced"),
        ({"a": 75, "b": 25}, "imbalanced"),
        ({"a": 97, "b": 3}, "severe"),
        ({"a": 25, "b": 25, "c": 25, "d": 25}, "balanced"),
        ({"a": 25, "b": 20, "c": 20, "d": 17, "e": 15, "f": 3}, "severe"),
    ]
    for counts, level in cases:
        b = class_balance(_labels(counts))
        assert b["level"] == level, (counts, b)
        smallest = b["smallest_class"]
        assert counts[smallest] == min(counts.values())
        assert b["smallest_class_pct"] == round(100 * counts[smallest] / sum(counts.values()), 2)
        assert b["balance_ratio"] == round(counts[smallest] / sum(counts.values()) * len(counts), 4)
        assert ("message" in b) == (level != "balanced")


def test_target_insights_uses_class_balance():
    df = pd.DataFrame({"x": range(200), "y": ["a"] * 194 + ["b"] * 6})
    ti = EdaToolKit().target_insights(df, "y")
    assert ti["class_balance"]["level"] == "severe" and ti["is_imbalanced"]


def test_two_digit_year_dates_are_parsed_and_flagged_when_ambiguous():
    ambiguous = pd.Series(["03-07-24", "11-05-23", "01-12-22", "12-01-21"] * 30)
    clear = pd.Series(["03-27-24", "11-05-23", "01-12-22", "12-31-21"] * 30)  # day > 12: month first
    dotted = pd.Series(["27.03.24", "05.11.23", "31.12.21"] * 40)
    df = pd.DataFrame({"a": ambiguous, "c": clear, "d": dotted, "x": range(120)})
    views = EdaToolKit().build_column_views(df)
    assert set(views["datetime_cols"]) == {"a", "c", "d"}
    ranges = EdaToolKit.date_ranges(df, ["a", "c", "d"])
    assert ranges["a"]["format"] == "ambiguous"
    assert ranges["a"]["month_first"] == {"min": "2021-12-01", "max": "2024-03-07"}
    assert ranges["c"] == {"min": "2021-12-31", "max": "2024-03-27", "format": "%m-%d-%y"}
    assert ranges["d"]["format"] == "%d.%m.%y" and ranges["d"]["max"] == "2024-03-27"
