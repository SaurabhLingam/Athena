import pandas as pd
import numpy as np
from scipy import stats


class DriftAnalyzer:
    def compare(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        common_cols = [c for c in train_df.columns if c in test_df.columns]

        n_rows = len(train_df)
        common_cols = [
            c for c in common_cols
            if train_df[c].nunique() / n_rows < 0.95
        ]

        column_drift = []
        new_categories = {}
        missing_rate_delta = {}

        for col in common_cols:
            train_series = train_df[col]
            test_series = test_df[col]

            # Missing rate delta
            train_missing = float(train_series.isnull().mean())
            test_missing = float(test_series.isnull().mean())
            delta = test_missing - train_missing
            if abs(delta) > 0.01:
                missing_rate_delta[col] = round(delta, 4)

            is_numeric = pd.api.types.is_numeric_dtype(train_series)

            if is_numeric:
                drift_entry = self._numeric_drift(col, train_series, test_series)
            else:
                drift_entry, new_cats = self._categorical_drift(col, train_series, test_series)
                if new_cats:
                    new_categories[col] = new_cats

            column_drift.append(drift_entry)

        overall_drift_score = self._overall_score(column_drift)

        drifted_cols = [c for c in column_drift if c["drift_score"] > 25]
        summary = self._build_summary(drifted_cols, new_categories, missing_rate_delta)

        return {
            "overall_drift_score": overall_drift_score,
            "column_drift": column_drift,
            "new_categories": new_categories,
            "missing_rate_delta": missing_rate_delta,
            "summary": summary
        }

    def _numeric_drift(self, col: str, train: pd.Series, test: pd.Series) -> dict:
        train_clean = train.dropna()
        test_clean = test.dropna()

        if len(train_clean) < 5 or len(test_clean) < 5:
            return {"column": col, "drift_score": 0, "ks_statistic": None, "ks_pvalue": None}

        ks_stat, ks_pvalue = stats.ks_2samp(train_clean, test_clean)

        train_mean = float(train_clean.mean())
        test_mean = float(test_clean.mean())

        # Score based on KS statistic — 0 to 1, scale to 0-100
        # Also penalize if p-value is very low (statistically significant)
        significance_boost = 20 if ks_pvalue < 0.05 else 0
        drift_score = min(100, round(float(ks_stat) * 100 + significance_boost, 1))

        return {
            "column": col,
            "drift_score": drift_score,
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pvalue), 4),
            "train_mean": round(train_mean, 4),
            "test_mean": round(test_mean, 4),
        }

    def _categorical_drift(self, col: str, train: pd.Series, test: pd.Series):
        train_clean = train.dropna().astype(str)
        test_clean = test.dropna().astype(str)

        train_cats = set(train_clean.unique())
        test_cats = set(test_clean.unique())
        new_cats = list(test_cats - train_cats)

        # Distribution shift — compare value frequencies
        train_dist = train_clean.value_counts(normalize=True)
        test_dist = test_clean.value_counts(normalize=True)

        all_cats = set(train_dist.index) | set(test_dist.index)
        shift = sum(
            abs(train_dist.get(c, 0) - test_dist.get(c, 0))
            for c in all_cats
        )

        # New categories are a hard penalty
        new_cat_penalty = min(40, len(new_cats) * 10)
        drift_score = min(100, round(float(shift) * 50 + new_cat_penalty, 1))

        return {
            "column": col,
            "drift_score": drift_score,
            "distribution_shift": round(float(shift), 4),
            "new_category_count": len(new_cats),
        }, new_cats

    def _overall_score(self, column_drift: list) -> int:
        if not column_drift:
            return 0
        scores = [c["drift_score"] for c in column_drift]
        # Weight higher drift columns more
        weighted = np.average(scores, weights=[s + 1 for s in scores])
        return min(100, int(round(weighted)))

    def _build_summary(self, drifted_cols, new_categories, missing_rate_delta) -> str:
        parts = []

        if not drifted_cols and not new_categories and not missing_rate_delta:
            return "No significant drift detected between train and test datasets."

        if drifted_cols:
            parts.append(f"{len(drifted_cols)} column(s) show significant distribution drift")

        if new_categories:
            total_new = sum(len(v) for v in new_categories.values())
            parts.append(f"{total_new} unseen category value(s) detected in test across {len(new_categories)} column(s)")

        if missing_rate_delta:
            high_delta = {k: v for k, v in missing_rate_delta.items() if abs(v) > 0.1}
            if high_delta:
                parts.append(f"{len(high_delta)} column(s) have >10% change in missing rate")

        return ". ".join(parts) + "."