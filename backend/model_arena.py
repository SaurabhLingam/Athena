import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score

try:
    from backend.ml import SEED, lgb_params, fit_lgb_early_stopped, predict_labels
except ImportError:
    from ml import SEED, lgb_params, fit_lgb_early_stopped, predict_labels

warnings.filterwarnings("ignore")


class ModelArenaKit:


    def __init__(self, cfg: dict = None):
        try:
            from backend.profiles import get_profile
        except ImportError:
            from profiles import get_profile
        self.cfg = cfg or get_profile("standard")

    @staticmethod
    def safe_float(x):
        if x is None:
            return None
        try:
            v = float(x)
            if np.isnan(v) or np.isinf(v):
                return None
            return round(v, 4)
        except Exception:
            return None

    # ─────────────────────────────────────────────
    #  Main entry point
    # ─────────────────────────────────────────────

    def run_arena(
        self,
        df: pd.DataFrame,
        col_views: Dict,
        target_col: str,
        task_type: str,
        n_folds: int = 3,
    ) -> dict:
        """
        Train all models and return ranked results.
        Called from /analyze after labeled_diagnostics completes.
        """
        X, y, num_cols, cat_cols = self._prepare(df, col_views, target_col, task_type)

        if X is None or X.shape[1] == 0:
            return {
                "status": "skipped",
                "reason": "No usable features for Model Arena.",
            }

        # Cap dataset size for speed — sample if needed
        MAX_ROWS = 50_000
        if len(X) > MAX_ROWS:
            idx = np.random.default_rng(SEED).choice(len(X), MAX_ROWS, replace=False)
            X, y = X.iloc[idx], y[idx]

        results = self._train_all(X, y, num_cols, cat_cols, task_type, n_folds)

        if task_type == "classification":
            # Higher balanced accuracy is better
            ranked = sorted(
                [r for r in results if r.get("score") is not None],
                key=lambda r: r["score"],
                reverse=True,
            )
            metric_name = "balanced_accuracy"
        else:
            # Lower RMSE is better
            ranked = sorted(
                [r for r in results if r.get("rmse") is not None],
                key=lambda r: r["rmse"],
            )
            metric_name = "rmse"

        return {
            "status": "ok",
            "task_type": task_type,
            "metric": metric_name,
            "n_folds": n_folds,
            "n_features": X.shape[1],
            "n_samples": len(X),
            "models": results,
            "ranking": [
                {
                    "rank":    i + 1,
                    "model":   r["model"],
                    "score":   r.get("score"),
                    "rmse":    r.get("rmse"),
                    "std":     r.get("std"),
                    "time_s":  r.get("time_s"),
                }
                for i, r in enumerate(ranked)
            ],
            "winner": ranked[0]["model"] if ranked else None,
        }

    # ─────────────────────────────────────────────
    #  Data preparation
    #  Only type cleanup happens here; imputation, scaling and encoding are
    #  fitted inside each CV fold by the pipelines below.
    # ─────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame, col_views: Dict, target_col: str, task_type: str):
        feature_cols = (
            col_views.get("numeric_features", []) +
            col_views.get("categorical_features", [])
        )
        exclude = (
            col_views.get("identifiers", []) +
            col_views.get("entity_keys", []) +
            [target_col]
        )
        feature_cols = [c for c in feature_cols if c not in exclude and c in df.columns]

        if not feature_cols:
            return None, None, [], []

        data = df.loc[df[target_col].notna(), feature_cols + [target_col]]
        X = data[feature_cols].copy()
        y_raw = data[target_col]

        num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in feature_cols if c not in num_cols]

        X[num_cols] = X[num_cols].astype(float)
        for c in cat_cols:
            X[c] = X[c].astype(str).where(X[c].notna(), "NA")

        if task_type == "classification":
            unique_labels = sorted(y_raw.unique())
            label_map = {v: i for i, v in enumerate(unique_labels)}
            y = y_raw.map(label_map).values.astype(int)
        else:
            y = y_raw.values.astype(float)

        return X.reset_index(drop=True), y, num_cols, cat_cols

    @staticmethod
    def _preprocessor(num_cols, cat_cols, linear: bool) -> ColumnTransformer:
        if linear:
            num_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ])
            cat_enc = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20)
        else:
            num_pipe = SimpleImputer(strategy="median")
            cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        return ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_enc, cat_cols),
        ])

    # ─────────────────────────────────────────────
    #  Train all models
    # ─────────────────────────────────────────────

    def _train_all(self, X, y, num_cols, cat_cols, task_type: str, n_folds: int) -> list:
        results = []

        if task_type == "classification":
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
            scoring = "balanced_accuracy"
            models = self._classification_models()
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
            scoring = "neg_root_mean_squared_error"
            models = self._regression_models()

        for name, model, linear in models:
            pipe = Pipeline([
                ("pre", self._preprocessor(num_cols, cat_cols, linear)),
                ("model", model),
            ])
            results.append(self._evaluate_model(name, pipe, X, y, cv, scoring, task_type))

        results.append(self._evaluate_lgbm(X, y, num_cols, cat_cols, cv, task_type))

        return results

    def _classification_models(self):
        return [
            ("Logistic Regression", LogisticRegression(
                max_iter=1000, random_state=SEED, solver="lbfgs"
            ), True),
            ("Random Forest", RandomForestClassifier(
                n_estimators=100, random_state=SEED, n_jobs=1,
                max_depth=8, min_samples_leaf=2
            ), False),
        ]

    def _regression_models(self):
        return [
            ("Ridge Regression", Ridge(alpha=1.0), True),
            ("Lasso Regression", Lasso(alpha=0.01, max_iter=2000), True),
            ("Random Forest", RandomForestRegressor(
                n_estimators=100, random_state=SEED, n_jobs=1,
                max_depth=8, min_samples_leaf=2
            ), False),
        ]

    def _evaluate_model(self, name, model, X, y, cv, scoring, task_type) -> dict:
        try:
            start = time.time()
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
            elapsed = round(time.time() - start, 2)

            if task_type == "classification":
                return {
                    "model":   name,
                    "score":   self.safe_float(np.mean(scores)),
                    "std":     self.safe_float(np.std(scores)),
                    "rmse":    None,
                    "time_s":  elapsed,
                    "error":   None,
                }
            else:
                # scoring is neg_root_mean_squared_error
                rmse = self.safe_float(-np.mean(scores))
                return {
                    "model":   name,
                    "score":   rmse,  # positive RMSE
                    "rmse":    rmse,
                    "std":     self.safe_float(np.std(scores)),
                    "time_s":  elapsed,
                    "error":   None,
                }
        except Exception as e:
            return {
                "model":   name,
                "score":   None,
                "rmse":    None,
                "std":     None,
                "time_s":  None,
                "error":   str(e),
            }

    def _evaluate_lgbm(self, X, y, num_cols, cat_cols, cv, task_type: str) -> dict:
        """
        Same folds as the other models. Early stopping uses a holdout carved
        out of each training fold (shared with MLdiagnosticskit), so the
        validation fold is only used for scoring.
        """
        try:
            start = time.time()

            if task_type == "classification":
                params = lgb_params(task_type, len(np.unique(y)))
                metric_fn = balanced_accuracy_score
            else:
                params = lgb_params(task_type)
                metric_fn = lambda a, p: float(np.sqrt(np.mean((a - p) ** 2)))

            fold_scores = []
            split_y = y if task_type == "classification" else None
            for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, split_y)):
                pre = self._preprocessor(num_cols, cat_cols, linear=False)
                X_train = pre.fit_transform(X.iloc[train_idx])
                X_val = pre.transform(X.iloc[val_idx])
                y_train, y_val = y[train_idx], y[val_idx]

                model, _ = fit_lgb_early_stopped(X_train, y_train, params, task_type, seed=SEED + fold_i)
                y_pred = predict_labels(model, X_val, task_type)

                fold_scores.append(float(metric_fn(y_val, y_pred)))

            elapsed = round(time.time() - start, 2)
            mean_s  = self.safe_float(np.mean(fold_scores))
            std_s   = self.safe_float(np.std(fold_scores))

            return {
                "model":   "LightGBM",
                "score":   mean_s,
                "rmse":    mean_s if task_type == "regression" else None,
                "std":     std_s,
                "time_s":  elapsed,
                "error":   None,
            }
        except Exception as e:
            return {
                "model":   "LightGBM",
                "score":   None,
                "rmse":    None,
                "std":     None,
                "time_s":  None,
                "error":   str(e),
            }
