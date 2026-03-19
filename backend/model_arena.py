import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Optional

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, r2_score

import lightgbm as lgb

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
        X, y, feature_cols = self._prepare(df, col_views, target_col, task_type)

        if X is None:
            return {
                "status": "skipped",
                "reason": "No usable features for Model Arena.",
            }

        if len(feature_cols) == 0:
            return {
                "status": "skipped",
                "reason": "No usable features for Model Arena.",
            }

        # Cap dataset size for speed — sample if needed
        MAX_ROWS = 50_000
        if len(X) > MAX_ROWS:
            idx = np.random.choice(len(X), MAX_ROWS, replace=False)
            X, y = X[idx], y[idx]

        results = self._train_all(X, y, task_type, n_folds)

        ranked = sorted(
            [r for r in results if r.get("score") is not None],
            key=lambda r: r["score"],
            reverse=(task_type == "classification"),  # higher is better for classification
        )

        # For regression sort by RMSE ascending (lower is better)
        if task_type == "regression":
            ranked = sorted(
                [r for r in results if r.get("rmse") is not None],
                key=lambda r: r["rmse"],
            )

        metric_name = "balanced_accuracy" if task_type == "classification" else "rmse"

        return {
            "status": "ok",
            "task_type": task_type,
            "metric": metric_name,
            "n_folds": n_folds,
            "n_features": len(feature_cols),
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
    #  Data preparation — same as ml.py
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
            return None, None, []

        X_raw = df[feature_cols].copy()
        y_raw = df[target_col].copy()

        # Impute numeric
        for c in X_raw.select_dtypes(include="number").columns:
            X_raw[c] = X_raw[c].fillna(X_raw[c].median())

        # Encode + impute categoricals
        for c in X_raw.select_dtypes(exclude="number").columns:
            X_raw[c] = pd.factorize(X_raw[c].fillna("NA").astype(str))[0]

        X = X_raw.values.astype(float)

        # Encode target
        if task_type == "classification":
            unique_labels = sorted(y_raw.dropna().unique())
            label_map = {v: i for i, v in enumerate(unique_labels)}
            y = y_raw.map(label_map).values
        else:
            y = y_raw.values.astype(float)

        # Drop rows where y is NaN
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]

        return X, y, feature_cols

    # ─────────────────────────────────────────────
    #  Train all models
    # ─────────────────────────────────────────────

    def _train_all(self, X, y, task_type: str, n_folds: int) -> list:
        results = []

        if task_type == "classification":
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            scoring = "balanced_accuracy"
            models = self._classification_models(y)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            scoring = "neg_root_mean_squared_error"
            models = self._regression_models()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for name, model in models:
            result = self._evaluate_model(
                name, model, X_scaled, y, cv, scoring, task_type
            )
            results.append(result)

        # LightGBM — uses raw X (handles its own scaling internally)
        lgbm_result = self._evaluate_lgbm(X, y, task_type, n_folds)
        results.append(lgbm_result)

        return results

    def _classification_models(self, y):
        n_classes = len(np.unique(y))
        models = [
            ("Logistic Regression", LogisticRegression(
                max_iter=500, random_state=42, n_jobs=1,
                solver="lbfgs", multi_class="auto"
            )),
            ("Random Forest", RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=1,
                max_depth=8, min_samples_leaf=2
            )),
        ]
        return models

    def _regression_models(self):
        return [
            ("Ridge Regression", Ridge(alpha=1.0)),
            ("Lasso Regression", Lasso(alpha=0.01, max_iter=2000)),
            ("Random Forest", RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=1,
                max_depth=8, min_samples_leaf=2
            )),
        ]

    def _evaluate_model(self, name, model, X, y, cv, scoring, task_type) -> dict:
        try:
            start = time.time()
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            elapsed = round(time.time() - start, 2)

            if task_type == "classification":
                mean_score = self.safe_float(np.mean(scores))
                std_score  = self.safe_float(np.std(scores))
                return {
                    "model":   name,
                    "score":   mean_score,
                    "std":     std_score,
                    "rmse":    None,
                    "time_s":  elapsed,
                    "error":   None,
                }
            else:
                # scoring is neg_root_mean_squared_error
                rmse = self.safe_float(-np.mean(scores))
                std  = self.safe_float(np.std(scores))
                return {
                    "model":   name,
                    "score":   self.safe_float(-np.mean(scores)),  # positive RMSE
                    "rmse":    rmse,
                    "std":     std,
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

    def _evaluate_lgbm(self, X, y, task_type: str, n_folds: int) -> dict:
        """
        Evaluate LightGBM using the same approach as MLdiagnosticskit.run_baseline_probe
        so results are directly comparable.
        """
        try:
            start = time.time()
            n_classes = len(np.unique(y))

            if task_type == "classification":
                if n_classes == 2:
                    params = {"objective": "binary", "metric": "binary_logloss"}
                else:
                    params = {"objective": "multiclass", "num_class": n_classes, "metric": "multi_logloss"}
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                metric_fn = balanced_accuracy_score
            else:
                params = {"objective": "regression", "metric": "l2"}
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                metric_fn = lambda a, p: float(np.sqrt(np.mean((a - p) ** 2)))

            params.update({
                "verbosity": -1,
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "num_threads": 1,
            })

            fold_scores = []
            for train_idx, val_idx in kf.split(X, y if task_type == "classification" else None):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                train_data = lgb.Dataset(X_train, y_train)
                val_data   = lgb.Dataset(X_val, y_val, reference=train_data)

                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=200,
                    valid_sets=[val_data],
                    valid_names=["val"],
                    callbacks=[lgb.early_stopping(15, verbose=False)],
                )

                y_pred = model.predict(X_val)
                if task_type == "classification":
                    y_pred = (y_pred > 0.5).astype(int) if y_pred.ndim == 1 else np.argmax(y_pred, axis=1)

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