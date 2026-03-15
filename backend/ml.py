import os
import platform



import pandas as pd
import numpy as np
from typing import Dict, Optional

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score

import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class MLdiagnosticskit:
    def __init__(self, cfg: dict = None):
        from profiles import get_profile
        self.cfg = cfg or get_profile("standard")
        self.penalties = []
        self.recommendations = []
        self.component_scores = {
            "data_health": 100,
            "trainability": 100,
            "leakage": 100
        }
        self.penalty_buckets = {}
        self.recommendation_buckets = {}

    def add_penalty(self, source: str, text: str):
        if source not in self.penalty_buckets:
            self.penalty_buckets[source] = []
        self.penalty_buckets[source].append(text)


    def add_recommendation(self, source: str, text: str):
        if source not in self.recommendation_buckets:
            self.recommendation_buckets[source] = []
        self.recommendation_buckets[source].append(text)
        
    def run_health_checks(self, df: pd.DataFrame, col_views: Dict):
        n_rows, n_cols = df.shape

        num_cols = col_views.get("numeric_features", [])
        cat_cols = col_views.get("categorical_features", [])

        usable_features = len(num_cols) + len(cat_cols)

        health_score = 100

        if usable_features == 0:
            health_score = 0
            self.add_penalty("Health Checks", "No usable features detected")

        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        high_missing = [col for col in df.columns if df[col].isnull().mean() > 0.5]

        if n_rows < (5 * usable_features):
            health_score -= 40
            self.add_penalty("Health Checks","Dataset too small, model will likely overfit")

        if usable_features < 3:
            health_score -= 30
            self.add_penalty("Health Checks","Too few usable features, insufficient signal for learning")

        if usable_features > n_rows:
            health_score -= 30
            self.add_penalty("Health Checks","More features than samples, extreme overfitting risk")

        if len(constant_cols) > 0:
            health_score -= 10
            self.add_recommendation(
                "Health Checks",
                f"Consider Dropping constant columns: {constant_cols}"
            )

        if len(high_missing) > 0:
            health_score -= 20
            self.add_recommendation("Health Checks", f"High missing rate columns (>50%): {high_missing}")

        self.component_scores["data_health"] = max(0, health_score)

        return {
            "health_score": health_score,
            "usable_features": usable_features,
            "constant_cols": constant_cols,
            "high_missing_columns": high_missing
        }


    def run_diagnostics(self, df: pd.DataFrame, col_views: Dict, mode: str, target_col: Optional[str] = None):

        self.penalty_buckets = {}
        self.recommendation_buckets = {}

        self.penalties = []
        self.recommendations = []
        self.component_scores = {
            "data_health": 100,
            "trainability": 100,
            "leakage": 100
        }

        health_report = self.run_health_checks(df, col_views)

        if health_report["health_score"] < 30:
            return {
                "status": "critical",
                "stage": "data_health",
                "report": health_report,
                "penalties": self.penalty_buckets,
                "recommendations": self.recommendation_buckets,
                "component_scores": self.component_scores,
                "ML_readiness_score": 0
            }

        if mode == "labeled":
            if target_col is None or target_col not in df.columns:
                return {
                    "status": "failed",
                    "stage": "input",
                    "issue": "Target column missing for labeled diagnostics"
                }

            return {
                "status": "ready_for_labeled_pipeline",
                "health": health_report,
                "component_scores": self.component_scores
            }

        else:
            return {
                "status": "ready_for_unlabeled_pipeline",
                "health": health_report,
                "component_scores": self.component_scores
            }


    def labeled_diagnostics(
        self,
        df: pd.DataFrame,
        col_views: Dict,
        target_insights: Dict,
        feature_correlation: Dict,
        target_col: str
    ):


        self.component_scores["trainability"] = 100

        if target_col is None or target_col not in df.columns:

            return {
                "status": "failed",
                "stage": "input",
                "issue": "Target column missing or invalid"
            }

        task_type = target_insights.get("task_type")

        if task_type == "regression":
            stats = target_insights.get("target_stats", {})
            std = stats.get("std", 0)

            if std < 1e-6:
                self.component_scores["trainability"] = 0
                self.add_penalty("Labeled Diagnostics","Target has near-zero variance, regression impossible")

                return {
                    "status": "failed",
                    "stage": "trainability",
                    "issue": "Target variance too small"
                }

            if target_insights.get("likely_transform_needed", False):
                self.add_recommendation("Labeled_diagnostics", "Target is skewed, consider log transformation before training")

            metric_name = "r2"

        elif task_type == "classification":
            class_dist = target_insights.get("class_distribution", {})
            n_classes = len(class_dist)
            imbalance_ratio = target_insights.get("imbalance_ratio", 1)
            is_imbalance = target_insights.get("is_imbalanced", False)

            if n_classes < 2:
                self.component_scores["trainability"] = 0
                self.add_penalty("Labeled Diagnostics", "Target has only one class, classification is not possible")


                return {
                    "status": "failed",
                    "stage": "trainability",
                    "issue": "Only one target class present"
                }

            if is_imbalance:
                if imbalance_ratio > 5:
                    self.component_scores["trainability"] -= 30
                    self.add_penalty("Labeled Diagnostics","Severe class imbalance detected")
                elif imbalance_ratio > 2:
                    self.component_scores["trainability"] -= 15
                    self.add_recommendation("Labeled Diagnostics", "Class imbalance detected, consider sampling or class weights")

            metric_name = "balanced_accuracy"

        else:

            return {
                "status": "failed",
                "stage": "input",
                "issue": "Unknown or unsupported task type"
            }

        leakage_report = self.run_leakage_checks(df, feature_correlation, col_views, target_col)

        if leakage_report["leakage_score"] > 90:


            return {
                "status": "failed",
                "stage": "leakage",
                "leakage": leakage_report,
                "penalties": self.penalties,
                "recommendations": self.recommendations,
                "component_scores": self.component_scores
            }

        baseline_report = self.run_baseline_probe(
            df=df,
            col_views=col_views,
            target_col=target_col,
            task_type=task_type,
            n_folds=3
        )

        stability_report = self.run_stability_checks(
            df=df,
            col_views= col_views,
            target_col=target_col,
            task_type=task_type
        )

        ml_readiness_score = int(
            np.mean([
                self.component_scores["data_health"],
                self.component_scores["trainability"],
                100 - self.component_scores["leakage"]
            ])
        )

        return {
            "status": "ok",
            "task_type": task_type,
            "metric": metric_name,
            "baseline": baseline_report,
            "leakage": leakage_report,
            "stability": stability_report,
            "penalties": self.penalty_buckets,
            "recommendations": self.recommendation_buckets,
            "component_scores": self.component_scores,
            "ML_readiness_score": ml_readiness_score
        }


    def run_leakage_checks(self, df: pd.DataFrame, feature_correlation: Dict, col_views: Dict, target_col: str):
        leakage_score = 0
        suspicious = []

        num_corr = feature_correlation.get("numeric_target_correlation", {})
        for col, vals in num_corr.items():
            if abs(vals.get("pearson", 0)) > 0.95:
                leakage_score += 40
                suspicious.append(col)
                self.add_penalty("Leakage Checks", f"Severe leakage risk: {col} is highly correlated with target")

        cat_impact = feature_correlation.get("categorical_target_impact", [])
        for item in cat_impact:
            shift = item.get("distribution_shift", 0)
            col = item.get("column")
            if shift > 0.99:
                
                leakage_score += 25
                if col:
                    suspicious.append(col)
                    self.add_penalty("Leakage Checks", f"Potential leakage: {col} perfectly separates target (shift={shift:.3f})")
            elif shift > 0.95:
                
                if col:
                    self.add_recommendation("Leakage Checks", f"Review {col} — very high target separation (shift={shift:.3f}), verify it's not derived from target")

        id_cols = col_views.get("identifiers", [])
        if id_cols:
            leakage_score += 0
            suspicious.extend(id_cols)
            self.add_recommendation(
                "Leakage Checks",
                f"Identifier-like columns detected: {id_cols}. These were excluded from baseline training, but should be dropped or feature-engineered."
            )

        try:
            dup_frac = df.drop(columns=[target_col]).duplicated().mean()
            if dup_frac > 0.20:
                leakage_score += 25
            elif dup_frac > 0.10:
                leakage_score += 10
                self.add_penalty(
                    "Leakage Checks",
                    f"{dup_frac:.1%} duplicate feature rows detected, possible train/test leakage"
                )
        except:
            pass
        

        final_score = min(100, leakage_score)
        self.component_scores["leakage"] = final_score

        suspicious = list(set(suspicious))

        return {    
            "leakage_score": final_score,
            "suspicious_features": suspicious
        }

    def run_baseline_probe(
        self,
        df: pd.DataFrame,
        col_views: Dict,
        target_col: str,
        task_type: str,
        n_folds: int = 3
    ):

        feature_cols = col_views.get("numeric_features", []) + col_views.get("categorical_features", [])
        feature_cols = [c for c in feature_cols if c != target_col]

        identifier_cols = col_views.get("identifiers", []) + col_views.get("entity_keys", [])
        feature_cols = [c for c in feature_cols if c not in identifier_cols]

        if len(feature_cols) == 0:
            self.component_scores["trainability"] = 0
            self.add_penalty("Baseline Probe", "No usable feature columns for baseline training")
            return {"mean_score": 0, "std_score": 0, "fold_scores": [], "metric": None, "n_folds": n_folds}

        X = df[feature_cols].copy()
        y_raw = df[target_col].copy()

        if task_type == "classification":
            unique_labels = sorted(y_raw.unique())
            label_map = {v: i for i, v in enumerate(unique_labels)}
            y = y_raw.map(label_map).values

            num_classes = len(unique_labels)
            metric_fn = balanced_accuracy_score
            metric_name = "balanced_accuracy"

            if num_classes == 2:
                params = {"objective": "binary", "metric": "binary_logloss"}
            else:
                params = {"objective": "multiclass", "num_class": num_classes, "metric": "multi_logloss"}

            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        else:
            y = y_raw.values
            metric_fn = r2_score
            metric_name = "r2"

            params = {"objective": "regression", "metric": "l2"}
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        params.update({
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "num_threads": 1
        })

        fold_scores = []
        fold_curves = []
        best_iters = []

        final_val_losses = []
        train_val_gaps = []
        improvements = []

        for train_idx, val_idx in kf.split(X, y if task_type == "classification" else None):

            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y[train_idx], y[val_idx]

            for c in X_train.select_dtypes(include="number").columns:
                med = X_train[c].median()
                X_train[c] = X_train[c].fillna(med)
                X_val[c] = X_val[c].fillna(med)

            for c in X_train.select_dtypes(exclude="number").columns:
                X_train[c] = X_train[c].fillna("NA").astype(str)
                X_val[c] = X_val[c].fillna("NA").astype(str)

                train_codes, uniques = pd.factorize(X_train[c])
                X_train[c] = train_codes

                mapping = {u: i for i, u in enumerate(uniques)}
                X_val[c] = X_val[c].map(mapping).fillna(-1)

            evals_result = {}

            train_data = lgb.Dataset(X_train, y_train)
            val_data = lgb.Dataset(X_val, y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data, val_data],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.record_evaluation(evals_result),
                    lgb.early_stopping(15)
                ]
            )

            best_iters.append(model.best_iteration)

            train_curve = evals_result["train"][params["metric"]]
            val_curve = evals_result["val"][params["metric"]]

            best_i = model.best_iteration - 1

            train_best = train_curve[best_i]
            val_best = val_curve[best_i]

            final_val_losses.append(float(val_best))
            train_val_gaps.append(float(val_best - train_best))

            initial_val = val_curve[0]
            improvement = float(initial_val - min(val_curve))
            improvements.append(improvement)

            y_pred = model.predict(X_val)

            if task_type == "classification":
                if y_pred.ndim == 1:
                    y_pred = (y_pred > 0.5).astype(int)
                else:
                    y_pred = np.argmax(y_pred, axis=1)

            score = float(metric_fn(y_val, y_pred))
            fold_curves.append({
                "fold": len(fold_scores) + 1,
                "train": [round(v, 6) for v in train_curve],
                "val": [round(v, 6) for v in val_curve],
            })
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        mean_best_iter = float(np.mean(best_iters))
        std_best_iter = float(np.std(best_iters))

        mean_gap = float(np.mean(train_val_gaps))
        std_gap = float(np.std(train_val_gaps))

        mean_improvement = float(np.mean(improvements))
        val_loss_floor_gap = float(max(final_val_losses) - min(final_val_losses))

        overfitting_detected = mean_gap > 0.05
        underfitting_detected = mean_improvement < 0.01

        if underfitting_detected:
            self.add_recommendation("Baseline probe", "Model shows weak learning improvement dataset may lack signal")

        if overfitting_detected:
            self.add_recommendation("Baseline probe","Train–validation gap is moderately high, mild overfitting is possible")

        if val_loss_floor_gap > 0.3 * np.mean(final_val_losses):
            self.add_recommendation("Baseline probe", "Different folds converge to very different loss floors, unstable dataset")

        if std_score > 0.1:
            self.add_recommendation("Baseline probe", "High variance across folds, model performance is unstable")

        return {
            "mean_score": mean_score,
            "std_score": std_score,
            "fold_scores": fold_scores,
            "fold_curves": fold_curves,
            "metric": metric_name,
            "n_folds": n_folds,
            "learning_dynamics": {
                "mean_best_iteration": mean_best_iter,
                "std_best_iteration": std_best_iter,
                "mean_train_val_gap": mean_gap,
                "std_train_val_gap": std_gap,
                "mean_improvement": mean_improvement,
                "val_loss_floor_gap": val_loss_floor_gap,
                "overfitting_detected": overfitting_detected,
                "underfitting_detected": underfitting_detected
            }
        }


    def run_stability_checks(
        self,
        df: pd.DataFrame,
        col_views: Dict,
        target_col: str,
        task_type: str,
        n_runs: int = 5,
        sample_frac: float = 0.8
    ):
        from sklearn.model_selection import train_test_split
        feature_cols = col_views.get("numeric_features", []) + col_views.get("categorical_features", [])
        feature_cols = [c for c in feature_cols if c != target_col]

        identifier_cols = col_views.get("identifiers", []) + col_views.get("entity_keys", [])
        feature_cols = [c for c in feature_cols if c not in identifier_cols]

        if len(feature_cols) == 0:
            return {"status": "failed", "issue": "No usable feature columns for stability analysis"}

        X_full = df[feature_cols].copy()
        y_full = df[target_col].copy()

        scores = []
        best_iters = []
        final_losses = []

        for _ in range(n_runs):

            sample_idx = np.random.choice(len(df), int(len(df) * sample_frac), replace=False)
            X = X_full.iloc[sample_idx].copy()
            y_raw = y_full.iloc[sample_idx].copy()

            if task_type == "classification":
                unique_labels = sorted(y_raw.unique())
                label_map = {v: i for i, v in enumerate(unique_labels)}
                y = y_raw.map(label_map).values

                num_classes = len(unique_labels)

                if num_classes == 2:
                    params = {"objective": "binary", "metric": "binary_logloss"}
                else:
                    params = {"objective": "multiclass", "num_class": num_classes, "metric": "multi_logloss"}

                metric_fn = balanced_accuracy_score

            else:
                y = y_raw.values
                params = {"objective": "regression", "metric": "l2"}
                metric_fn = r2_score

            params.update({
                "verbosity": -1,
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "num_threads": 1
            })

            for c in X.select_dtypes(include="number").columns:
                X[c] = X[c].fillna(X[c].median())

            for c in X.select_dtypes(exclude="number").columns:
                X[c] = pd.factorize(X[c].fillna("NA").astype(str))[0]

            if task_type == "classification":
                try:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=None, stratify=y
                    )
                except ValueError:
                    # Fallback if any class has only 1 sample
                    split = int(0.8 * len(X))
                    X_train, X_val = X.iloc[:split], X.iloc[split:]
                    y_train, y_val = y[:split], y[split:]
            else:
                split = int(0.8 * len(X))
                X_train, X_val = X.iloc[:split], X.iloc[split:]
                y_train, y_val = y[:split], y[split:]
            evals_result = {}

            train_data = lgb.Dataset(X_train, y_train)
            val_data = lgb.Dataset(X_val, y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                valid_names=["val"],
                callbacks=[
                    lgb.record_evaluation(evals_result),
                    lgb.early_stopping(15)
                ]
            )

            best_iters.append(model.best_iteration)

            val_curve = evals_result["val"][params["metric"]]
            final_losses.append(float(min(val_curve)))

            y_pred = model.predict(X_val)

            if task_type == "classification":
                if y_pred.ndim == 1:
                    y_pred = (y_pred > 0.5).astype(int)
                else:
                    y_pred = np.argmax(y_pred, axis=1)

            score = float(metric_fn(y_val, y_pred))
            scores.append(score)

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        mean_best_iter = float(np.mean(best_iters))
        std_best_iter = float(np.std(best_iters))

        mean_loss = float(np.mean(final_losses))
        std_loss = float(np.std(final_losses))

        loss_floor_gap = float(max(final_losses) - min(final_losses))
        loss_gap_ratio = loss_floor_gap / max(1e-6, abs(mean_loss))

        is_unstable = (std_score > 0.08) or (loss_gap_ratio > 0.3) or (std_best_iter > 15)

        if is_unstable:
            self.add_recommendation(
                "Stability Checks",
                "Dataset convergence is unstable across subsamples, noisy, heterogeneous, or inconsistent signal"
            )

        return {
            "mean_score": mean_score,
            "std_score": std_score,
            "mean_best_iteration": mean_best_iter,
            "std_best_iteration": std_best_iter,
            "mean_final_loss": mean_loss,
            "std_final_loss": std_loss,
            "loss_floor_gap": loss_floor_gap,
            "loss_gap_ratio": loss_gap_ratio,
            "is_unstable": is_unstable,
            "n_runs": n_runs,
            "sample_fraction": sample_frac
        }
    
    def unlabeled_diagnostics(self, df: pd.DataFrame, col_views: Dict):
        self.component_scores["trainability"] = 100

        numeric_cols = col_views.get("numeric_features", [])
        categorical_cols = col_views.get("categorical_features", [])
        identifier_cols = col_views.get("identifiers", [])
        entity_keys = col_views.get("entity_keys", [])

        usable_cols = [c for c in numeric_cols + categorical_cols
                    if c not in identifier_cols and c not in entity_keys]

        if len(usable_cols) == 0:
            self.component_scores["trainability"] = 0
            self.add_penalty("Unlabeled Diagnostics", "No usable features for unsupervised diagnostics")
            return {
                "status": "failed",
                "issue": "No usable features found",
                "component_scores": self.component_scores
            }

        n_rows = df.shape[0]
        n_features = len(usable_cols)

        if n_rows < 200:
            self.component_scores["trainability"] -= 25
            self.add_penalty("Unlabeled Diagnostics", "Dataset is too small for reliable clustering/anomaly detection")

        if n_features > n_rows:
            self.component_scores["trainability"] -= 20
            self.add_penalty("Unlabeled Diagnostics", "More features than rows, clustering will be unreliable")
            self.add_recommendation("Unlabeled Diagnostics", "Reduce dimensionality (PCA) or drop weak features")

        if identifier_cols or entity_keys:
            self.add_recommendation(
                "Unlabeled Diagnostics",
                f"Identifier-like columns excluded: {list(set(identifier_cols + entity_keys))}"
            )

        import gc 
        gc.collect()
        X = df[usable_cols].copy()

        MAX_CELLS = 500_000
        if X.shape[0] * X.shape[1] > MAX_CELLS:
            max_rows = MAX_CELLS // X.shape[1]
            X = X.sample(n=max_rows, random_state = 42)
        for c in numeric_cols:
            if c in X.columns:
                X[c] = X[c].fillna(X[c].median())

        if X.shape[1] > 50:
            variances = X.var(numeric_only = True)
            top_cols = variances.nlargest(50).index.tolist()
            X = X[top_cols]

        for c in categorical_cols:
            if c in X.columns:
                X[c] = pd.factorize(X[c].fillna("NA").astype(str))[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        del(X)
        gc.collect()

        binary_cols = [c for c in usable_cols if df[c].dropna().nunique() <= 2]
        binary_ratio = len(binary_cols) / len(usable_cols) if usable_cols else 0

        anomaly_report = self.run_anomaly_detection(X_scaled, binary_ratio=binary_ratio)
        clustering_report = self.run_clustering_diagnostics(X_scaled)
        stability_report = self.run_unlabeled_stability(X_scaled)

        ml_readiness_score = int(
            np.mean([
                self.component_scores["data_health"],
                self.component_scores["trainability"]
            ])
        )

        return {
            "status": "ok",
            "mode": "unlabeled",
            "anomaly_detection": anomaly_report,
            "clustering": clustering_report,
            "stability": stability_report,
            "penalties": self.penalty_buckets,
            "recommendations": self.recommendation_buckets,
            "component_scores": self.component_scores,
            "ML_readiness_score": ml_readiness_score
        }


    def run_anomaly_detection(self, X_scaled, binary_ratio: float = 0.0):
        iso = IsolationForest(
            n_estimators=50,
            contamination="auto",
            random_state=42,

        )

        preds = iso.fit_predict(X_scaled)
        scores = iso.decision_function(X_scaled)

        anomaly_pct = float((preds == -1).mean() * 100)
        predominantly_binary = binary_ratio > 0.5

        if predominantly_binary:
            # Don't penalize, just warn
            self.add_recommendation(
                "Anomaly Detection",
                f"Anomaly detection less reliable on predominantly binary features ({int(binary_ratio*100)}% binary columns). Results may not reflect true data quality."
            )
        elif anomaly_pct > self.cfg["anomaly_rate_threshold"] * 100 * 2:
            self.component_scores["trainability"] -= 35
            self.add_penalty(
                "Anomaly Detection",
                f"Very high anomaly rate ({anomaly_pct:.1f}%), data may be corrupted/noisy"
            )
            self.add_recommendation(
                "Anomaly Detection",
                "Check for corrupted rows, extreme outliers, mixed regimes, or wrong scaling"
            )
        elif anomaly_pct > self.cfg["anomaly_rate_threshold"] * 100:
            self.component_scores["trainability"] -= 15
            self.add_penalty(
                "Anomaly Detection",
                f"High anomaly rate ({anomaly_pct:.1f}%) detected"
            )
            self.add_recommendation(
                "Anomaly Detection",
                "Consider outlier handling, robust scaling, or filtering rare regimes"
            )
        elif anomaly_pct < 1:
            self.add_recommendation(
                "Anomaly Detection",
                "Very low anomaly rate, dataset seems clean"
            )
        
        max_points = 5000
        if len(scores) > max_points:
            idx = np.random.choice(len(scores), max_points, replace=False)
            score_sample = scores[idx].tolist()
        else:
            score_sample = scores.tolist()

        return {
            "method": "IsolationForest",
            "anomaly_percent": anomaly_pct,
            "mean_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "predominantly_binary": predominantly_binary,
            "score_distribution": [round(float(s), 4) for s in score_sample],
        }

    def run_clustering_diagnostics(self, X_scaled):
        best_k = None
        best_score = -1
        scores = {}

        max_samples = 10000
        if X_scaled.shape[0] > max_samples:
            idx = np.random.choice(X_scaled.shape[0], max_samples, replace=False)
            X_metric = X_scaled[idx]
        else:
            X_metric = X_scaled

        for k in range(2, 6):
            km = MiniBatchKMeans(
                n_clusters=k,
                random_state=42,
                n_init=3,
                batch_size=1024
            )

            km.fit(X_metric)
            labels_metric = km.predict(X_metric)

            sil = silhouette_score(X_metric, labels_metric)
            scores[k] = float(sil)

            if sil > best_score:
                best_score = sil
                best_k = k

        if best_score < 0.10:
            self.component_scores["trainability"] -= 30
            self.add_penalty(
                "Clustering",
                f"Almost no cluster structure (best silhouette={best_score:.3f})"
            )
            self.add_recommendation(
                "Clustering",
                "Try PCA/UMAP, remove noisy features, or check if clustering is meaningful for this dataset"
            )

        elif best_score < 0.25:
            self.component_scores["trainability"] -= 15
            self.add_penalty(
                "Clustering",
                f"Weak cluster structure (best silhouette={best_score:.3f})"
            )
            self.add_recommendation(
                "Clustering",
                "Clustering may not produce clean groups, try feature engineering or dimensionality reduction"
            )

        else:
            self.add_recommendation(
                "Clustering",
                f"Dataset shows usable clustering structure (best_k={best_k}, silhouette={best_score:.3f})"
            )

        return {
            "best_k": best_k,
            "best_silhouette": float(best_score),
            "silhouette_scores": scores
        }


    def run_unlabeled_stability(self, X_scaled, n_runs=3, sample_frac=0.8):
        sil_scores = []
        max_stability_samples = 5000

        for _ in range(n_runs):
            sample_size = int(len(X_scaled) * sample_frac)
            idx = np.random.choice(len(X_scaled), min(sample_size, 20000), replace=False)
            X_sub = X_scaled[idx]

            metric_idx = np.random.choice(
                len(X_sub),
                min(len(X_sub), max_stability_samples),
                replace=False
            )
            X_metric = X_sub[metric_idx]

            km = MiniBatchKMeans(n_clusters=3, random_state=None, n_init=3)
            km.fit(X_sub)

            labels_metric = km.predict(X_metric)
            sil = silhouette_score(X_metric, labels_metric)
            sil_scores.append(float(sil))

        mean_sil = float(np.mean(sil_scores))
        std_sil = float(np.std(sil_scores))

        unstable = std_sil > 0.10

        if unstable:
            self.component_scores["trainability"] -= 20
            self.add_penalty(
                "Unlabeled Stability",
                f"Clustering structure unstable (std silhouette={std_sil:.3f})"
            )
            self.add_recommendation(
                "Unlabeled Stability",
                "Data may contain mixed regimes or noisy signal, try PCA or split dataset by segments/time"
            )
        else:
            self.add_recommendation(
                "Unlabeled Stability",
                f"Clustering stability looks consistent (std silhouette={std_sil:.3f})"
            )

        return {
            "mean_silhouette": mean_sil,
            "std_silhouette": std_sil,
            "unstable": unstable,
            "n_runs": n_runs,
            "sample_fraction": sample_frac
        }