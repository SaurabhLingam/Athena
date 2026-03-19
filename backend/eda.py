import re
import pandas as pd
import numpy as np
class EdaToolKit:
    def __init__(self, cfg: dict = None):
        try:
            from backend.profiles import get_profile
        except ImportError:
            from profiles import get_profile
        self.cfg = cfg or get_profile("standard")
    def basicdatainfo(self, df: pd.DataFrame ):
        rows = df.shape[0]
        cols = df.shape[1]
        col_names = df.columns.tolist()
        unique_values = df.nunique()

        return {
            "rows": rows,
            "cols": cols,
            "col_names": col_names,
            "unique_values": unique_values.to_dict()
        }
    @staticmethod
    def safe_float(x):
        if x is None:
            return None
        if isinstance(x, (np.floating, float)):
            if np.isnan(x) or np.isinf(x):
                return None
            return float(x)
        return x
    
    def build_column_views(self, df: pd.DataFrame, target_col: str | None = None):
        target = [target_col] if target_col and target_col in df.columns else []

        identifiers = []
        numeric_features = []
        categorical_features = []
        entity_keys = []
        free_text_cols = []

        n_rows = len(df)
        for col in df.columns:
            if col in target:
                continue

            s = df[col]
            n_unique = s.nunique(dropna=True)
            unique_ratio = n_unique / n_rows
            is_numeric = pd.api.types.is_numeric_dtype(s)
            is_integer = is_numeric and (s.dropna() % 1 == 0).all()



            if unique_ratio >= 0.96:
                identifiers.append(col)
                
                continue
            if is_numeric and is_integer and unique_ratio > 0.5:
                entity_keys.append(col)
                
                continue
            if is_numeric and is_integer:
                non_zero = s.dropna()
                non_zero = non_zero[non_zero != 0]
                p5 = float(non_zero.quantile(0.05)) if len(non_zero) > 0 else 0
                if p5 > 100 and n_unique > 50:
                    categorical_features.append(col)
                    
                    continue

            if not is_numeric and 0.02 < unique_ratio < 0.95:
                sample = s.dropna().astype(str).head(100)
                avg_words = sample.apply(lambda x: len(x.split())).mean()
                avg_length = sample.apply(len).mean()
                word_count_std = sample.apply(lambda x: len(x.split())).std()

                address_pattern = re.compile(r'^\d+\s+\w+', re.IGNORECASE)
                looks_like_address = sample.apply(lambda x: bool(address_pattern.match(x))).mean() > 0.7
                if looks_like_address:
                    entity_keys.append(col)
                    continue
                if avg_words > 3 or avg_length > 20 or (avg_length > 15 and unique_ratio > 0.7):
                    if unique_ratio > 0.8 and avg_words < 8 and avg_length < 50 and word_count_std < 3:
                        entity_keys.append(col)
                    else:
                        free_text_cols.append(col)
                else:
                    entity_keys.append(col)
                continue

            if not is_numeric:
                categorical_features.append(col)
                continue

            if is_integer and n_unique <= 30:
                sample_vals = set(s.dropna().unique())
                already_encoded = sample_vals.issubset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
                if already_encoded:
                    numeric_features.append(col)
                else:
                    categorical_features.append(col)
                continue

            numeric_features.append(col)

        for col in free_text_cols[:]:
            sample = df[col].dropna().astype(str).head(20)
            coord_pattern = re.compile(r'^\s*\(?\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+')
            if sample.apply(lambda x: bool(coord_pattern.match(x))).mean() > 0.7:
                free_text_cols.remove(col)
                identifiers.append(col)


        ID_SEGMENTS = {"id", "key", "code", "num", "no", "number", "postal",
                       "permit", "fips", "idx", "seq", "ref", "oid"}

        ID_FULLNAMES = {"uuid", "guid", "zipcode", "postcode", "fips", "oid",
                        "uid", "nid", "pid", "rid", "sid", "cid"}

        ID_SUFFIXES = ("_id", "_Id", "_ID", "_idx", "_Idx", "_cd", "_Cd")

        reclassify_as_identifier = []
        for col in categorical_features + numeric_features:
            col_lower = col.lower().strip()
            col_parts = set(col_lower.replace("-", "_").replace(" ", "_").split("_"))

            segment_match = bool(col_parts & ID_SEGMENTS)
            fullname_match = col_lower in ID_FULLNAMES
            suffix_match = col.endswith(ID_SUFFIXES)
            id_word_match = col_lower == "id" or (
                col_lower.endswith("id") and (len(col_lower) == 2 or col_lower[-3] in ("_", "-", " "))
            )

            if segment_match or fullname_match or suffix_match or id_word_match:
                reclassify_as_identifier.append(col)

        for col in reclassify_as_identifier:
            if col in categorical_features:
                categorical_features.remove(col)
            elif col in numeric_features:
                numeric_features.remove(col)
            identifiers.append(col)
            
        return {
            "target": target,
            "identifiers": identifiers,
            "entity_keys": entity_keys,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "free_text_cols": free_text_cols

        }
    
    def missingdata(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
        col_threshold: float = 30.0,
        row_threshold=(30, 50, 70),
        cooccur_threshold: float = 0.134111,
        min_missing_frac: float = 0.05,
    ):
        col_views = self.build_column_views(df, target_col)
        feature_cols = col_views["numeric_features"] + col_views["categorical_features"]
        feature_df = df[feature_cols]
        nrows, ncols = feature_df.shape  # use feature_df, not original df

        # Column-wise missingness
        missing_vals_cols = feature_df.isnull().sum()
        missing_values_percent_cols = (missing_vals_cols / nrows * 100).astype(float)
        flagged_cols = missing_values_percent_cols[
            missing_values_percent_cols > col_threshold
        ].index.tolist()

        # Row-wise missingness
        missing_vals_rows = feature_df.isnull().sum(axis=1)
        missing_values_percent_rows = (missing_vals_rows / ncols * 100).astype(float)  # <- enforce float
        flagged_rows = {
            f"rows_gt_{t}_pct": int((missing_values_percent_rows > t).sum())
            for t in row_threshold
        }
        severe_rows = missing_values_percent_rows[
            missing_values_percent_rows > max(row_threshold)
        ].index.tolist()

        # Row missing distribution
        bins = [-1, 0, 10, 30, 50, 100]
        labels = ["0% Missing", "1–10%", "11–30%", "31–50%", ">50%"]
        row_missing_distribution = (
            pd.cut(
                missing_values_percent_rows, bins=bins, labels=labels, include_lowest=True
            )
            .value_counts()
            .sort_index()
            .to_dict()
        )

        # Systematic missingness (co-occurrence)
        missing_matrix = feature_df.isnull().astype(int)
        valid_cols = missing_matrix.columns[missing_matrix.mean() >= min_missing_frac]
        filtered = missing_matrix[valid_cols]

        coocur_pairs = []
        if filtered.shape[1] >= 2:
            corr = filtered.corr()
            corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr = corr.stack().reset_index()
            high_corr.columns = ["col_a", "col_b", "correlation"]

            high_corr = high_corr[high_corr["correlation"].abs() >= cooccur_threshold]
            coocur_pairs = high_corr.sort_values(by="correlation", ascending=False).to_dict(
                orient="records"
            )

        return {
            "missing_values_percent_cols": missing_values_percent_cols.to_dict(),
            "flagged_columns": flagged_cols,
            "row_missingness_flags": flagged_rows,
            "severely_incomplete_rows": severe_rows[:50],
            "row_missing_distribution": row_missing_distribution,
            "systematic_missingness": coocur_pairs,
        }
    
    def numeric_summary(self, df: pd.DataFrame, target_col: str | None = None):
        col_views = self.build_column_views(df, target_col)
        numeric_cols = col_views["numeric_features"]
        numeric_df = df[numeric_cols].copy()
        numeric_df = numeric_df.loc[:, numeric_df.nunique() != 1]
        if numeric_df.empty:
            return []
        stats = numeric_df.agg([
            'count', 'mean', 'std', 'min', 'max',
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75),
            lambda x: x.skew(),
            lambda x: (x == 0).mean()
        ]).T

        stats.columns = ['count', 'mean', 'std', 'min', 'max', 'Q1', 'Q3', 'skewness', 'zero_pct']

        stats['iqr'] = stats['Q3'] - stats['Q1']
        stats['missing_pct'] = (df[numeric_cols].isnull().mean() * 100).to_dict()

        return stats.reset_index().rename(columns={'index': 'column'}).to_dict('records')

    def categorical_summary(self, df: pd.DataFrame, target_col: str | None = None):
        col_views = self.build_column_views(df, target_col)
        cat_cols = col_views["categorical_features"]
        if not cat_cols:
            return []
        
        summary = []

        for col in cat_cols:
            series = df[col].fillna("MISSING")
            value_counts=  series.value_counts(normalize = True)
            top_categories = value_counts.head(5).to_dict()
            most_frequent_pct = float(value_counts.iloc[0]) if len(value_counts) > 0 else 0.0

            summary.append({
                "column": col,
                "n_unique": int(series.nunique()),
                "top_categories": top_categories,
                "missing_pct": float(df[col].isnull().mean() * 100),
                "most_frequent_pct": most_frequent_pct,
                "high_cardinality": series.nunique() > self.cfg["high_cardinality_threshold"]
            })

        return summary
    
    
    def feature_correlation(self, df: pd.DataFrame, target_col: str | None = None):
        col_views = self.build_column_views(df, target_col)

        if not target_col or target_col not in df.columns:
            raise ValueError("target_col is required for supervised analysis")

        target = df[target_col]
        target_numeric = pd.to_numeric(target, errors="coerce")
        if target_numeric.isna().all():
            codes, uniques = pd.factorize(target)
            
            majority_class = target.value_counts().idxmax()
            majority_code = list(uniques).index(majority_class)
            if majority_code == 0:
                codes = 1 - codes  
            target_numeric = pd.Series(codes, index=target.index, dtype=float)
        is_regression = bool(target_numeric.nunique() > 10)

        # --- Only numeric columns (avoid strings like host_name) ---
        num_corr_cols = [
            col for col in (col_views["numeric_features"] + col_views.get("entity_keys", []))
            if pd.api.types.is_numeric_dtype(df[col])
        ]

        cat_cols = col_views["categorical_features"]
        result = {}


        numeric_target_corr = {}
        for col in num_corr_cols:
            numeric_target_corr[col] = {
                "pearson": float(df[col].corr(target_numeric)),
                "spearman": float(df[col].corr(target_numeric, method="spearman"))
            }

        result["numeric_target_correlation"] = numeric_target_corr


        categorical_impact = []
        if is_regression:
            global_mean = float(target_numeric.mean())
            global_var = float(target_numeric.var())

            for col in cat_cols:
                grp_mean = df.groupby(col)[target_col].mean()
                max_shift = float((grp_mean - global_mean).abs().max())
                explained_var = float(grp_mean.var() / global_var) if global_var > 0 else 0.0

                categorical_impact.append({
                    "column": col,
                    "max_mean_shift": max_shift,
                    "explained_variance_ratio": explained_var,
                    "likely_influential": bool(explained_var > 0.02)
                })
        else:
            global_dist = target.value_counts(normalize=True)
            for col in cat_cols:
                max_shift = 0.0
                for _, group in df.groupby(col):
                    local_dist = group[target_col].value_counts(normalize=True)
                    shift = float((local_dist - global_dist).abs().sum() / 2)
                    max_shift = max(max_shift, shift)

                categorical_impact.append({
                    "column": col,
                    "distribution_shift": max_shift,
                    "likely_influential": bool(max_shift > 0.3)
                })

        result["categorical_target_impact"] = categorical_impact
        return result
    
    def feature_redundancy(self, df: pd.DataFrame, target_col: str | None = None):
        col_views = self.build_column_views(df, target_col)
        numeric_cols = col_views["numeric_features"]

        if not numeric_cols or len(numeric_cols) < 2:
            return {}
        df_sample = df[numeric_cols]
        if len(df_sample) > 50_000:
            df_sample = df_sample.sample(n=50_000, random_state=42)
        corr_matrix = df_sample.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        high_corr_pairs = []

        for i, j in zip(*np.where(upper > self.cfg["correlation_threshold"])):
            corr_val = float(upper.iloc[i, j])
            high_corr_pairs.append({
                "feature_1": numeric_cols[i],
                "feature_2": numeric_cols[j],
                "correlation": corr_val,
                "interpretation": "Likely measuring overlapping or scale-related information",
                "recommended_action": (
                    "Consider dropping one for linear models; safe for tree-based models"
                )
            })

        if not high_corr_pairs:
            overall_risk = "Low"
        elif any(pair["correlation"] > self.cfg["correlation_threshold"] for pair in high_corr_pairs):
            overall_risk = "High"
        else:
            overall_risk = "Moderate"

        return {
            "highly_correlated_pairs": high_corr_pairs,
            "overall_redundancy_risk": overall_risk
        }
    
    def outlier_detection(self, df: pd.DataFrame, target_col: str | None = None):
        col_views = self.build_column_views(df, target_col)
        numeric_cols = col_views["numeric_features"]

        if not numeric_cols:
            return {}

        results = []
        heavy_tailed = []
        safe_features = []

        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty or series.nunique() < 10:
                continue

            skew = self.safe_float(series.skew())
            kurt = self.safe_float(series.kurtosis())

            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_pct = self.safe_float(
                ((series < lower) | (series > upper)).mean() * 100
            )

            is_heavy_tailed = bool(kurt is not None and abs(kurt) > 3)

            if is_heavy_tailed:
                heavy_tailed.append(col)
            else:
                safe_features.append(col)

            results.append({
                "column": col,
                "skewness": round(skew, 3) if skew is not None else None,
                "kurtosis": round(kurt, 3) if kurt is not None else None,
                "outlier_pct": round(outlier_pct, 2) if outlier_pct is not None else None,
                "distribution_type": (
                    "Gaussian-like"
                    if skew is not None and kurt is not None and abs(skew) < 0.5 and abs(kurt) < 1
                    else "Heavy-tailed / Skewed"
                ),
                "suggest_log_transform": bool(
                    is_heavy_tailed and skew is not None and skew > 1.0
                ),
                "suggest_robust_loss": bool(
                    is_heavy_tailed and skew is not None and abs(skew) <= 1.0
                )
            })

        return {
            "column_outliers": results,
            "dataset_outlier_diagnosis": {
                "heavy_tailed_features": heavy_tailed,
                "safe_features": safe_features,
                "outliers_are_signal": len(heavy_tailed) > 0
            }
        }
    
    def feature_similarity(self, df: pd.DataFrame, target_col: str | None = None):
        col_views = self.build_column_views(df, target_col)
        numeric_cols = col_views["numeric_features"]

        if not numeric_cols or len(numeric_cols) < 2:
            return {}

        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics.pairwise import cosine_similarity

        X = df[numeric_cols].copy()

        if len(X) > 50_000:
            X = X.sample(n=50_000, random_state=42)


        X = X.fillna(X.median())


        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        similarity_matrix = cosine_similarity(X_scaled.T)

        similar_pairs = []

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                sim = similarity_matrix[i, j]

                if sim > 0.85:
                    similar_pairs.append({
                        "feature_1": numeric_cols[i],
                        "feature_2": numeric_cols[j],
                        "cosine_similarity": round(float(sim), 3),
                        "relationship_strength": (
                            "very_high" if sim > 0.95 else "high"
                        ),
                        "ml_implication": (
                            "multicollinearity_risk" if sim > 0.9 else "redundant_information"
                        )
                    })

        return {
            "highly_similar_features": similar_pairs,
            "summary": {
                "potential_multicollinearity": any(
                    p["cosine_similarity"] > 0.9 for p in similar_pairs
                )
            }
        }
        
    def target_insights(self, df: pd.DataFrame, target_col: str | None = None):

        if not target_col or target_col not in df.columns:
            raise ValueError("target_col is required for supervised analysis")

        col_views = self.build_column_views(df, target_col)
        target = df[target_col]

        is_numeric_target = bool(pd.api.types.is_numeric_dtype(target))
        is_classification = bool((not is_numeric_target) or (target.nunique() <= 20))

        insights = {
            "target_name": target_col,
            "missing_pct": float(target.isnull().mean() * 100),
        }

        if is_classification:
            balance = target.value_counts(normalize=True)

            max_ratio = float(balance.max())
            min_ratio = float(balance.min()) if balance.min() > 0 else None

            insights.update({
                "task_type": "classification",
                "class_distribution": balance.astype(float).to_dict(),
                "imbalance_ratio": float(max_ratio / min_ratio) if min_ratio else None,
                "is_imbalanced": bool(max_ratio > 0.7)
            })

            numeric_features = [
                c for c in col_views["numeric_features"] if c != target_col
            ][:5]

            insights["numeric_feature_separation"] = {
                col: df.groupby(target_col)[col]
                    .agg(["mean", "std"])
                    .astype(float)
                    .round(3)
                    .to_dict()
                for col in numeric_features
            }

        else:
            insights.update({
                "task_type": "regression",
                "target_stats": {
                    "mean": float(target.mean()),
                    "std": float(target.std()),
                    "skewness": float(target.skew()),
                    "kurtosis": float(target.kurtosis())
                },
                "likely_transform_needed": bool(abs(target.skew()) > 1.0)
            })

        return insights
    
    def data_health_report(self, df: pd.DataFrame, target_col: str | None = None):
        n_rows, n_cols = df.shape
        col_views = self.build_column_views(df, target_col)

        numeric_features = col_views["numeric_features"]
        categorical_features = col_views["categorical_features"]
        identifiers = set(col_views["identifiers"])
        entity_keys = set(col_views.get("entity_keys", []))  # don't drop these

        critical_alerts = []
        drop_columns = set()
        log_transform = set()
        log1p_transform = set()
        categorical_warnings = []
        redundant_features = []


        for col in numeric_features:
            series = df[col]
            uniqueness_ratio = series.nunique(dropna=False) / max(1, n_rows)


            if col in identifiers:
                drop_columns.add(col)
                continue

            if bool(series.is_monotonic_increasing) and uniqueness_ratio > 0.95:
                critical_alerts.append(f"Index-like monotonic identifier: '{col}'")
                drop_columns.add(col)
                continue

            if series.nunique(dropna=False) <= 1:
                drop_columns.add(col)
                continue

            if float(series.isnull().mean()) > 0.7:
                drop_columns.add(col)
                continue

            if float((series == 0).mean()) > 0.95:
                n_unique = series.nunique(dropna=True)
                if n_unique <= 2:
                    continue
                drop_columns.add(col)


        for col in numeric_features:
            if col in drop_columns:
                continue

            series = df[col].fillna(0)
            if series.nunique() < 10:
                continue

            full_skew = float(series.skew())
            zero_ratio = float((series == 0).mean())
            val_min = float(series.min())
            val_max = float(series.max())
            is_bounded_range = (val_min >= -180 and val_max <= 180) and (series.mean() != 0)

            if full_skew > self.cfg["skew_threshold"] and not is_bounded_range:
                if zero_ratio > 0:
                    log1p_transform.add(col)
                else:
                    log_transform.add(col)

        if len(numeric_features) >= 2:
            df_num = df[numeric_features]
            if len(df_num) > 50_000:
                df_num = df_num.sample(n=50_000, random_state=42)
            corr_matrix = df_num.corr().abs()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > 0.98:
                        col_to_drop = corr_matrix.columns[i]
                        base_col = corr_matrix.columns[j]

                        if col_to_drop not in drop_columns:
                            redundant_features.append({
                                "redundant_column": col_to_drop,
                                "base_column": base_col,
                                "correlation": float(corr_val)
                            })
                            critical_alerts.append(
                                f"Redundant feature: '{col_to_drop}' is {corr_val:.2f} correlated with '{base_col}'"
                            )

        for col in categorical_features:
            series = df[col].astype(str)
            unique_ratio = series.nunique() / max(1, n_rows)

            if unique_ratio > 0.8:
                categorical_warnings.append({
                    "column": col,
                    "issue": "high_cardinality",
                    "unique_ratio": unique_ratio,
                    "recommendation": "Target encoding, hashing, or drop"
                })

            value_dist = series.value_counts(normalize=True)
            if not value_dist.empty and value_dist.iloc[0] > 0.99:
                categorical_warnings.append({
                    "column": col,
                    "issue": "near_zero_variance",
                    "dominant_class_pct": float(value_dist.iloc[0]),
                    "recommendation": "Drop or group rare categories"
                })


        effective_drops = len(drop_columns)
        data_health_score = int(100 * (1 - (effective_drops / max(1, n_cols))))

        return {
            "critical_alerts": sorted(set(critical_alerts)),
            "action_plan": {
                "drop_columns": sorted(drop_columns),
                "apply_log_transform": sorted(log_transform),
                "apply_log1p_transform": sorted(log1p_transform)
            },
            "feature_redundancy": redundant_features,
            "categorical_health": categorical_warnings,
            "data_structural_score": max(0, data_health_score)
        }
    
    def free_text_detection(self, df: pd.DataFrame, target_col: str | None = None):
        text_cols = []
        for col in df.columns:
            if col == target_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            sample = df[col].dropna().astype(str)
            if len(sample) == 0:
                continue

            avg_len = sample.str.len().mean()
            avg_words = sample.apply(lambda x: len(x.split())).mean()
            unique_ratio = sample.nunique() / max(1, len(sample))
            word_count_std = sample.apply(lambda x: len(x.split())).std()
            is_free_text = (avg_len > 20 or (avg_len > 15 and unique_ratio > 0.7)) and unique_ratio > 0.1
            if unique_ratio > 0.8 and avg_words < 8 and avg_len < 50 and word_count_std < 3:
                is_free_text = False


        

            if is_free_text:
                text_cols.append({
                    "column": col,
                    "avg_length": round(float(avg_len), 1),
                    "unique_ratio": round(float(unique_ratio), 3),
                    "recommendation": "Exclude from ML features or apply NLP embeddings. Not suitable for standard encoding."
                })

        return {
            "free_text_columns": text_cols,
            "count": len(text_cols)
        }


    def bivariate_analysis(self, df: pd.DataFrame, target_col: str):
        if not target_col or target_col not in df.columns:
            raise ValueError("target_col is required for bivariate analysis")

        col_views = self.build_column_views(df, target_col)
        numeric_cols = col_views["numeric_features"][:6]
        target = df[target_col]

        is_classification = not pd.api.types.is_numeric_dtype(target) or target.nunique() <= 20

        results = []

        for col in numeric_cols:
            series = df[col].dropna()
            if series.nunique() < 5:
                continue

            try:
                buckets = pd.qcut(df[col], q=5, duplicates="drop")
            except Exception:
                continue

            if is_classification:
                grouped = df.groupby(buckets, observed=True)[target_col].value_counts(normalize=True).unstack(fill_value=0)
                grouped.index = grouped.index.astype(str)  # ← add this
                results.append({
                    "feature": col,
                    "type": "classification",
                    "buckets": grouped.round(3).reset_index().to_dict(orient="records")
                })
            else:
                grouped = df.groupby(buckets, observed=True)[target_col].agg(["mean", "std"]).round(3)
                grouped.index = grouped.index.astype(str)  # ← add this
                results.append({
                    "feature": col,
                    "type": "regression",
                    "buckets": grouped.reset_index().rename(columns={col: "bucket"}).to_dict(orient="records")
                })

        return {
            "bivariate_results": results,
            "target_col": target_col,
            "task_type": "classification" if is_classification else "regression"
        }
    
    def nlp_analysis(self, df: pd.DataFrame, text_cols: list, target_col: str | None = None) -> dict:
        import re
        from collections import Counter
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        stop_words = set(stopwords.words('english'))
        results = []

        for col in text_cols:
            series = df[col].dropna().astype(str)

            # Cap at 5000 rows for performance
            if len(series) > 5000:
                series = series.sample(5000, random_state=42)

            total = len(series)

            # ── Text Statistics ──────────────────────────────────────
            word_counts = series.apply(lambda x: len(x.split()))
            char_counts = series.apply(len)

            all_words = []
            for text in series:
                text = re.sub(r'<[^>]+>', ' ', text)
                tokens = word_tokenize(text.lower())
                all_words.extend([w for w in tokens if w.isalpha()])

            vocab = set(all_words)
            filtered_words = [w for w in all_words if w not in stop_words]
            top_words = [{"word": w, "count": c} for w, c in Counter(filtered_words).most_common(15)]

            lexical_diversity = round(len(vocab) / len(all_words), 4) if all_words else 0

            # ── Text Quality ─────────────────────────────────────────
            empty_ratio = float((series.str.strip() == "").mean())

            hashes = series.apply(lambda x: hash(x.strip().lower()))
            duplicate_ratio = float(1 - hashes.nunique() / total)

            url_pattern = re.compile(r'http\S+|www\.\S+')
            url_ratio = float(series.apply(lambda x: bool(url_pattern.search(x))).mean())

            number_ratio = float(series.apply(lambda x: bool(re.search(r'\d', x))).mean())

            special_char_ratio = float(series.apply(
                lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x)) / max(len(x), 1)
            ).mean())

            # ── ML Readiness ─────────────────────────────────────────
            vocab_size = len(vocab)
            if vocab_size < 100:
                vectorization_recommendation = "Vocabulary too small for embeddings. Use simple TF-IDF or bag-of-words."
            elif vocab_size < 1000:
                vectorization_recommendation = "Small vocabulary. TF-IDF recommended."
            elif total < 10000:
                vectorization_recommendation = "Medium dataset. TF-IDF or fine-tuned embeddings (Word2Vec, FastText)."
            else:
                vectorization_recommendation = "Large dataset. Consider BERT/sentence-transformers for best results."

            col_result = {
                "column": col,
                "text_stats": {
                    "avg_word_count": round(float(word_counts.mean()), 2),
                    "avg_char_length": round(float(char_counts.mean()), 2),
                    "max_word_count": int(word_counts.max()),
                    "min_word_count": int(word_counts.min()),
                    "vocab_size": vocab_size,
                    "lexical_diversity": lexical_diversity,
                    "top_words": top_words,
                },
                "text_quality": {
                    "empty_ratio": round(empty_ratio, 4),
                    "duplicate_ratio": round(duplicate_ratio, 4),
                    "url_ratio": round(url_ratio, 4),
                    "number_ratio": round(number_ratio, 4),
                    "special_char_ratio": round(special_char_ratio, 4),
                },
                "ml_readiness": {
                    "vocab_size": vocab_size,
                    "vectorization_recommendation": vectorization_recommendation,
                    "suitable_for_embeddings": vocab_size >= 500,
                }
            }

            # ── Per-class analysis (labeled only) ────────────────────
            if target_col and target_col in df.columns:
                class_stats = []
                for cls in df[target_col].dropna().unique():
                    cls_texts = df[df[target_col] == cls][col].dropna().astype(str)
                    # Cap per class
                    if len(cls_texts) > 1000:
                        cls_texts = cls_texts.sample(1000, random_state=42)

                    cls_words = []
                    for text in cls_texts:
                        text = re.sub(r'<[^>]+>', ' ', text)
                        tokens = word_tokenize(text.lower())
                        cls_words.extend([w for w in tokens if w.isalpha() and w not in stop_words])

                    top_cls_words = [{"word": w, "count": c} for w, c in Counter(cls_words).most_common(10)]
                    class_stats.append({
                        "class": str(cls),
                        "count": len(cls_texts),
                        "avg_word_count": round(float(cls_texts.apply(lambda x: len(x.split())).mean()), 2),
                        "avg_char_length": round(float(cls_texts.apply(len).mean()), 2),
                        "top_words": top_cls_words,
                    })

                col_result["per_class_analysis"] = class_stats

            results.append(col_result)

        return {
            "nlp_analysis": results,
            "text_col_count": len(text_cols)
        }