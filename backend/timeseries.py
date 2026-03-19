import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


class TimeSeriesKit:

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
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            if np.isnan(x) or np.isinf(x):
                return None
            return float(x)
        return x

    @staticmethod
    def safe_list(arr):
        result = []
        for x in arr:
            try:
                if isinstance(x, np.integer):
                    result.append(int(x))
                    continue
                v = float(x)
                result.append(None if (np.isnan(v) or np.isinf(v)) else round(v, 4))
            except Exception:
                result.append(None)
        return result


    def detect_datetime_cols(self, df: pd.DataFrame) -> list[str]:
        candidates = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                candidates.append(col)
                continue
            if df[col].dtype == object:
                sample = df[col].dropna().head(50)
                try:
                    parsed = pd.to_datetime(sample, infer_datetime_format=True)
                    if parsed.notna().mean() > 0.8:
                        candidates.append(col)
                except Exception:
                    pass
        return candidates

    def detect_target_col(self, df: pd.DataFrame, dt_col: str) -> str | None:
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != dt_col
        ]
        if not numeric_cols:
            return None
        return max(numeric_cols, key=lambda c: df[c].var())

    def get_candidates(self, df: pd.DataFrame) -> dict:
        dt_cols  = self.detect_datetime_cols(df)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return {
            "datetime_candidates": dt_cols,
            "target_candidates":   num_cols,
            "auto_datetime":       dt_cols[0] if dt_cols else None,
            "auto_target":         self.detect_target_col(df, dt_cols[0]) if dt_cols else None,
        }


    def prepare_series(self, df: pd.DataFrame, dt_col: str, tgt_col: str) -> pd.Series:
        tmp = df[[dt_col, tgt_col]].copy()
        tmp[dt_col] = pd.to_datetime(tmp[dt_col], infer_datetime_format=True)
        tmp = tmp.dropna(subset=[dt_col])
        tmp = tmp.sort_values(dt_col).set_index(dt_col)
        series = tmp[tgt_col].dropna()
        freq = pd.infer_freq(series.index)
        if freq:
            series = series.asfreq(freq)
        return series

    def infer_freq(self, ts: pd.Series) -> str:
        try:
            return pd.infer_freq(ts.index) or "unknown"
        except Exception:
            return "unknown"



    def run_timeseries(
        self,
        df: pd.DataFrame,
        datetime_col: str | None = None,
        target_col: str | None = None,
    ) -> dict:
        """
        Called from /analyze. Auto-detects datetime + target if not passed.
        Returns full time series EDA + baseline results.
        """
        dt_col = datetime_col or (self.detect_datetime_cols(df) or [None])[0]
        if dt_col is None:
            return {"detected": False, "reason": "No datetime column found."}

        tgt_col = target_col or self.detect_target_col(df, dt_col)
        if tgt_col is None:
            return {"detected": False, "reason": "No numeric target column found."}

        try:
            ts = self.prepare_series(df, dt_col, tgt_col)
        except Exception as e:
            return {"detected": False, "reason": f"Could not prepare time series: {str(e)}"}

        if len(ts) < 10:
            return {"detected": False, "reason": "Time series too short (< 10 observations)."}

        return {
            "detected":       True,
            "datetime_col":   dt_col,
            "target_col":     tgt_col,
            "n_observations": int(len(ts)),
            "freq":           self.infer_freq(ts),
            "eda":            self.run_eda(ts),
            "baselines":      self.run_baselines(ts),
        }


    def run_eda(self, ts: pd.Series) -> dict:
        return {
            "summary":            self.summary_stats(ts),
            "stationarity":       self.stationarity_tests(ts),
            "decomposition":      self.decompose(ts),
            "acf_pacf":           self.acf_pacf(ts),
            "missing_timestamps": self.missing_timestamps(ts),
            "rolling_stats":      self.rolling_stats(ts),
        }

    def summary_stats(self, ts: pd.Series) -> dict:
        clean = ts.dropna()
        return {
            "mean":        self.safe_float(clean.mean()),
            "std":         self.safe_float(clean.std()),
            "min":         self.safe_float(clean.min()),
            "max":         self.safe_float(clean.max()),
            "median":      self.safe_float(clean.median()),
            "missing_pct": self.safe_float(ts.isna().mean() * 100),
        }

    def stationarity_tests(self, ts: pd.Series) -> dict:
        clean = ts.dropna()
        result = {"adf": None, "kpss": None}

        try:
            from statsmodels.tsa.stattools import adfuller
            stat, p, _, _, crit, _ = adfuller(clean, autolag="AIC")
            result["adf"] = {
                "statistic":       self.safe_float(stat),
                "p_value":         self.safe_float(p),
                "is_stationary":   bool(p < 0.05),
                "critical_values": {k: self.safe_float(v) for k, v in crit.items()},
                "interpretation":  "Stationary (p < 0.05)" if p < 0.05 else "Non-stationary (p ≥ 0.05)",
            }
        except ImportError:
            result["adf"] = {"error": "statsmodels not installed. pip install 'athena-eda[timeseries]'"}
        except Exception as e:
            result["adf"] = {"error": str(e)}

        try:
            from statsmodels.tsa.stattools import kpss
            stat, p, _, crit = kpss(clean, regression="c", nlags="auto")
            result["kpss"] = {
                "statistic":       self.safe_float(stat),
                "p_value":         self.safe_float(p),
                "is_stationary":   bool(p > 0.05),
                "critical_values": {k: self.safe_float(v) for k, v in crit.items()},
                "interpretation":  "Stationary (p > 0.05)" if p > 0.05 else "Non-stationary (p ≤ 0.05)",
            }
        except ImportError:
            result["kpss"] = {"error": "statsmodels not installed."}
        except Exception as e:
            result["kpss"] = {"error": str(e)}

        adf_ok  = result["adf"].get("is_stationary")  if result["adf"]  else None
        kpss_ok = result["kpss"].get("is_stationary") if result["kpss"] else None

        if adf_ok is True and kpss_ok is True:
            verdict = "stationary"
        elif adf_ok is False and kpss_ok is False:
            verdict = "non_stationary"
        elif adf_ok is None or kpss_ok is None:
            verdict = "unknown"
        else:
            verdict = "inconclusive"

        result["verdict"]            = verdict
        result["needs_differencing"] = verdict in ("non_stationary", "inconclusive")
        return result

    def decompose(self, ts: pd.Series) -> dict:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            clean = ts.dropna()
            if len(clean) < 4:
                return {"error": "Too few observations for decomposition."}

            freq = self.infer_freq(ts)
            period_map = {
                "D": 7, "W": 52, "M": 12, "MS": 12,
                "Q": 4, "QS": 4, "A": 1, "Y": 1, "H": 24, "T": 60,
            }
            period = next((v for k, v in period_map.items() if freq.startswith(k)), 7)
            period = min(period, len(clean) // 2)

            decomp = seasonal_decompose(
                clean, model="additive", period=period, extrapolate_trend="freq"
            )
            return {
                "period":   int(period),
                "trend":    self.safe_list(decomp.trend.values),
                "seasonal": self.safe_list(decomp.seasonal.values),
                "residual": self.safe_list(decomp.resid.values),
                "dates":    [str(d) for d in decomp.trend.index],
            }
        except ImportError:
            return {"error": "statsmodels not installed."}
        except Exception as e:
            return {"error": str(e)}

    def acf_pacf(self, ts: pd.Series, nlags: int = 40) -> dict:
        try:
            from statsmodels.tsa.stattools import acf, pacf
            clean = ts.dropna()
            nlags = min(nlags, len(clean) // 2 - 1)
            acf_vals  = acf(clean, nlags=nlags, fft=True)
            pacf_vals = pacf(clean, nlags=nlags)
            conf = 1.96 / np.sqrt(len(clean))
            return {
                "acf":        self.safe_list(acf_vals),
                "pacf":       self.safe_list(pacf_vals),
                "confidence": self.safe_float(conf),
                "lags":       list(range(nlags + 1)),
            }
        except ImportError:
            return {"error": "statsmodels not installed."}
        except Exception as e:
            return {"error": str(e)}

    def missing_timestamps(self, ts: pd.Series) -> dict:
        try:
            freq = pd.infer_freq(ts.index)
            if not freq:
                return {"gap_count": 0, "gaps": [], "coverage_pct": 100.0}
            full_idx = pd.date_range(ts.index.min(), ts.index.max(), freq=freq)
            missing  = full_idx.difference(ts.index)
            return {
                "gap_count":      int(len(missing)),
                "total_expected": int(len(full_idx)),
                "coverage_pct":   self.safe_float((1 - len(missing) / len(full_idx)) * 100),
                "gaps":           [str(d) for d in missing[:20]],
            }
        except Exception as e:
            return {"error": str(e)}

    def rolling_stats(self, ts: pd.Series, window: int = 12) -> dict:
        try:
            clean = ts.dropna()
            w = min(window, len(clean) // 3)
            return {
                "window": w,
                "mean":   self.safe_list(clean.rolling(w).mean().values),
                "std":    self.safe_list(clean.rolling(w).std().values),
                "dates":  [str(d) for d in clean.index],
                "values": self.safe_list(clean.values),
            }
        except Exception as e:
            return {"error": str(e)}



    def run_baselines(self, ts: pd.Series) -> dict:
        clean = ts.dropna()
        split = int(len(clean) * 0.8)
        train = clean.iloc[:split]
        test  = clean.iloc[split:]

        if len(test) == 0:
            return {"error": "Not enough data for a train/test split."}

        results = {
            "naive":          self.naive_baseline(train, test),
            "moving_average": self.moving_average_baseline(train, test),
            "exp_smoothing":  self.exp_smoothing_baseline(train, test),
            "arima":          self.arima_baseline(train, test),
        }

        ranked = sorted(
            [(k, v) for k, v in results.items() if isinstance(v, dict) and v.get("rmse") is not None],
            key=lambda x: x[1]["rmse"],
        )
        results["ranking"] = [
            {"model": k, "rmse": v["rmse"], "mae": v.get("mae"), "mape": v.get("mape")}
            for k, v in ranked
        ]
        return results

    def metrics(self, actual, predicted) -> dict:
        actual    = np.array(actual, dtype=float)
        predicted = np.array(predicted, dtype=float)
        mask      = ~np.isnan(actual) & ~np.isnan(predicted)
        actual, predicted = actual[mask], predicted[mask]
        if len(actual) == 0:
            return {"rmse": None, "mae": None, "mape": None}
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mae  = float(np.mean(np.abs(actual - predicted)))
        mape = float(np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100)
        return {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 4)}

    def naive_baseline(self, train: pd.Series, test: pd.Series) -> dict:
        try:
            pred = np.full(len(test), train.iloc[-1])
            return {**self.metrics(test.values, pred), "model": "Naive (last value)", "forecast": self.safe_list(pred)}
        except Exception as e:
            return {"error": str(e)}

    def moving_average_baseline(self, train: pd.Series, test: pd.Series, window: int = 7) -> dict:
        try:
            w    = min(window, len(train))
            pred = np.full(len(test), train.iloc[-w:].mean())
            return {**self.metrics(test.values, pred), "model": f"Moving average (w={w})", "window": w, "forecast": self.safe_list(pred)}
        except Exception as e:
            return {"error": str(e)}

    def exp_smoothing_baseline(self, train: pd.Series, test: pd.Series) -> dict:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            fit  = ExponentialSmoothing(train, trend="add", seasonal=None, initialization_method="estimated").fit(optimized=True)
            pred = fit.forecast(len(test))
            return {**self.metrics(test.values, pred.values), "model": "Exponential Smoothing (Holt)", "forecast": self.safe_list(pred.values)}
        except ImportError:
            return {"error": "statsmodels not installed. pip install 'athena-eda[timeseries]'"}
        except Exception as e:
            return {"error": str(e)}

    def arima_baseline(self, train: pd.Series, test: pd.Series) -> dict:
        try:
            import pmdarima as pm
            model = pm.auto_arima(
                train, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action="ignore",
                max_p=3, max_q=3, max_d=2,
            )
            pred  = model.predict(n_periods=len(test))
            order = model.order
            return {**self.metrics(test.values, pred), "model": f"ARIMA{order}", "order": [int(x) for x in order], "forecast": self.safe_list(pred)}
        except ImportError:
            return {"error": "pmdarima not installed. pip install 'athena-eda[timeseries]'"}
        except Exception as e:
            return {"error": str(e)}