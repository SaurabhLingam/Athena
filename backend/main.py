from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi.responses import Response
from dotenv import load_dotenv
import os
from pathlib import Path
import math
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

try:
    from backend.eda import EdaToolKit
    from backend.profiles import get_profile
    from backend.ml import MLdiagnosticskit
    from backend.timeseries import TimeSeriesKit
    from backend.model_arena import ModelArenaKit
except ImportError:
    from eda import EdaToolKit
    from profiles import get_profile
    from ml import MLdiagnosticskit
    from timeseries import TimeSeriesKit
    from model_arena import ModelArenaKit




limiter = Limiter(key_func=get_remote_address)




MAX_FILE_SIZE = 1024 * 1024 * 1024  
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://athena-alpha-hazel.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)




def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

def safe_run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return {"status": "error", "error_msg": str(e)}

def read_csv_sample(upload_file, sample_rows=200_000):
    upload_file.file.seek(0)
    try:
        try:
            df = pd.read_csv(upload_file.file, encoding="utf-8", nrows=sample_rows)
        except UnicodeDecodeError:
            upload_file.file.seek(0)
            df = pd.read_csv(upload_file.file, encoding="latin1", nrows=sample_rows)
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV sample: {str(e)}")

@app.post("/analyze")
@limiter.limit("5/minute")
def analyze_csv(request: Request, file: UploadFile = File(...), data_type: str = "unlabeled", target_col: str = None, profile: str = "standard", datetime_col: str = None, ts_target_col: str = None):
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    cfg = get_profile(profile)
    eda = EdaToolKit(cfg=cfg)
    ml_kit = MLdiagnosticskit(cfg=cfg)
    ts_kit = TimeSeriesKit(cfg=cfg)
    arena_kit = ModelArenaKit(cfg=cfg)

    if data_type not in {"unlabeled", "labeled"}:
        raise HTTPException(400, detail="data_type must be unlabeled, labeled")

    df = read_csv_sample(file)

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    if target_col is not None and target_col not in df.columns:
        raise HTTPException(status_code=400, detail="target_col not found in dataframe")

    column_views = eda.build_column_views(df, target_col)
    free_text_result = safe_run(eda.free_text_detection, df, target_col)
    text_cols = [c["column"] for c in free_text_result.get("free_text_columns", [])] if free_text_result else []
    nlp_result = safe_run(eda.nlp_analysis, df, text_cols, target_col) if text_cols else None
    # Logic grouped inside safe_run to handle edge-case crashes
    base_eda = {
        "basic": safe_run(eda.basicdatainfo, df),
        "col_views": column_views,
        "missingness": safe_run(eda.missingdata, df, target_col, eda.cfg["missing_threshold"] * 100),
        "numeric_summary": safe_run(eda.numeric_summary, df, target_col),
        "categorical_summary": safe_run(eda.categorical_summary, df, target_col),
        "outliers": safe_run(eda.outlier_detection, df, target_col),
        "similar_pairs": safe_run(eda.feature_similarity, df, target_col),
        "feature_redundancy": safe_run(eda.feature_redundancy, df, target_col),
        "data_structural_report": safe_run(eda.data_health_report, df, target_col),
        "free_text": free_text_result,
        "nlp": nlp_result,
    }
    ts_result = safe_run(
        ts_kit.run_timeseries,
        df,
        datetime_col=datetime_col,
        target_col=ts_target_col or target_col,
    )

    if data_type == "unlabeled":
        routing_report = ml_kit.run_diagnostics(df=df, col_views=column_views, mode="unlabeled")
        ml_report = routing_report
        
        if routing_report.get("status") == "ready_for_unlabeled_pipeline":
            ml_report = safe_run(ml_kit.unlabeled_diagnostics, df=df, col_views=column_views)
        
        result = {"mode": "unlabeled", "eda": base_eda, "ml_diagnostics": ml_report, "timeseries": ts_result, "model_arena": {"status": "skipped", "reason": "Only runs on labeled datasets."}}

    elif data_type == "labeled":
        if not target_col:
            raise HTTPException(status_code=400, detail="target col required for labeled data")

        feature_correlation = safe_run(eda.feature_correlation, df, target_col)
        target_insights = safe_run(eda.target_insights, df, target_col)

        routing_report = ml_kit.run_diagnostics(df=df, col_views=column_views, mode="labeled", target_col=target_col)

        ml_report = routing_report
        if routing_report.get("status") == "ready_for_labeled_pipeline":
            ml_report = ml_kit.labeled_diagnostics(
                df=df,
                col_views=column_views,
                target_insights=target_insights,
                feature_correlation=feature_correlation,
                target_col=target_col
            )
            arena_result = {"status": "skipped", "reason": "Only runs on labeled datasets."}
            if data_type == "labeled" and target_col:
                task_type = ml_report.get("task_type") if isinstance(ml_report, dict) else None
                if task_type in ("classification", "regression"):
                    arena_result = safe_run(
                        arena_kit.run_arena,
                        df=df,
                        col_views=column_views,
                        target_col=target_col,
                        task_type=task_type,
                    )


        result = {
            "mode": "labeled",
            "eda": {
                **base_eda,
                "feature_correlation": feature_correlation,
                "target_insights": target_insights,
                "bivariate": safe_run(eda.bivariate_analysis, df, target_col),
            },
            "ml_diagnostics": ml_report,
            "timeseries": ts_result,        
            "model_arena": arena_result
        }
    else:
        raise HTTPException(400, detail="invalid data type")

    return sanitize_for_json(result)

@app.post("/compare")
@limiter.limit("5/minute")
def compare_datasets(request: Request, train_file: UploadFile = File(...), test_file: UploadFile = File(...)):
    if train_file.size > MAX_FILE_SIZE or test_file.size > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
    if not train_file.filename.endswith(".csv") or not test_file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Both files must be CSV")

    train_df = read_csv_sample(train_file)
    test_df = read_csv_sample(test_file)

    if train_df.empty or test_df.empty:
        raise HTTPException(status_code=400, detail="One or both files are empty")

    try:
        from backend.drift import DriftAnalyzer
    except ImportError:
        from drift import DriftAnalyzer
    analyzer = DriftAnalyzer()
    result = analyzer.compare(train_df, test_df)

    return sanitize_for_json(result)

@app.post("/script")
def generate_script(payload: dict):
    eda = payload.get("eda", {})
    timeseries = payload.get("timeseries", {})
    free_text = eda.get("free_text", {})
    has_text_cols = len(free_text.get("free_text_columns", [])) > 0
    has_timeseries = timeseries.get("detected", False)
 
    try:
        from backend.codegen import generate_nlp_script, generate_preprocessing_script, generate_timeseries_script
    except ImportError:
        from codegen import generate_nlp_script, generate_preprocessing_script, generate_timeseries_script
 
    if has_timeseries:
        script = generate_timeseries_script(eda, timeseries)
        filename = "timeseries_pipeline.py"
    elif has_text_cols:
        script = generate_nlp_script(eda)
        filename = "nlp_preprocessing.py"
    else:
        script = generate_preprocessing_script(eda)
        filename = "preprocessing.py"
 
    return Response(
        content=script,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.post("/timeseries/detect")
def detect_timeseries(file: UploadFile = File(...)):
    df     = read_csv_sample(file)
    ts_kit = TimeSeriesKit()
    result = ts_kit.get_candidates(df)
    return sanitize_for_json(result)