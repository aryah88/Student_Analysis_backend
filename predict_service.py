# backend/predict_service.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
from io import BytesIO
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predict_service")

app = FastAPI(title="Student Prediction Service")

# allow dev origin (frontend runs on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to artifacts (relative to working directory where you run uvicorn)
MODEL_PATH = os.path.join("models", "pipeline.joblib")
KMEANS_PATH = os.path.join("models", "kmeans.joblib")
SAMPLE_CSV_PATH = os.path.join("models", "sample_train.csv")

# expected numeric features (lowercase)
EXPECTED = ["comprehension", "attention", "focus", "retention", "engagement_time"]

# Attempt to load artifacts, but continue even if missing (server will still run)
pipeline = None
kmeans = None

def try_load_models():
    global pipeline, kmeans
    try:
        if os.path.exists(MODEL_PATH):
            pipeline = joblib.load(MODEL_PATH)
            logger.info(f"Loaded pipeline from {MODEL_PATH}")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}. Predictions will fail until you provide it.")

        if os.path.exists(KMEANS_PATH):
            kmeans = joblib.load(KMEANS_PATH)
            logger.info(f"Loaded kmeans from {KMEANS_PATH}")
        else:
            logger.warning(f"KMeans file not found at {KMEANS_PATH}. Persona assignment will fallback.")
    except Exception as e:
        logger.exception("Error loading model artifacts: %s", e)

# load at startup
try_load_models()


@app.post("/predict")
async def predict_csv(file: UploadFile = File(...)):
    """
    Accepts a CSV file upload. CSV must contain columns:
    comprehension, attention, focus, retention, engagement_time
    (column names are matched case-insensitively).

    Returns: {"n": <rows>, "preview": [ first 200 rows with predicted_score & persona ] }
    """
    # ensure models are available
    if pipeline is None:
        return JSONResponse({"error": "Model pipeline not loaded on server. Put pipeline.joblib in models/ and restart."}, status_code=500)

    content = await file.read()
    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as e:
        return JSONResponse({"error": f"Could not parse CSV: {e}"}, status_code=400)

    # account for case-insensitive column names
    cols = {c.lower(): c for c in df.columns}
    missing = [c for c in EXPECTED if c not in cols]
    if missing:
        return JSONResponse({"error": f"Missing columns: {missing}. Provide columns: {EXPECTED}"}, status_code=400)

    # select expected columns and coerce to float
    try:
        X = df[[cols[c] for c in EXPECTED]].astype(float)
    except Exception as e:
        return JSONResponse({"error": f"Could not convert feature columns to numeric: {e}"}, status_code=400)

    # predict
    try:
        preds = pipeline.predict(X)
        # round predictions to 2 decimals
        preds = pd.Series(preds).round(2).tolist()
    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse({"error": f"Prediction failed: {e}"}, status_code=500)

    # persona via kmeans if available; otherwise try to use scaler from pipeline
    try:
        if kmeans is not None:
            # attempt to use a scaler inside pipeline if exists
            try:
                scaler = pipeline.named_steps.get("scaler") if hasattr(pipeline, "named_steps") else None
                if scaler is not None:
                    X_scaled = scaler.transform(X)
                else:
                    X_scaled = X.values
            except Exception:
                X_scaled = X.values

            personas = kmeans.predict(X_scaled).tolist()
        else:
            # fallback: persona = 0 for all
            personas = [0] * len(X)
    except Exception as e:
        logger.exception("Persona assignment failed")
        personas = [0] * len(X)

    # assemble output (preserve original df and append predictions)
    out = df.copy()
    out["predicted_score"] = preds
    out["persona"] = personas

    # prepare preview (first 200 rows)
    preview = out.head(200).where(pd.notnull(out.head(200)), None).to_dict(orient="records")

    return {"n": len(out), "preview": preview}


@app.get("/predict-sample")
def predict_sample():
    """
    Returns the sample CSV preview (if present in models/sample_train.csv).
    Frontend uses this to load demo data for charts.
    """
    if not os.path.exists(SAMPLE_CSV_PATH):
        return JSONResponse({"error": "sample_train.csv not found on server. Put sample_train.csv in models/ and restart."}, status_code=404)
    try:
        df = pd.read_csv(SAMPLE_CSV_PATH)
    except Exception as e:
        return JSONResponse({"error": f"Could not read sample CSV: {e}"}, status_code=500)

    preview = df.head(1000).where(pd.notnull(df.head(1000)), None).to_dict(orient="records")
    return {"n": len(df), "preview": preview}
