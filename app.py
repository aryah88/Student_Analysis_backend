# backend/app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io
import os
from typing import List

app = FastAPI(title="Student Predictions API")

# Allow local frontend (adjust origin for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to your saved model file (joblib / pickle etc.)
MODEL_PATH = os.environ.get("MODEL_PATH", "backend/models/model.joblib")
SAMPLE_CSV = os.environ.get("SAMPLE_CSV", "backend/models/sample_train.csv")

# Load model once at startup
model = None
feature_cols = None  # set if needed

@app.on_event("startup")
def load_model():
    global model, feature_cols
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            # Optional: if you saved feature column names with the model, set them:
            # feature_cols = getattr(model, "feature_columns", None)
            print("Model loaded:", MODEL_PATH)
        else:
            print("Model not found at", MODEL_PATH, "- running in 'no-model' mode.")
            model = None
    except Exception as e:
        print("Failed to load model:", e)
        model = None


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, run the model and append columns:
    - predicted_score (float)
    - persona (int)  (optional)
    If model only outputs one value per row, we'll map that to predicted_score.
    """
    df_out = df.copy()

    if model is None:
        # No model: return dummy predictions or raise
        # Here we'll add a simple fallback: predicted_score = assessment_score (if exists) or mean
        if "assessment_score" in df_out.columns:
            df_out["predicted_score"] = df_out["assessment_score"]
        else:
            df_out["predicted_score"] = 0.0
        if "persona" not in df_out.columns:
            df_out["persona"] = 0
        return df_out

    # Determine features
    X = None
    if feature_cols:
        missing = [c for c in feature_cols if c not in df_out.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required columns for model: {missing}")
        X = df_out[feature_cols]
    else:
        # Fallback: use all numeric columns except id/name
        numeric_cols = df_out.select_dtypes(include=["number"]).columns.tolist()
        # try to remove student_id if present
        numeric_cols = [c for c in numeric_cols if c not in ("student_id",)]
        if not numeric_cols:
            raise HTTPException(status_code=400, detail="No numeric features found for prediction")
        X = df_out[numeric_cols]

    # Some models return 1D predictions, some dict/tuple or array. Try common patterns.
    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    # If preds shape matches rows -> numeric predicted score
    if hasattr(preds, "__len__") and len(preds) == len(df_out):
        df_out["predicted_score"] = pd.Series(preds).astype(float).round(2)
    else:
        # fallback
        df_out["predicted_score"] = 0.0

    # If the model also supports personas (class prediction or cluster), attempt predict_proba or predict_classes
    # We'll try model.predict_proba -> persona = argmax(probabilities) if number of classes small
    persona = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if getattr(probs, "shape", (0, 0))[1] <= 20:
                persona = probs.argmax(axis=1).astype(int)
                df_out["persona"] = persona
    except Exception:
        # ignore persona if not available
        pass

    if "persona" not in df_out.columns:
        df_out["persona"] = 0

    return df_out


def df_to_preview_json(df: pd.DataFrame, max_rows: int = 500):
    # Keep only columns frontend expects; ensure serializable
    df_small = df.head(max_rows).copy()
    # convert numpy types
    for c in df_small.columns:
        if pd.api.types.is_float_dtype(df_small[c]) or pd.api.types.is_integer_dtype(df_small[c]):
            df_small[c] = df_small[c].apply(lambda x: None if pd.isna(x) else (float(x) if pd.api.types.is_float_dtype(df_small[c]) else int(x)))
        else:
            df_small[c] = df_small[c].astype(str)
    records = df_small.to_dict(orient="records")
    return {"preview": records}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept multipart upload of CSV file (field name: file).
    Returns JSON: { preview: [ {row... predicted_score, persona}, ... ] }
    """
    # Accept CSV only
    if not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        data = await file.read()  # bytes
        s = io.BytesIO(data)
        df = pd.read_csv(s)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    try:
        df_pred = predict_df(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return JSONResponse(content=df_to_preview_json(df_pred, max_rows=1000))


@app.get("/predict-sample")
def predict_sample():
    """
    Read sample CSV from disk, run prediction and return preview.
    """
    if not os.path.exists(SAMPLE_CSV):
        raise HTTPException(status_code=404, detail="Sample CSV not found on server")
    try:
        df = pd.read_csv(SAMPLE_CSV)
        df_pred = predict_df(df)
        return JSONResponse(content=df_to_preview_json(df_pred, max_rows=1000))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sample or predict: {e}")
