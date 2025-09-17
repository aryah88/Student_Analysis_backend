# backend/train_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

DATA_PATH = Path("backend/models/sample_train.csv")
MODEL_OUT = Path("backend/models/model.joblib")

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # Ensure those columns present
    features = ["comprehension","attention","focus","retention","engagement_time"]
    X = df[features].astype(float)
    y = df["assessment_score"].astype(float)
    return X, y, df

def train_and_save(n_estimators=200, random_state=42):
    X, y, df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    # eval
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)

    print("Model saved to", MODEL_OUT)
    print(f"MSE: {mse:.4f}  RMSE: {mse**0.5:.4f}  R2: {r2:.4f}")
    return model

if __name__ == "__main__":
    train_and_save()
