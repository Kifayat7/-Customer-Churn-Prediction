"""Load the trained model and score customers."""
from functools import lru_cache

import joblib
import pandas as pd

from src.data_preprocessing import FEATURES, ROOT, add_features, clean_data

MODEL_PATH = ROOT / "models" / "churn_model.pkl"
RISK_BANDS = [(0.30, "Low"), (0.60, "Medium"), (1.01, "High")]


@lru_cache(maxsize=1)
def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.train` first.")
    return joblib.load(path)


def risk_band(prob: float) -> str:
    return next(label for limit, label in RISK_BANDS if prob < limit)


def predict_proba_df(df: pd.DataFrame) -> pd.Series:
    """Churn probability for every row of a raw-format dataframe."""
    prepared = add_features(clean_data(df.drop(columns=["Churn"], errors="ignore")))
    return pd.Series(load_model().predict_proba(prepared[FEATURES])[:, 1], index=prepared.index)


def predict_one(customer: dict) -> dict:
    """Score a single customer described by a dict of raw column values."""
    row = dict(customer)
    if row.get("TotalCharges") in (None, ""):
        row["TotalCharges"] = float(row["MonthlyCharges"]) * float(row["tenure"])
    prob = float(predict_proba_df(pd.DataFrame([row])).iloc[0])
    return {
        "churn_probability": prob,
        "prediction": "Yes" if prob >= 0.5 else "No",
        "risk": risk_band(prob),
    }
