"""Data loading, cleaning, feature engineering and preprocessing utilities."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
TARGET = "Churn"
ID_COL = "customerID"

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "NumServices", "AvgMonthlySpend",
]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "TenureGroup",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Read the raw Telco churn CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes and missing values. Keeps Churn as 'Yes'/'No'."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # TotalCharges is stored as text and has blanks for brand-new customers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df = df.drop_duplicates()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features used by the model."""
    df = df.copy()
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72, 1000],
        labels=["0-12", "13-24", "25-48", "49-72", "72+"],
    ).astype(str)
    return df


def prepare_xy(df: pd.DataFrame):
    """Return the feature matrix X and the binary target y (1 = churned)."""
    df = add_features(clean_data(df))
    X = df[FEATURES]
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def split_data(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
