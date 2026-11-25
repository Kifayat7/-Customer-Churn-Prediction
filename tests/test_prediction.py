import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import predict as pr
from src.data_preprocessing import FEATURES, add_features, clean_data, load_data, prepare_xy


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    if not pr.MODEL_PATH.exists():
        from src.train import train
        train()


LOYAL = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
    "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
    "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.0,
}
RISKY = {**LOYAL, "gender": "Male", "Partner": "No", "Dependents": "No", "tenure": 1,
         "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
         "DeviceProtection": "No", "TechSupport": "No", "Contract": "Month-to-month",
         "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0}


def test_clean_data_fixes_total_charges():
    df = clean_data(load_data())
    assert pd.api.types.is_float_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 0


def test_features_and_target():
    X, y = prepare_xy(load_data())
    assert list(X.columns) == FEATURES
    assert set(y.unique()) == {0, 1}
    assert {"NumServices", "AvgMonthlySpend", "TenureGroup"} <= set(add_features(clean_data(load_data())).columns)


def test_predict_one_output_format():
    res = pr.predict_one(LOYAL)
    assert set(res) == {"churn_probability", "prediction", "risk"}
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["prediction"] in {"Yes", "No"}
    assert res["risk"] in {"Low", "Medium", "High"}


def test_risky_customer_scores_higher_than_loyal():
    assert pr.predict_one(RISKY)["churn_probability"] > pr.predict_one(LOYAL)["churn_probability"]


def test_predict_proba_df_length():
    df = load_data().head(25)
    assert len(pr.predict_proba_df(df)) == 25
