# Customer Churn Prediction

End-to-end churn project on the Telco Customer Churn dataset (7,043 customers): EDA, preprocessing,
feature engineering, model training/evaluation, and a dark-themed Streamlit **Churn Indicator Dashboard**.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train          # trains models, saves models/churn_model.pkl + reports/
streamlit run app.py         # opens the dashboard
pytest                       # run tests
```

## Structure
| Path | Purpose |
|---|---|
| `data/telco_churn.csv` | Raw dataset (`prediction_history.csv` is created by the app) |
| `notebooks/` | 01 EDA → 05 Evaluation |
| `src/data_preprocessing.py` | Cleaning, engineered features, sklearn preprocessor |
| `src/train.py` | Trains Logistic Regression / Random Forest / Gradient Boosting, keeps best by ROC-AUC |
| `src/predict.py` | `predict_one(dict)` and `predict_proba_df(df)` |
| `src/evaluation.py` | Metrics and figure generation |
| `app.py`, `app/dashboard.py` | Streamlit app and dashboard charts |
| `models/`, `reports/` | Saved model, metrics, figures |
| `.streamlit/config.toml` | Dark theme |

## App pages
Home · Data · Predict (scores a customer, saves to history) · History · **Dashboard**
(Inbuilt Data or Prediction History; EDA or Analytical view; sliders for Tenure, Total/Monthly Charges).

## Results (hold-out 20%)
| Model | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| Logistic Regression (selected) | 0.846 | 0.612 | 0.737 |
| Random Forest | 0.836 | 0.627 | 0.765 |
| Gradient Boosting | 0.843 | 0.578 | 0.801 |

Churn-risk bands: Low < 30%, Medium 30–60%, High ≥ 60%.
