"""Streamlit entry point:  streamlit run app.py"""
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app import dashboard
from src.data_preprocessing import DATA_PATH, clean_data, load_data
from src.predict import predict_one

ROOT = Path(__file__).resolve().parent
HISTORY_PATH = ROOT / "data" / "prediction_history.csv"
METRICS_PATH = ROOT / "reports" / "model_metrics.txt"

st.set_page_config(page_title="Churn Indicator Dashboard", page_icon="💡", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] .stRadio > div { gap: .15rem; }
[data-testid="stMetricValue"] { font-size: 2rem; }
[data-testid="stMetricLabel"] p { font-size: .8rem; }
.block-container { padding-top: 2.5rem; }
</style>""", unsafe_allow_html=True)

PAGES = {"🏠 Home": "Home", "🗃️ Data": "Data", "📈 Predict": "Predict",
         "⏰ History": "History", "📊 Dashboard": "Dashboard"}
HISTORY_COLS = ["Timestamp", "ChurnProbability", "Risk"]


@st.cache_data
def get_data():
    return clean_data(load_data(DATA_PATH))


def load_history() -> pd.DataFrame:
    return pd.read_csv(HISTORY_PATH) if HISTORY_PATH.exists() else pd.DataFrame()


# ------------------------------------------------------------------ pages
def page_home(df):
    st.title("💡 Customer Churn Prediction")
    st.write("Explore the Telco customer base, score individual customers with the trained "
             "model, and monitor churn drivers on the dashboard.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{(df['Churn'] == 'Yes').mean() * 100:.1f}%")
    c3.metric("Avg. monthly charge", f"${df['MonthlyCharges'].mean():.2f}")
    if METRICS_PATH.exists():
        st.subheader("Model performance")
        st.code(METRICS_PATH.read_text())
    figs = sorted((ROOT / "reports" / "figures").glob("*.png"))
    if figs:
        cols = st.columns(2)
        for i, f in enumerate(figs):
            cols[i % 2].image(str(f), caption=f.stem.replace("_", " ").title())


def page_data(df):
    st.title("🗃️ Data")
    st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df, use_container_width=True, height=420)
    with st.expander("Summary statistics"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)
    st.download_button("Download cleaned CSV", df.to_csv(index=False), "telco_churn_clean.csv")


def page_predict(df):
    st.title("📈 Predict Churn")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        yn = ["No", "Yes"]
        gender = c1.selectbox("Gender", ["Female", "Male"])
        senior = c2.selectbox("Senior citizen", yn)
        partner = c3.selectbox("Partner", yn)
        dependents = c1.selectbox("Dependents", yn)
        tenure = c2.slider("Tenure (months)", 0, 72, 12)
        monthly = c3.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
        phone = c1.selectbox("Phone service", ["Yes", "No"])
        lines = c2.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = c3.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        no_int = "No internet service"
        sec = c1.selectbox("Online security", ["No", "Yes", no_int])
        backup = c2.selectbox("Online backup", ["No", "Yes", no_int])
        protect = c3.selectbox("Device protection", ["No", "Yes", no_int])
        tech = c1.selectbox("Tech support", ["No", "Yes", no_int])
        tv = c2.selectbox("Streaming TV", ["No", "Yes", no_int])
        movies = c3.selectbox("Streaming movies", ["No", "Yes", no_int])
        contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = c2.selectbox("Paperless billing", ["Yes", "No"])
        pay = c3.selectbox("Payment method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)",
            "Credit card (automatic)"])
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = {"gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
               "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
               "MultipleLines": lines, "InternetService": internet, "OnlineSecurity": sec,
               "OnlineBackup": backup, "DeviceProtection": protect, "TechSupport": tech,
               "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
               "PaperlessBilling": paperless, "PaymentMethod": pay,
               "MonthlyCharges": monthly, "TotalCharges": monthly * tenure}
        try:
            res = predict_one(row)
        except FileNotFoundError as e:
            st.error(str(e))
            return
        p = res["churn_probability"]
        icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}[res["risk"]]
        st.subheader(f"{icon} {res['risk']} churn risk — {p * 100:.1f}%")
        st.progress(min(p, 1.0))
        st.write("**Prediction:** this customer is likely to " +
                 ("**churn**." if res["prediction"] == "Yes" else "**stay**."))
        record = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **row,
                  "Churn": res["prediction"], "ChurnProbability": round(p, 4),
                  "Risk": res["risk"]}
        hist = pd.concat([load_history(), pd.DataFrame([record])], ignore_index=True)
        hist.to_csv(HISTORY_PATH, index=False)


def page_history():
    st.title("⏰ Prediction History")
    hist = load_history()
    if hist.empty:
        st.info("No predictions yet. Make one on the Predict page.")
        return
    st.dataframe(hist.iloc[::-1], use_container_width=True)
    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button("Download CSV", hist.to_csv(index=False), "prediction_history.csv")
    if c2.button("Clear history"):
        HISTORY_PATH.unlink(missing_ok=True)
        st.rerun()


def page_dashboard(df):
    st.sidebar.divider()
    st.sidebar.caption("Choose Data Source")
    source = st.sidebar.radio("Choose Data Source", ["Inbuilt Data", "Prediction History"],
                              label_visibility="collapsed")
    st.sidebar.caption("Display a Dashboard")
    kind = st.sidebar.radio("Display a Dashboard", ["EDA Dashboard", "Analytical Dashboard"],
                            index=1, label_visibility="collapsed")
    if source == "Prediction History":
        data = load_history()
        if data.empty:
            st.info("No prediction history yet. Make some predictions first.")
            return
        data = data.drop(columns=[c for c in HISTORY_COLS if c in data.columns])
    else:
        data = df
    st.sidebar.divider()
    filtered = dashboard.sidebar_filters(data)
    if kind == "EDA Dashboard":
        dashboard.eda_dashboard(filtered)
    else:
        dashboard.analytical_dashboard(filtered, data)


# ------------------------------------------------------------------ main
def main():
    df = get_data()
    choice = st.sidebar.radio("Navigation", list(PAGES), index=4, label_visibility="collapsed")
    page = PAGES[choice]
    if page == "Home":
        page_home(df)
    elif page == "Data":
        page_data(df)
    elif page == "Predict":
        page_predict(df)
    elif page == "History":
        page_history()
    else:
        page_dashboard(df)


main()
