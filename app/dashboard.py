"""Churn Indicator Dashboard: KPI cards, filters and Plotly charts (dark theme)."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BG = "#0e1117"
LIGHT_BLUE, DARK_BLUE, PINK = "#82c9ff", "#1166c9", "#ffa8b8"
TITLE_CYAN = "#8be9fd"


def _style(fig, height=340):
    fig.update_layout(
        height=height, paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#fafafa", size=11),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(gridcolor="#262730", zeroline=False)
    fig.update_yaxes(gridcolor="#262730", zeroline=False)
    return fig


def _section(title):
    st.markdown(f"<p style='font-weight:700;font-size:0.95rem;margin:1rem 0 0'>{title}</p>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------- filters
def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Filter Options")
    out = df
    for col, label in [("tenure", "Tenure"), ("TotalCharges", "Total Charges"),
                       ("MonthlyCharges", "Monthly Charges")]:
        lo, hi = float(df[col].min()), float(df[col].max())
        if lo == hi:
            continue
        is_int = col == "tenure"
        lo_v, hi_v = st.sidebar.slider(
            label, int(lo) if is_int else lo, int(hi) if is_int else hi,
            (int(lo), int(hi)) if is_int else (lo, hi), key=f"flt_{col}")
        out = out[(out[col] >= lo_v) & (out[col] <= hi_v)]
    return out


# ---------------------------------------------------------------- KPIs
def _delta(current, baseline):
    return f"{(current - baseline) / baseline * 100:.2f}%" if baseline else "0.00%"


def kpi_row(df, base):
    def fmt(v):
        return f"${v / 1e6:.2f}M" if v >= 1e6 else f"${v / 1e3:.2f}K"
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Monthly Charge", fmt(df["MonthlyCharges"].sum()),
              _delta(df["MonthlyCharges"].sum(), base["MonthlyCharges"].sum()))
    c2.metric("Total Charge", fmt(df["TotalCharges"].sum()),
              _delta(df["TotalCharges"].sum(), base["TotalCharges"].sum()))
    kept, kept_base = int((df["Churn"] == "No").sum()), int((base["Churn"] == "No").sum())
    c3.metric("Total Customers Retained", f"{kept:,}".replace(",", ""), _delta(kept, kept_base))


# ---------------------------------------------------------------- charts
def churn_by_tenure(df):
    _section("Churn Rate by Tenure")
    d = df.assign(Yes=(df["Churn"] == "Yes").astype(int)).groupby("tenure", as_index=False)["Yes"].mean()
    fig = px.line(d, x="tenure", y="Yes")
    fig.update_traces(line=dict(color=LIGHT_BLUE, width=1.5))
    fig.update_yaxes(title="Yes")
    st.plotly_chart(_style(fig, 320), use_container_width=True)


def churn_gauge(df):
    rate = (df["Churn"] == "Yes").mean() * 100 if len(df) else 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(rate, 1),
        title={"text": "Churn Rate (%)", "font": {"size": 16}},
        number={"font": {"size": 44}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [0, 20, 40, 60, 80, 100]},
            "bar": {"color": "rgba(0,0,0,0)"},
            "steps": [{"range": [0, 20], "color": "#1010ff"},
                      {"range": [20, 60], "color": "#8a8a8a"},
                      {"range": [60, 100], "color": "#ee1111"}],
            "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.9, "value": rate},
        }))
    st.plotly_chart(_style(fig, 340), use_container_width=True)


def churn_by_internet(df):
    _section("Churn Rate by InternetService")
    d = df[df["Churn"] == "Yes"]["InternetService"].value_counts().reset_index()
    d.columns = ["InternetService", "count"]
    fig = px.pie(d, names="InternetService", values="count", color="InternetService",
                 color_discrete_map={"Fiber optic": LIGHT_BLUE, "DSL": DARK_BLUE, "No": PINK})
    fig.update_traces(textinfo="percent", textfont_size=11)
    st.plotly_chart(_style(fig, 340), use_container_width=True)


def churn_by_contract(df):
    _section("Churn Rate by Contract")
    d = (df.assign(rate=(df["Churn"] == "Yes") * 100)
           .groupby("Contract", as_index=False)["rate"].mean())
    fig = px.bar(d, x="Contract", y="rate", labels={"rate": "Churn Rate (%)"})
    fig.update_traces(marker_color=LIGHT_BLUE, texttemplate="%{y:.1f}%", textposition="outside")
    st.plotly_chart(_style(fig, 340), use_container_width=True)


# ---------------------------------------------------------------- pages
def analytical_dashboard(df, base):
    st.markdown(f"<h1 style='text-align:center;color:{TITLE_CYAN};font-weight:800'>"
                "💡 Churn Indicator Dashboard</h1>", unsafe_allow_html=True)
    if df.empty:
        st.warning("No customers match the current filters.")
        return
    kpi_row(df, base)
    left, right = st.columns([3, 2])
    with left:
        churn_by_tenure(df)
    with right:
        churn_gauge(df)
    left, right = st.columns([2, 3])
    with left:
        churn_by_internet(df)
    with right:
        churn_by_contract(df)


def eda_dashboard(df):
    st.markdown(f"<h1 style='text-align:center;color:{TITLE_CYAN};font-weight:800'>"
                "📊 EDA Dashboard</h1>", unsafe_allow_html=True)
    if df.empty:
        st.warning("No customers match the current filters.")
        return
    cat_cols = ["Contract", "PaymentMethod", "InternetService", "gender", "SeniorCitizen",
                "TechSupport", "PaperlessBilling", "Partner"]
    c1, c2 = st.columns(2)
    with c1:
        _section("Distribution of Monthly Charges by Churn")
        fig = px.histogram(df, x="MonthlyCharges", color="Churn", barmode="overlay", nbins=40,
                           color_discrete_map={"No": LIGHT_BLUE, "Yes": "#ee5555"})
        st.plotly_chart(_style(fig), use_container_width=True)
    with c2:
        _section("Tenure by Churn")
        fig = px.box(df, x="Churn", y="tenure", color="Churn",
                     color_discrete_map={"No": LIGHT_BLUE, "Yes": "#ee5555"})
        st.plotly_chart(_style(fig), use_container_width=True)
    _section("Churn by Category")
    col = st.selectbox("Feature", cat_cols, label_visibility="collapsed")
    d = df.groupby([col, "Churn"]).size().reset_index(name="count")
    fig = px.bar(d, x=col, y="count", color="Churn", barmode="group",
                 color_discrete_map={"No": LIGHT_BLUE, "Yes": "#ee5555"})
    st.plotly_chart(_style(fig), use_container_width=True)
    _section("Correlation Matrix")
    num = df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]].copy()
    num["Churn"] = (df["Churn"] == "Yes").astype(int)
    fig = px.imshow(num.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(_style(fig, 380), use_container_width=True)
