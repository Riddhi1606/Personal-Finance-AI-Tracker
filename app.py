"""
Personal Finance AI Tracker
============================
A Streamlit dashboard that ingests bank/UPI transactions, cleans messy
Indian UPI narrations, auto-categorizes spending with ML, flags
anomalous transactions, and forecasts next month's spend.

Run locally:
    streamlit run app.py
"""

import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import preprocess_transactions
from categorization import build_pipeline, train, load_model, predict_batch, MODEL_PATH
from anomaly_detection import detect_anomalies, top_anomalies
from forecasting import forecast_next_month_total, forecast_by_category
import theme

st.set_page_config(
    page_title="Personal Finance AI Tracker",
    page_icon="\U0001F4B0",
    layout="wide",
)
theme.inject_css()

DATA_PATH = "data/transactions.csv"


# ---------------------------------------------------------------------------
# Data loading / caching
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    else:
        if not os.path.exists(DATA_PATH):
            # Sample CSV missing from the deployed repo (e.g. wasn't
            # uploaded, or .gitignore/upload quirk) - regenerate it on
            # the fly instead of crashing.
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
            from generate_sample_data import generate
            generate(out_path=DATA_PATH)
        raw = pd.read_csv(DATA_PATH)
    df = preprocess_transactions(raw)
    return df


@st.cache_resource
def get_model(df: pd.DataFrame):
    """Trains the categorizer on first run, or loads a cached model."""
    if os.path.exists(MODEL_PATH) and "true_category" not in df.columns:
        return load_model(MODEL_PATH)
    if "true_category" in df.columns:
        pipeline, _ = train(df.assign(true_category=df["true_category"]), MODEL_PATH)
        return pipeline
    return load_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    "### \U0001F4B0 Finance AI Tracker\nUpload your own bank/UPI CSV export, or explore with sample data."
)

uploaded = st.sidebar.file_uploader(
    "Upload transactions CSV",
    type=["csv"],
    help="Expected columns: date, narration, amount, type (Credit/Debit)",
)

df = load_data(uploaded)

# Category source: use true_category if present (demo data), else predict
model = get_model(df)
if "category" not in df.columns:
    df["category"] = df["clean_merchant"].pipe(lambda s: predict_batch(model, s))
    df.loc[df["is_credit"], "category"] = df.loc[df["is_credit"], "true_category"] \
        if "true_category" in df.columns else "Income"

date_min, date_max = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max
)
if len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df["date"] >= start) & (df["date"] <= end)]

st.sidebar.markdown("---")
st.sidebar.caption("Built by Riddhi Sharma \u2022 AI/ML, Arya College")

# ---------------------------------------------------------------------------
# Hero + KPIs
# ---------------------------------------------------------------------------
theme.hero(
    "Personal Finance AI Tracker",
    "ML-powered transaction categorization, anomaly detection & spend forecasting for Indian UPI users",
    eyebrow="AI / ML DASHBOARD",
)

spend_df = df[~df["is_credit"]]
income_df = df[df["is_credit"]]

total_spend = spend_df["abs_amount"].sum()
total_income = income_df["amount"].sum()
net_savings = total_income - total_spend
avg_daily_spend = spend_df.groupby(df["date"].dt.date)["abs_amount"].sum().mean()
savings_pct = (net_savings / total_income * 100) if total_income else 0

st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    theme.kpi_card("Total Spend", f"\u20B9{total_spend:,.0f}", "across selected period",
                    accent=theme.COLORS["terracotta"])
with k2:
    theme.kpi_card("Total Income", f"\u20B9{total_income:,.0f}", "credits received",
                    accent=theme.COLORS["teal"])
with k3:
    theme.kpi_card("Net Savings", f"\u20B9{net_savings:,.0f}", f"{savings_pct:.1f}% of income",
                    accent=theme.COLORS["marigold"])
with k4:
    theme.kpi_card("Avg Daily Spend", f"\u20B9{avg_daily_spend:,.0f}", "per active day",
                    accent=theme.COLORS["ink"])
st.markdown('</div>', unsafe_allow_html=True)

theme.hairline()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["\U0001F4CA Overview", "\U0001F3F7\ufe0f Categorization", "\u26A0\ufe0f Anomalies", "\U0001F52E Forecast"]
)

# --- Tab 1: Overview ---
with tab1:
    theme.section_title("Spending overview", "How your money moved across the selected period.")
    col1, col2 = st.columns([2, 1])

    with col1:
        daily = spend_df.groupby("date")["abs_amount"].sum().reset_index()
        fig = px.line(daily, x="date", y="abs_amount",
                       title="Daily Spend Over Time", labels={"abs_amount": "Amount (\u20B9)"})
        fig.update_traces(line_color=theme.COLORS["marigold"], line_width=2.5)
        st.plotly_chart(theme.style_fig(fig), use_container_width=True)

    with col2:
        cat_totals = spend_df.groupby("category")["abs_amount"].sum().sort_values(ascending=False)
        fig2 = px.pie(cat_totals, values=cat_totals.values, names=cat_totals.index,
                       title="Spend by Category", hole=0.55,
                       color_discrete_sequence=theme.CATEGORY_COLORS)
        st.plotly_chart(theme.style_fig(fig2), use_container_width=True)

    monthly = spend_df.groupby("month")["abs_amount"].sum().reset_index()
    fig3 = px.bar(monthly, x="month", y="abs_amount", title="Monthly Spend Totals",
                   labels={"abs_amount": "Amount (\u20B9)"})
    fig3.update_traces(marker_color=theme.COLORS["ink"])
    st.plotly_chart(theme.style_fig(fig3, height=340), use_container_width=True)

# --- Tab 2: Categorization ---
with tab2:
    theme.section_title(
        "ML transaction categorization",
        "Each transaction's UPI narration is cleaned (VPA handles, reference numbers, and "
        "bank codes stripped) and classified into a spend category using a TF-IDF + Naive Bayes model.",
    )

    if "true_category" in df.columns:
        acc = (df["category"] == df["true_category"]).mean()
        st.info(f"Model accuracy on this dataset: **{acc*100:.1f}%**")

    example_cols = ["date", "narration", "clean_merchant", "category", "abs_amount"]
    st.dataframe(
        spend_df[example_cols].sort_values("date", ascending=False).head(50),
        use_container_width=True,
    )

    theme.section_title("Try it yourself", "")
    sample_input = st.text_input(
        "Paste a raw UPI narration",
        value="UPI-SWIGGY-swiggy@ybl-4441-223344556677-Payment from Phone",
    )
    if sample_input:
        from preprocessing import clean_narration
        cleaned = clean_narration(sample_input)
        predicted = model.predict([cleaned])[0]
        c1, c2 = st.columns(2)
        with c1:
            theme.kpi_card("Cleaned merchant", cleaned, accent=theme.COLORS["slate"])
        with c2:
            theme.kpi_card("Predicted category", predicted, accent=theme.COLORS["marigold"])

# --- Tab 3: Anomalies ---
with tab3:
    theme.section_title(
        "Anomaly detection",
        "Isolation Forest + category-relative z-scores flag transactions that are unusually large "
        "overall, or unusually large for their specific category (e.g. \u20B93,000 is normal for "
        "Shopping but very unusual for Bills & Utilities).",
    )

    contamination = st.slider("Sensitivity (expected anomaly rate)", 0.01, 0.10, 0.03, 0.01)
    result = detect_anomalies(df, contamination=contamination)
    anomalies = top_anomalies(result, n=25)

    theme.kpi_card("Anomalies flagged", str(int(result["is_anomaly"].sum())),
                    "out of " + str(len(result)) + " transactions", accent=theme.COLORS["terracotta"])
    st.markdown("<br/>", unsafe_allow_html=True)
    st.dataframe(anomalies, use_container_width=True)

    if len(anomalies) > 0:
        fig4 = px.scatter(
            result[~result["is_credit"]], x="date", y="abs_amount", color="is_anomaly",
            title="Spend Timeline \u2014 Anomalies Highlighted",
            labels={"abs_amount": "Amount (\u20B9)"},
            color_discrete_map={True: theme.COLORS["terracotta"], False: theme.COLORS["slate"]},
        )
        st.plotly_chart(theme.style_fig(fig4), use_container_width=True)

# --- Tab 4: Forecast ---
with tab4:
    theme.section_title(
        "Spend forecasting",
        "Forecasts next month's spend using a weighted moving average blended with a linear "
        "trend across your monthly totals.",
    )

    forecast = forecast_next_month_total(df)
    f1, f2, f3 = st.columns(3)
    with f1:
        theme.kpi_card("Forecasted Next Month", f"\u20B9{forecast['forecast']:,.0f}",
                        accent=theme.COLORS["marigold"])
    with f2:
        theme.kpi_card("Range", f"\u20B9{forecast['low']:,.0f} \u2013 \u20B9{forecast['high']:,.0f}",
                        accent=theme.COLORS["slate"])
    with f3:
        theme.kpi_card("Trend", forecast["trend"].capitalize(),
                        accent=theme.COLORS["teal"] if forecast["trend"] != "increasing" else theme.COLORS["terracotta"])

    st.markdown("<br/>", unsafe_allow_html=True)
    theme.section_title("Forecast by category", "")
    cat_forecast = forecast_by_category(df)
    fig5 = px.bar(
        cat_forecast, x="category", y="forecast_next_month",
        title="Next Month Forecast by Category", labels={"forecast_next_month": "Forecasted Amount (\u20B9)"},
    )
    fig5.update_traces(marker_color=theme.COLORS["marigold"])
    st.plotly_chart(theme.style_fig(fig5), use_container_width=True)
    st.dataframe(cat_forecast, use_container_width=True)

theme.hairline()
st.caption(
    "Personal Finance AI Tracker \u2014 built with Streamlit, scikit-learn & Plotly. "
    "Sample data is synthetic and for demonstration only."
)
