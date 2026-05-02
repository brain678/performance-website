import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fraunces:wght@600;700&display=swap');

        :root {
            --ink: #0f1720;
            --muted: #526072;
            --surface: #f6f2ec;
            --surface-2: #efe8df;
            --accent: #1b6d5c;
            --accent-strong: #135246;
            --accent-2: #c25b22;
            --card: #ffffff;
            --border: #e6ded3;
        }

        .stApp {
            background: radial-gradient(circle at top, #f9f5ef 0%, #f2ede6 45%, #e8e1d7 100%);
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7f1ea 0%, #efe6dc 100%);
            border-right: 1px solid var(--border);
        }

        .hero {
            background: linear-gradient(120deg, #ffffff 0%, #f1faf6 100%);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 28px 32px;
            margin-bottom: 20px;
            box-shadow: 0 12px 28px rgba(18, 24, 38, 0.08);
        }

        .hero-title {
            font-family: 'Fraunces', serif;
            font-size: 34px;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .hero-sub {
            color: var(--muted);
            font-size: 16px;
            margin-bottom: 12px;
        }

        .badge-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .badge {
            background: rgba(27, 109, 92, 0.14);
            color: #0e4a3e;
            padding: 6px 12px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 12px;
        }

        .section-title {
            font-size: 18px;
            font-weight: 700;
            margin: 14px 0 10px 0;
        }

        .kpi-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 8px 18px rgba(18, 24, 38, 0.05);
        }

        .kpi-title {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .kpi-value {
            font-size: 22px;
            font-weight: 700;
            margin-top: 4px;
        }

        .tier-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 8px 18px rgba(18, 24, 38, 0.05);
            border-left: 6px solid var(--accent);
        }

        .tier-high {
            border-left-color: #c25b22;
        }

        .tier-medium {
            border-left-color: #1b6d5c;
        }

        .tier-low {
            border-left-color: #4b7bbd;
        }

        .tier-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
        }

        .tier-value {
            font-size: 20px;
            font-weight: 700;
            margin-top: 4px;
        }

        .panel {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 8px 18px rgba(18, 24, 38, 0.05);
        }

        .stButton > button,
        .stDownloadButton > button {
            background: var(--accent);
            color: #ffffff;
            border: 1px solid var(--accent-strong);
            border-radius: 12px;
            padding: 0.55rem 1rem;
            font-weight: 600;
            box-shadow: 0 8px 16px rgba(27, 109, 92, 0.2);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: var(--accent-strong);
            border-color: var(--accent-strong);
            color: #ffffff;
            box-shadow: 0 10px 20px rgba(27, 109, 92, 0.26);
        }

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--ink);
        }

        div[data-baseweb="select"] > div {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--ink);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
        }

        [data-testid="stFileUploader"],
        [data-testid="stSelectbox"],
        [data-testid="stCheckbox"],
        [data-testid="stTextInput"],
        [data-testid="stNumberInput"] {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 12px 14px;
            box-shadow: 0 8px 18px rgba(18, 24, 38, 0.04);
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stCheckbox"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stNumberInput"] label {
            color: var(--accent-2);
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        [data-testid="stCheckbox"] {
            background: linear-gradient(120deg, rgba(27, 109, 92, 0.08), rgba(194, 91, 34, 0.06));
        }

        [data-testid="stCheckbox"] label {
            font-size: 14px;
        }

        [data-testid="baseButton-primary"] > button {
            background: linear-gradient(120deg, #1b6d5c 0%, #135246 100%);
            border: 1px solid #0f3f36;
            color: #ffffff;
        }

        [data-testid="stFileUploader"] {
            border: 2px dashed rgba(27, 109, 92, 0.45);
            background: linear-gradient(120deg, rgba(27, 109, 92, 0.06), rgba(194, 91, 34, 0.05));
        }

        [data-testid="stFileUploader"] button {
            background: var(--accent-2);
            color: #ffffff;
            border: 1px solid #9b481a;
            border-radius: 12px;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Student Performance Intelligence</div>
        <div class="hero-sub">Predict CGPA and identify at-risk students from your uploaded data.</div>
        <div class="badge-row">
            <span class="badge">Risk Detection</span>
            <span class="badge">CGPA Forecast</span>
            <span class="badge">Actionable Insights</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("API Connection")
    api_url = st.text_input("API base URL", value=os.getenv("API_URL", "http://127.0.0.1:8000"))
    timeout_seconds = st.number_input("Timeout (seconds)", min_value=3, max_value=60, value=20)
    check_api = st.button("Check API health")

    if check_api:
        try:
            response = requests.get(f"{api_url}/health", timeout=timeout_seconds)
            if response.status_code == 200:
                st.success("API is reachable.")
            else:
                st.error(f"API health check failed: {response.status_code}")
        except requests.RequestException as exc:
            st.error(f"API health check error: {exc}")

predict_choice = st.selectbox("Prediction type", ["risk", "both", "performance"])
model_family = st.selectbox(
    "Model family",
    ["linear", "decision_tree", "random_forest", "gradient_boosting", "svm"],
)
st.caption("Using the selected model family for both risk and performance.")
show_only_at_risk = st.checkbox("Show only at-risk students", value=True)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if "predictions" not in st.session_state:
    st.session_state["predictions"] = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)
    preview_df = df.copy()
    preview_df = preview_df.drop(
        columns=[
            col
            for col in ["at_risk", "predicted_at_risk", "risk_probability", "predicted_cgpa"]
            if col in preview_df.columns
        ],
        errors="ignore",
    )
    st.dataframe(preview_df, use_container_width=True, height=350)

    if st.button("Run predictions"):
        try:
            records = json.loads(df.to_json(orient="records"))
            payload = {
                "records": records,
                "predict": "both",
                "model_family": model_family,
                "risk_model": None,
                "performance_model": None,
            }
            response = requests.post(
                f"{api_url}/predict",
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
        else:
            data = response.json()
            st.session_state["predictions"] = pd.DataFrame(data.get("predictions", []))

predictions = st.session_state.get("predictions")

if predictions is not None:
    if predictions.empty:
        st.warning("No predictions returned from the API.")
    else:
        st.markdown('<div class="section-title">Risk Dashboard</div>', unsafe_allow_html=True)

        total_count = len(predictions)
        risk_count = int(
            predictions["predicted_at_risk"].sum()
        ) if "predicted_at_risk" in predictions.columns else 0
        risk_rate = (risk_count / total_count) if total_count else 0
        avg_cgpa = (
            predictions["predicted_cgpa"].mean()
            if "predicted_cgpa" in predictions.columns
            else None
        )
        avg_risk = (
            predictions["risk_probability"].mean()
            if "risk_probability" in predictions.columns
            else None
        )

        kpi_cols = st.columns(4)
        kpi_values = [
            ("Total Students", f"{total_count}"),
            ("At-Risk Count", f"{risk_count}"),
            ("Risk Rate", f"{risk_rate:.1%}"),
            ("Avg Predicted CGPA", f"{avg_cgpa:.2f}" if avg_cgpa is not None else "N/A"),
        ]
        for col, (title, value) in zip(kpi_cols, kpi_values):
            col.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        tier_filter = "all"
        if "risk_tier" not in predictions.columns and "risk_probability" in predictions.columns:
            predictions = predictions.copy()
            predictions["risk_tier"] = pd.cut(
                predictions["risk_probability"],
                bins=[-0.001, 0.33, 0.66, 1.0],
                labels=["low", "medium", "high"],
            ).astype("string")

        if "risk_tier" in predictions.columns:
            tier_series = predictions["risk_tier"].astype("string").str.lower()
            st.markdown('<div class="section-title">Risk Tier Distribution</div>', unsafe_allow_html=True)
            tier_counts = (
                tier_series.value_counts().reindex(["high", "medium", "low"], fill_value=0)
            )
            tier_cols = st.columns(3)
            tier_items = [
                ("High", int(tier_counts.get("high", 0)), "tier-high"),
                ("Medium", int(tier_counts.get("medium", 0)), "tier-medium"),
                ("Low", int(tier_counts.get("low", 0)), "tier-low"),
            ]
            for col, (label, value, css_class) in zip(tier_cols, tier_items):
                col.markdown(
                    f"""
                    <div class="tier-card {css_class}">
                        <div class="tier-label">{label} Risk</div>
                        <div class="tier-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            tier_filter = st.selectbox("Risk tier filter", ["all", "high", "medium", "low"], index=0)

        chart_cols = st.columns((3, 2))
        if "department" in predictions.columns and "predicted_at_risk" in predictions.columns:
            dept_risk = (
                predictions.groupby("department")["predicted_at_risk"]
                .mean()
                .sort_values(ascending=False)
            )
            dept_df = dept_risk.reset_index(name="risk_rate")
            with chart_cols[0]:
                st.markdown('<div class="section-title">Risk Rate by Department</div>', unsafe_allow_html=True)
                st.bar_chart(dept_df.set_index("department"))

        if "risk_probability" in predictions.columns:
            with chart_cols[1]:
                st.markdown('<div class="section-title">Risk Probability Spread</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.hist(predictions["risk_probability"].dropna(), bins=20, color="#ff8a3d")
                ax.set_xlabel("Risk probability")
                ax.set_ylabel("Count")
                st.pyplot(fig, clear_figure=True)

        st.markdown('<div class="section-title">At-Risk Roster</div>', unsafe_allow_html=True)
        display_predictions = predictions.copy()
        if predict_choice == "risk":
            display_predictions = display_predictions.drop(
                columns=["predicted_cgpa"],
                errors="ignore",
            )
        if predict_choice == "performance":
            display_predictions = display_predictions.drop(
                columns=["predicted_at_risk", "risk_probability"],
                errors="ignore",
            )
        tier_filtered = display_predictions
        if tier_filter != "all" and "risk_tier" in display_predictions.columns:
            tier_filtered = display_predictions[
                display_predictions["risk_tier"].astype("string").str.lower() == tier_filter
            ]

        display_predictions = tier_filtered
        if show_only_at_risk and "predicted_at_risk" in display_predictions.columns:
            display_predictions = display_predictions[display_predictions["predicted_at_risk"] == 1]

        if display_predictions.empty and tier_filter != "all":
            display_predictions = tier_filtered

        if "risk_probability" in display_predictions.columns:
            display_predictions = display_predictions.sort_values(
                "risk_probability", ascending=False
            )

        st.dataframe(display_predictions, use_container_width=True, height=500)

        csv_data = display_predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv",
        )
