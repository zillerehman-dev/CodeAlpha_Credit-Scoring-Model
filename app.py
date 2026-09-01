"""
Credit Risk Intelligence — Streamlit App
Beautiful, production-style UI for credit default prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — modern dark-glass aesthetic
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f766e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 10px 40px rgba(15, 23, 42, 0.3);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-top: 0.4rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        text-align: center;
        height: 100%;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.25rem;
    }

    .risk-low { color: #059669 !important; }
    .risk-medium { color: #d97706 !important; }
    .risk-high { color: #dc2626 !important; }

    .section-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0f766e, #0d9488);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0d9488, #14b8a6);
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.4);
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding: 1rem;
    }

    div[data-testid="stSidebar"] {
        background: #0f172a;
    }
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] .stNumberInput label {
        color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load model & schema
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("credit_default_model.joblib")
    with open("feature_columns.json") as f:
        schema = json.load(f)
    return model, schema

model, schema = load_artifacts()
FEATURE_COLS = schema["feature_columns"]

# ──────────────────────────────────────────────
# Helper: risk level
# ──────────────────────────────────────────────
def risk_level(prob: float) -> tuple[str, str, str]:
    if prob < 0.25:
        return "Low Risk", "risk-low", "🟢"
    elif prob < 0.50:
        return "Moderate Risk", "risk-medium", "🟡"
    else:
        return "High Risk", "risk-high", "🔴"

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>💳 Credit Risk Intelligence</h1>
    <p>AI-powered default prediction · Built with machine learning on real banking data · Instant risk scoring for credit decisions</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar — Customer Inputs
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Customer Profile")
    st.markdown("---")

    # Demographics
    st.markdown("**Demographics**")
    limit_bal = st.number_input("Credit Limit (NT$)", min_value=10000, max_value=1000000, value=150000, step=10000,
                                help="Customer’s approved credit limit")
    sex = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    education = st.selectbox("Education", options=[1, 2, 3, 4],
                             format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}[x])
    marriage = st.selectbox("Marital Status", options=[1, 2, 3],
                            format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])
    age = st.slider("Age", 21, 79, 35)

    st.markdown("---")
    st.markdown("**Repayment Status** *(most recent → oldest)*")
    st.caption("−2 = no consumption · −1 = paid in full · 0 = revolving · 1–8 = months delayed")

    pay_0 = st.select_slider("PAY_0 (Sep)", options=list(range(-2, 9)), value=0)
    pay_2 = st.select_slider("PAY_2 (Aug)", options=list(range(-2, 9)), value=0)
    pay_3 = st.select_slider("PAY_3 (Jul)", options=list(range(-2, 9)), value=0)
    pay_4 = st.select_slider("PAY_4 (Jun)", options=list(range(-2, 9)), value=0)
    pay_5 = st.select_slider("PAY_5 (May)", options=list(range(-2, 9)), value=0)
    pay_6 = st.select_slider("PAY_6 (Apr)", options=list(range(-2, 9)), value=0)

    st.markdown("---")
    st.markdown("**Bill Amounts (NT$)**")
    bill1 = st.number_input("Bill Sep (BILL_AMT1)", value=45000, step=1000)
    bill2 = st.number_input("Bill Aug (BILL_AMT2)", value=42000, step=1000)
    bill3 = st.number_input("Bill Jul (BILL_AMT3)", value=40000, step=1000)
    bill4 = st.number_input("Bill Jun (BILL_AMT4)", value=38000, step=1000)
    bill5 = st.number_input("Bill May (BILL_AMT5)", value=35000, step=1000)
    bill6 = st.number_input("Bill Apr (BILL_AMT6)", value=32000, step=1000)

    st.markdown("---")
    st.markdown("**Payment Amounts (NT$)**")
    pay_amt1 = st.number_input("Paid Sep (PAY_AMT1)", value=2000, step=500)
    pay_amt2 = st.number_input("Paid Aug (PAY_AMT2)", value=2000, step=500)
    pay_amt3 = st.number_input("Paid Jul (PAY_AMT3)", value=1500, step=500)
    pay_amt4 = st.number_input("Paid Jun (PAY_AMT4)", value=1500, step=500)
    pay_amt5 = st.number_input("Paid May (PAY_AMT5)", value=1000, step=500)
    pay_amt6 = st.number_input("Paid Apr (PAY_AMT6)", value=1000, step=500)

    st.markdown("---")
    predict_btn = st.button("🔍 Assess Credit Risk", use_container_width=True)

# ──────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.markdown("### 📊 Risk Assessment")

    if predict_btn:
        # Build input vector
        input_data = {
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3,
            "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
            "BILL_AMT1": bill1, "BILL_AMT2": bill2, "BILL_AMT3": bill3,
            "BILL_AMT4": bill4, "BILL_AMT5": bill5, "BILL_AMT6": bill6,
            "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2, "PAY_AMT3": pay_amt3,
            "PAY_AMT4": pay_amt4, "PAY_AMT5": pay_amt5, "PAY_AMT6": pay_amt6,
        }
        input_df = pd.DataFrame([input_data])[FEATURE_COLS]

        # Predict
        pred = model.predict(input_df)[0]
        proba = float(model.predict_proba(input_df)[0][1])
        level, css_class, emoji = risk_level(proba)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": f"{emoji} Default Probability", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#0f766e" if proba < 0.25 else ("#d97706" if proba < 0.5 else "#dc2626")},
                "bgcolor": "#f1f5f9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#d1fae5"},
                    {"range": [25, 50], "color": "#fef3c7"},
                    {"range": [50, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 3},
                    "thickness": 0.75,
                    "value": proba * 100,
                },
            },
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Result cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Prediction</div>
                <div class="value {'risk-high' if pred == 1 else 'risk-low'}">
                    {"Default Likely" if pred == 1 else "No Default"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Risk Level</div>
                <div class="value {css_class}">{level}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Confidence</div>
                <div class="value">{proba*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        # Interpretation
        if proba < 0.25:
            st.success("✅ **Low risk profile.** Recent repayment history looks healthy. Suitable for standard credit terms.")
        elif proba < 0.50:
            st.warning("⚠️ **Moderate risk.** Some signs of stress in repayment or utilization. Consider tighter limits or closer monitoring.")
        else:
            st.error("🚨 **Elevated default risk.** Multiple delayed payments or high utilization detected. Recommend further review before extending credit.")

        # Feature snapshot
        with st.expander("🔎 Key Input Snapshot", expanded=False):
            snap = pd.DataFrame({
                "Feature": ["Credit Limit", "Age", "Most Recent Status (PAY_0)", "Avg Bill (last 3 mo)", "Avg Payment (last 3 mo)"],
                "Value": [
                    f"NT$ {limit_bal:,}",
                    age,
                    pay_0,
                    f"NT$ {np.mean([bill1,bill2,bill3]):,.0f}",
                    f"NT$ {np.mean([pay_amt1,pay_amt2,pay_amt3]):,.0f}",
                ]
            })
            st.dataframe(snap, hide_index=True, use_container_width=True)

    else:
        st.info("👈 Configure the customer profile in the sidebar and click **Assess Credit Risk** to generate a prediction.")
        # Placeholder illustration
        st.markdown("""
        <div class="section-card">
            <h4 style="margin-top:0">How it works</h4>
            <ol style="color:#475569; line-height:1.7">
                <li>Enter demographics, credit limit and 6-month payment history</li>
                <li>The model (HistGradientBoosting pipeline) scores default probability</li>
                <li>Receive instant risk level, confidence and actionable guidance</li>
            </ol>
            <p style="color:#64748b; font-size:0.9rem; margin-bottom:0">
                Model trained on 30,000 real Taiwanese credit-card clients (2005) · ~82% accuracy · ROC-AUC ≈ 0.78
            </p>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown("### 📈 Insights & Context")

    # Mini payment history chart (when prediction made)
    if predict_btn:
        months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
        bills = [bill6, bill5, bill4, bill3, bill2, bill1]
        pays = [pay_amt6, pay_amt5, pay_amt4, pay_amt3, pay_amt2, pay_amt1]
        statuses = [pay_6, pay_5, pay_4, pay_3, pay_2, pay_0]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(name="Bill Amount", x=months, y=bills, marker_color="#94a3b8"))
        fig_hist.add_trace(go.Bar(name="Amount Paid", x=months, y=pays, marker_color="#0d9488"))
        fig_hist.update_layout(
            barmode="group",
            height=260,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter", "size": 12},
            yaxis_title="NT$",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Status timeline
        status_colors = ["#059669" if s <= 0 else ("#d97706" if s == 1 else "#dc2626") for s in statuses]
        fig_status = go.Figure(go.Scatter(
            x=months, y=statuses, mode="lines+markers",
            line=dict(color="#64748b", width=2),
            marker=dict(size=12, color=status_colors),
        ))
        fig_status.update_layout(
            title="Repayment Status Timeline",
            height=220,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(title="Status code", dtick=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter", "size": 12},
        )
        st.plotly_chart(fig_status, use_container_width=True)

    # Always-visible model info
    st.markdown("""
    <div class="section-card">
        <h4 style="margin-top:0">Model Highlights</h4>
        <ul style="color:#475569; line-height:1.8; padding-left:1.2rem">
            <li><b>Algorithm:</b> HistGradientBoosting (sklearn pipeline)</li>
            <li><b>Features:</b> 23 (demographics + 6-mo history)</li>
            <li><b>Training set:</b> 24,000 clients (stratified)</li>
            <li><b>Test accuracy:</b> ~81.6%</li>
            <li><b>ROC-AUC:</b> ~0.78</li>
            <li><b>Strongest signals:</b> PAY_0, LIMIT_BAL, recent bills</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <h4 style="margin-top:0">Business Impact</h4>
        <p style="color:#475569; font-size:0.95rem; line-height:1.6">
            False positives (flagging good customers) create friction; false negatives (missing defaulters) create direct losses.
            The probability score lets risk teams set their own threshold based on cost of capital and risk appetite.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built for CodeAlpha Machine Learning Internship · Educational demo only · Not for production lending decisions<br>
    Data: UCI Default of Credit Card Clients (Taiwan, 2005)
</div>
""", unsafe_allow_html=True)
