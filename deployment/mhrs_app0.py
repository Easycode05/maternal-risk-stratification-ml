# =============================================================================
# Maternal Health Risk Prediction — Streamlit App
# Group 6 | TechCrush AI Bootcamp
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Materna | Maternal Health Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:       #0a1628;
    --navy-mid:   #0f2044;
    --navy-light: #1a3060;
    --teal:       #00c9a7;
    --teal-dim:   #00a08a;
    --amber:      #f59e0b;
    --crimson:    #ef4444;
    --crimson-dim:#dc2626;
    --text:       #e2e8f0;
    --text-dim:   #94a3b8;
    --card-bg:    rgba(15, 32, 68, 0.85);
    --border:     rgba(0, 201, 167, 0.2);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy);
    color: var(--text);
}
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #091522 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1400px; }

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #ffffff;
    margin: 0;
    letter-spacing: -1px;
}
.app-header h1 span { color: var(--teal); }
.app-header .tagline {
    font-size: 1rem;
    color: var(--text-dim);
    margin: 0.4rem 0 0 0;
    font-weight: 300;
    font-style: italic;
    letter-spacing: 0.3px;
}
.app-header .description {
    font-size: 0.88rem;
    color: #64748b;
    margin: 0.6rem auto 0 auto;
    max-width: 600px;
    line-height: 1.6;
}
.header-badge {
    display: inline-block;
    background: rgba(0,201,167,0.1);
    border: 1px solid var(--border);
    color: var(--teal);
    font-size: 0.72rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin-top: 0.75rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Section Labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.75rem;
}

/* ── Cards ── */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(10px);
}

/* ── Result Cards ── */
.result-card {
    border-radius: 16px;
    padding: 1.75rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
}
.result-card.high-risk {
    background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(220,38,38,0.05) 100%);
    border: 1px solid rgba(239,68,68,0.4);
}
.result-card.low-risk {
    background: linear-gradient(135deg, rgba(0,201,167,0.15) 0%, rgba(0,160,138,0.05) 100%);
    border: 1px solid rgba(0,201,167,0.4);
}
.result-model-name {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
}
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    font-weight: 400;
    margin: 0.25rem 0;
}
.result-label.high { color: var(--crimson); }
.result-label.low  { color: var(--teal); }
.result-prob {
    font-size: 3rem;
    font-weight: 600;
    line-height: 1;
    margin: 0.5rem 0;
}
.result-prob.high { color: var(--crimson); }
.result-prob.low  { color: var(--teal); }
.result-sub { font-size: 0.8rem; color: var(--text-dim); margin-top: 0.25rem; }
.prob-bar-container {
    background: rgba(255,255,255,0.08);
    border-radius: 8px; height: 8px;
    margin: 1rem 0 0.5rem 0; overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 8px; }
.prob-bar-fill.high { background: linear-gradient(90deg, #dc2626, #ef4444); }
.prob-bar-fill.low  { background: linear-gradient(90deg, #00a08a, #00c9a7); }

/* ── Clinical Action Banner ── */
.action-banner {
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
    font-size: 0.88rem;
    line-height: 1.6;
}
.action-banner.high {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.35);
    color: #fca5a5;
}
.action-banner.low {
    background: rgba(0,201,167,0.08);
    border: 1px solid rgba(0,201,167,0.3);
    color: #6ee7b7;
}
.action-banner strong {
    font-weight: 600; display: block; margin-bottom: 0.35rem;
    font-size: 0.82rem; letter-spacing: 1px; text-transform: uppercase;
}
.action-steps { margin: 0; padding-left: 1.1rem; }
.action-steps li { margin-bottom: 0.2rem; }

/* ── Contributors ── */
.contributor-item {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.contributor-item:last-child { border-bottom: none; }
.contributor-rank { font-size: 0.7rem; font-weight: 700; color: var(--teal); width: 1.5rem; }
.contributor-name { font-size: 0.88rem; color: var(--text); flex: 1; padding: 0 0.75rem; text-transform: capitalize; }
.contributor-bar-wrap { width: 100px; background: rgba(255,255,255,0.08); border-radius: 4px; height: 6px; overflow: hidden; }
.contributor-bar-fill { height: 100%; border-radius: 4px; }
.contributor-direction { font-size: 0.72rem; padding: 0.15rem 0.5rem; border-radius: 10px; margin-left: 0.5rem; font-weight: 600; }
.dir-up   { background: rgba(239,68,68,0.15);  color: #f87171; }
.dir-down { background: rgba(0,201,167,0.15);  color: #34d399; }

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 1.5rem;
    font-size: 0.8rem;
    color: #fbbf24;
    line-height: 1.6;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { color: var(--text-dim) !important; font-size: 0.85rem !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dim) 100%) !important;
    color: #0a1628 !important; font-weight: 700 !important;
    font-size: 1rem !important; border: none !important;
    border-radius: 12px !important; padding: 0.75rem 2rem !important;
    width: 100% !important; letter-spacing: 0.5px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover { box-shadow: 0 8px 25px rgba(0,201,167,0.3) !important; }

.teal-divider {
    height: 2px; background: linear-gradient(90deg, var(--teal), transparent);
    border: none; margin: 1.25rem 0; border-radius: 2px;
}

@keyframes pulse-border {
    0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    70%  { box-shadow: 0 0 0 10px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
.pulse { animation: pulse-border 2s infinite; }
</style>
""", unsafe_allow_html=True)


# ── Load Models & Scaler ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf_model  = joblib.load("saved_models/random_forest_model.pkl")
    lr_model  = joblib.load("saved_models/logistic_regression_model.pkl")
    lr_scaler = joblib.load("saved_models/lr_scaler.pkl")
    return rf_model, lr_model, lr_scaler

try:
    rf_model, lr_model, lr_scaler = load_artifacts()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)


# ── Clinical Action Logic ─────────────────────────────────────────────────────
def get_clinical_action(label):
    if label == "High Risk":
        return {
            "class": "high",
            "title": "🚨 Immediate Clinical Action Required",
            "steps": [
                "Refer patient to specialist or higher-level facility immediately",
                "Escalate to obstetric emergency team if applicable",
                "Document findings and notify supervising clinician",
            ]
        }
    else:
        return {
            "class": "low",
            "title": "✅ Routine Antenatal Care Recommended",
            "steps": [
                "Continue standard antenatal care schedule",
                "Reinforce nutrition, iron & folic acid adherence",
                "Schedule next routine follow-up visit",
                "Reassess at next visit or if symptoms change",
            ]
        }


# ── SHAP Contributors ─────────────────────────────────────────────────────────
def get_top_contributors(model, input_df, scaler=None, model_type="rf", n=3):
    try:
        if model_type == "lr":
            input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
            explainer    = shap.LinearExplainer(model, input_scaled)
            shap_values  = explainer(input_scaled)
            vals         = shap_values.values[0]
        else:
            explainer    = shap.Explainer(model, input_df)
            shap_values  = explainer(input_df, check_additivity=False)
            vals         = shap_values.values[0, :, 1]

        contrib_df = pd.DataFrame({
            "Feature": input_df.columns,
            "Impact" : vals,
            "Abs"    : np.abs(vals)
        }).sort_values("Abs", ascending=False).head(n)
        return contrib_df
    except Exception:
        return None


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(model, input_df, scaler=None, model_type="rf", threshold=0.35):
    if model_type == "lr":
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        prob = model.predict_proba(input_scaled)[0][1]
    else:
        prob = model.predict_proba(input_df)[0][1]
    label = "High Risk" if prob >= threshold else "Low Risk"
    return prob, label


# ── Render Result ─────────────────────────────────────────────────────────────
def render_result(model_name, prob, label, contributors):
    risk_class  = "high-risk" if label == "High Risk" else "low-risk"
    label_class = "high"      if label == "High Risk" else "low"
    icon        = "⚠️"        if label == "High Risk" else "✅"
    pulse_class = "pulse"     if label == "High Risk" else ""
    bar_pct     = int(prob * 100)

    st.markdown(f"""
    <div class="result-card {risk_class} {pulse_class}">
        <div class="result-model-name">{model_name}</div>
        <div class="result-label {label_class}">{icon} {label}</div>
        <div class="result-prob {label_class}">{bar_pct}<span style="font-size:1.2rem">%</span></div>
        <div class="result-sub">Probability of High Risk</div>
        <div class="prob-bar-container">
            <div class="prob-bar-fill {label_class}" style="width:{bar_pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    action     = get_clinical_action(label)
    steps_html = "".join(f"<li>{s}</li>" for s in action["steps"])
    st.markdown(f"""
    <div class="action-banner {action['class']}">
        <strong>{action['title']}</strong>
        <ul class="action-steps">{steps_html}</ul>
    </div>
    """, unsafe_allow_html=True)

    if contributors is not None:
        st.markdown("""
        <div class="card" style="margin-top:1rem;">
            <div style="font-family:'DM Serif Display',serif; font-size:1rem; color:#fff; margin-bottom:1rem;">
                🔬 Top 3 Risk Contributors
            </div>
        """, unsafe_allow_html=True)

        max_abs = contributors["Abs"].max()
        for i, row in enumerate(contributors.itertuples(), 1):
            direction  = "↑ Increases Risk" if row.Impact > 0 else "↓ Decreases Risk"
            dir_class  = "dir-up" if row.Impact > 0 else "dir-down"
            bar_width  = int((row.Abs / max_abs) * 100)
            bar_color  = "#ef4444" if row.Impact > 0 else "#00c9a7"
            feat_label = row.Feature.replace("_", " ").title()

            st.markdown(f"""
            <div class="contributor-item">
                <span class="contributor-rank">#{i}</span>
                <span class="contributor-name">{feat_label}</span>
                <div class="contributor-bar-wrap">
                    <div class="contributor-bar-fill" style="width:{bar_width}%;background:{bar_color}"></div>
                </div>
                <span class="contributor-direction {dir_class}">{direction}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
    <h1>Materna<span>.</span></h1>
    <div class="tagline">Intelligent Maternal Risk Stratification</div>
    <div class="description">
        An ML-powered clinical decision support tool for early identification of high-risk pregnancies
        in resource-limited settings — built on indigenous Tanzanian maternal health data.
    </div>
    <span class="header-badge">🩺 TechCrush AI Bootcamp · Group 6</span>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Could not load model files. Ensure `saved_models/` exists.\n\nError: {load_error}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 1.5rem 0;">
        <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; color:#fff;">
            Patient <span style="color:#00c9a7;">Input</span>
        </div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:0.25rem; letter-spacing:1px; text-transform:uppercase;">
            Enter Clinical Parameters
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Model Selection</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Model", ["Both Models", "Random Forest", "Logistic Regression"],
        index=0, label_visibility="collapsed"
    )

    st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Continuous Variables</div>', unsafe_allow_html=True)

    age             = st.number_input("Age (Years)",                     min_value=10,  max_value=99,   value=25,   step=1)
    systolic_bp     = st.number_input("Systolic Blood Pressure (mmHg)",  min_value=50,  max_value=300,  value=120,  step=1)
    diastolic_bp    = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30,  max_value=250,  value=80,   step=1)
    pulse_rate      = st.number_input("Pulse Rate (beats/min)",          min_value=30,  max_value=200,  value=80,   step=1)
    haemoglobin     = st.number_input("Haemoglobin Level (g/dL)",        min_value=3.0, max_value=30.0, value=11.5, step=0.1, format="%.1f")
    gestational_age = st.number_input("Gestational Age (Weeks)",         min_value=1,   max_value=42,   value=20,   step=1)

    st.markdown('<div class="section-label" style="margin-top:0.75rem;">Height & Weight</div>', unsafe_allow_html=True)
    height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=160.0, step=0.5, format="%.1f")
    weight_kg = st.number_input("Weight (kg)", min_value=30.0,  max_value=200.0, value=65.0,  step=0.5, format="%.1f")
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
    st.markdown(f"""
    <div style="background:rgba(0,201,167,0.08); border:1px solid rgba(0,201,167,0.2);
                border-radius:8px; padding:0.5rem 0.75rem; font-size:0.82rem;
                color:#00c9a7; margin-bottom:0.5rem;">
        Calculated BMI: <strong>{bmi} kg/m²</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Categorical Variables</div>', unsafe_allow_html=True)

    miscarriage_history = st.selectbox("Previous Miscarriage",         ["No (0)", "Yes (1)"])
    malaria_rdt         = st.selectbox("Malaria Rapid Test (RDT)",     ["Negative (0)", "Positive (1)"])
    iron_folic_acid     = st.selectbox("Iron & Folic Acid Supplement", ["No (0)", "Yes (1)"])
    alcohol_use         = st.selectbox("Alcohol Use",                  ["No (0)", "Yes (1)"])
    hiv_status          = st.selectbox("HIV Status",                   ["Negative (0)", "Positive (1)"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Generate Prediction", use_container_width=True)


# ── Parse & Build Input ───────────────────────────────────────────────────────
def parse_binary(val):
    return int(val.split("(")[1].replace(")", ""))

input_data = {
    "age"                : age,
    "systolic_bp"        : systolic_bp,
    "diastolic_bp"       : diastolic_bp,
    "pulse_rate"         : pulse_rate,
    "haemoglobin"        : haemoglobin,
    "gestational_age"    : gestational_age,
    "bmi"                : bmi,
    "miscarriage_history": parse_binary(miscarriage_history),
    "malaria_rdt"        : parse_binary(malaria_rdt),
    "iron_folic_acid"    : parse_binary(iron_folic_acid),
    "alcohol_use"        : parse_binary(alcohol_use),
    "hiv_status"         : parse_binary(hiv_status),
}

input_df = pd.DataFrame([input_data])[rf_model.feature_names_in_.tolist()]

# ── Main Content ──────────────────────────────────────────────────────────────
if not predict_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card" style="text-align:center; padding:2rem;">
            <div style="font-size:2.5rem; margin-bottom:0.75rem;">🤖</div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.1rem; color:#fff; margin-bottom:0.5rem;">Dual Model Engine</div>
            <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6;">Random Forest & Logistic Regression working in tandem for robust, cross-validated risk assessment</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card" style="text-align:center; padding:2rem;">
            <div style="font-size:2.5rem; margin-bottom:0.75rem;">🧬</div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.1rem; color:#fff; margin-bottom:0.5rem;">SHAP Explainability</div>
            <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6;">Every prediction is backed by the top 3 clinical factors driving the risk classification</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card" style="text-align:center; padding:2rem;">
            <div style="font-size:2.5rem; margin-bottom:0.75rem;">🏥</div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.1rem; color:#fff; margin-bottom:0.5rem;">Actionable Guidance</div>
            <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6;">High-risk predictions immediately trigger referral and escalation recommendations for clinicians</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:3rem 0;">
        <div style="font-size:3rem; margin-bottom:1rem;">🩺</div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; color:#64748b;">
            Enter patient parameters in the sidebar and click <span style="color:#00c9a7;">Generate Prediction</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown('<div class="section-label" style="margin-bottom:1.25rem;">Prediction Results</div>', unsafe_allow_html=True)

    THRESHOLD = 0.35

    if model_choice == "Both Models":
        col1, col2 = st.columns(2)
        with col1:
            rf_prob, rf_label = predict(rf_model, input_df, model_type="rf", threshold=THRESHOLD)
            rf_contrib        = get_top_contributors(rf_model, input_df, model_type="rf")
            render_result("Random Forest", rf_prob, rf_label, rf_contrib)
        with col2:
            lr_prob, lr_label = predict(lr_model, input_df, scaler=lr_scaler, model_type="lr", threshold=THRESHOLD)
            lr_contrib        = get_top_contributors(lr_model, input_df, scaler=lr_scaler, model_type="lr")
            render_result("Logistic Regression", lr_prob, lr_label, lr_contrib)

        if rf_label == lr_label:
            st.markdown(f"""
            <div style="background:rgba(0,201,167,0.08); border:1px solid rgba(0,201,167,0.3);
                        border-radius:12px; padding:1rem 1.5rem; text-align:center; margin-top:0.5rem;">
                <span style="color:#00c9a7; font-weight:600;">✓ Model Consensus</span>
                <span style="color:#94a3b8; font-size:0.88rem; margin-left:0.5rem;">
                    Both models agree — patient is <strong style="color:#e2e8f0;">{rf_label}</strong>
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.3);
                        border-radius:12px; padding:1rem 1.5rem; text-align:center; margin-top:0.5rem;">
                <span style="color:#f59e0b; font-weight:600;">⚡ Model Disagreement</span>
                <span style="color:#94a3b8; font-size:0.88rem; margin-left:0.5rem;">
                    RF predicts <strong style="color:#e2e8f0;">{rf_label}</strong> ·
                    LR predicts <strong style="color:#e2e8f0;">{lr_label}</strong> — defer to clinical judgement
                </span>
            </div>
            """, unsafe_allow_html=True)

    elif model_choice == "Random Forest":
        col1, _ = st.columns([1, 1])
        with col1:
            rf_prob, rf_label = predict(rf_model, input_df, model_type="rf", threshold=THRESHOLD)
            rf_contrib        = get_top_contributors(rf_model, input_df, model_type="rf")
            render_result("Random Forest", rf_prob, rf_label, rf_contrib)

    else:
        col1, _ = st.columns([1, 1])
        with col1:
            lr_prob, lr_label = predict(lr_model, input_df, scaler=lr_scaler, model_type="lr", threshold=THRESHOLD)
            lr_contrib        = get_top_contributors(lr_model, input_df, scaler=lr_scaler, model_type="lr")
            render_result("Logistic Regression", lr_prob, lr_label, lr_contrib)

    # Patient summary
    with st.expander("📋 View Patient Input Summary", expanded=False):
        display_labels = {
            "age"                : "Age (Years)",
            "systolic_bp"        : "Systolic BP (mmHg)",
            "diastolic_bp"       : "Diastolic BP (mmHg)",
            "pulse_rate"         : "Pulse Rate (beats/min)",
            "haemoglobin"        : "Haemoglobin (g/dL)",
            "gestational_age"    : "Gestational Age (Weeks)",
            "bmi"                : f"BMI (kg/m²) — from {height_cm}cm / {weight_kg}kg",
            "miscarriage_history": "Previous Miscarriage",
            "malaria_rdt"        : "Malaria Rapid Test (RDT)",
            "iron_folic_acid"    : "Iron & Folic Acid Supplement",
            "alcohol_use"        : "Alcohol Use",
            "hiv_status"         : "HIV Status",
        }
        c1, c2  = st.columns(2)
        items   = list(input_data.items())
        half    = len(items) // 2
        for col, chunk in [(c1, items[:half]), (c2, items[half:])]:
            with col:
                for k, v in chunk:
                    lbl = display_labels.get(k, k.replace("_", " ").title())
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; padding:0.4rem 0;
                                border-bottom:1px solid rgba(255,255,255,0.05); font-size:0.85rem;">
                        <span style="color:#94a3b8;">{lbl}</span>
                        <span style="color:#e2e8f0; font-weight:500;">{v}</span>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <strong>⚕️ Clinical Disclaimer:</strong> This tool is intended to support — not replace — clinical decision-making.
        All predictions should be interpreted alongside full clinical assessment by a qualified healthcare provider.
        Developed for research and educational purposes · TechCrush AI Bootcamp Group 6 · Tanzania Maternal Health Dataset.
    </div>
    """, unsafe_allow_html=True)