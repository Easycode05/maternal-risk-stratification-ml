# =============================================================================
# Materna — Maternal Health Risk Prediction App
# Group 6 | TechCrush AI Bootcamp
# =============================================================================
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "saved_models"))
# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Materna | Maternal Health Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --white:       #ffffff;
    --bg:          #f7f9fc;
    --bg-alt:      #eef2f7;
    --teal:        #0891b2;
    --teal-light:  #e0f2fe;
    --teal-mid:    #0e7490;
    --crimson:     #dc2626;
    --crimson-light: #fef2f2;
    --green:       #059669;
    --green-light: #ecfdf5;
    --amber:       #d97706;
    --amber-light: #fffbeb;
    --text:        #0f172a;
    --text-mid:    #475569;
    --text-dim:    #94a3b8;
    --border:      #e2e8f0;
    --border-mid:  #cbd5e1;
    --shadow:      0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.05);
    --shadow-md:   0 4px 16px rgba(0,0,0,0.1);
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text);
}
.stApp { background: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem 2.5rem; max-width: 1300px; }

/* ── Header ── */
.app-header {
    background: var(--white);
    border-bottom: 2px solid var(--teal);
    padding: 1.75rem 2.5rem;
    margin: -1.5rem -2.5rem 2rem -2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-left h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    color: var(--text);
    margin: 0;
    letter-spacing: -0.5px;
}
.header-left h1 span { color: var(--teal); }
.header-left p {
    font-size: 0.85rem;
    color: var(--text-mid);
    margin: 0.2rem 0 0 0;
    font-weight: 300;
}
.header-right {
    text-align: right;
    font-size: 0.75rem;
    color: var(--text-dim);
    line-height: 1.6;
}
.header-badge {
    display: inline-block;
    background: var(--teal-light);
    color: var(--teal-mid);
    font-size: 0.72rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ── Section Labels ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--teal-light);
}

/* ── Cards ── */
.card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
}

/* ── Result Cards ── */
.result-card {
    background: var(--white);
    border-radius: 12px;
    padding: 1.75rem;
    text-align: center;
    box-shadow: var(--shadow-md);
    margin-bottom: 1rem;
}
.result-card.high-risk {
    border-top: 4px solid var(--crimson);
    border-left: 1px solid #fecaca;
    border-right: 1px solid #fecaca;
    border-bottom: 1px solid #fecaca;
}
.result-card.low-risk {
    border-top: 4px solid var(--green);
    border-left: 1px solid #a7f3d0;
    border-right: 1px solid #a7f3d0;
    border-bottom: 1px solid #a7f3d0;
}
.result-model-name {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    margin: 0.25rem 0;
}
.result-label.high { color: var(--crimson); }
.result-label.low  { color: var(--green); }
.result-prob {
    font-size: 3.2rem;
    font-weight: 600;
    line-height: 1;
    margin: 0.5rem 0;
    font-family: 'IBM Plex Sans', sans-serif;
}
.result-prob.high { color: var(--crimson); }
.result-prob.low  { color: var(--green); }
.result-sub { font-size: 0.78rem; color: var(--text-dim); margin-top: 0.2rem; }

/* ── Progress Bar ── */
.prob-bar-container {
    background: var(--bg-alt);
    border-radius: 8px; height: 8px;
    margin: 1rem 0 0.5rem 0; overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 8px; }
.prob-bar-fill.high { background: linear-gradient(90deg, #b91c1c, #dc2626); }
.prob-bar-fill.low  { background: linear-gradient(90deg, #047857, #059669); }

/* ── Clinical Action ── */
.action-banner {
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
    font-size: 0.85rem;
    line-height: 1.7;
}
.action-banner.high {
    background: var(--crimson-light);
    border: 1px solid #fecaca;
    color: #991b1b;
}
.action-banner.low {
    background: var(--green-light);
    border: 1px solid #a7f3d0;
    color: #065f46;
}
.action-banner strong {
    display: block; margin-bottom: 0.35rem;
    font-size: 0.78rem; letter-spacing: 1px;
    text-transform: uppercase; font-weight: 700;
}
.action-steps { margin: 0; padding-left: 1.1rem; }
.action-steps li { margin-bottom: 0.2rem; }

/* ── Contributors ── */
.contrib-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
    box-shadow: var(--shadow);
}
.contrib-title {
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--text-mid); margin-bottom: 0.75rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
}
.contributor-item {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--bg-alt);
}
.contributor-item:last-child { border-bottom: none; }
.contributor-rank {
    font-size: 0.68rem; font-weight: 700;
    color: var(--teal); width: 1.5rem;
}
.contributor-name {
    font-size: 0.85rem; color: var(--text);
    flex: 1; padding: 0 0.75rem;
}
.contributor-bar-wrap {
    width: 90px; background: var(--bg-alt);
    border-radius: 4px; height: 5px; overflow: hidden;
}
.contributor-bar-fill { height: 100%; border-radius: 4px; }
.contributor-direction {
    font-size: 0.7rem; padding: 0.15rem 0.5rem;
    border-radius: 8px; margin-left: 0.5rem; font-weight: 600;
}
.dir-up   { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.dir-down { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }

/* ── BMI Display ── */
.bmi-display {
    background: var(--teal-light);
    border: 1px solid #bae6fd;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    font-size: 0.82rem;
    color: var(--teal-mid);
    margin-bottom: 0.75rem;
    font-weight: 500;
}

/* ── Consensus Banner ── */
.consensus-banner {
    border-radius: 10px;
    padding: 0.9rem 1.25rem;
    text-align: center;
    margin-top: 0.75rem;
    font-size: 0.88rem;
}
.consensus-banner.agree {
    background: var(--green-light);
    border: 1px solid #a7f3d0;
    color: #065f46;
}
.consensus-banner.disagree {
    background: var(--amber-light);
    border: 1px solid #fde68a;
    color: #92400e;
}

/* ── Disclaimer ── */
.disclaimer {
    background: var(--amber-light);
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 0.9rem 1.25rem;
    margin-top: 1.5rem;
    font-size: 0.78rem;
    color: #78350f;
    line-height: 1.6;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--white);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] label { color: var(--text-mid) !important; font-size: 0.83rem !important; }

/* ── Button ── */
.stButton > button {
    background: var(--teal) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s ease !important;
}
.stButton > button:hover { background: var(--teal-mid) !important; }

.divider {
    height: 1px; background: var(--border);
    border: none; margin: 1rem 0;
}

/* ── Landing feature cards ── */
.feature-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.75rem 1.5rem;
    text-align: center;
    box-shadow: var(--shadow);
    height: 100%;
}
.feature-icon { font-size: 2.2rem; margin-bottom: 0.75rem; }
.feature-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem; color: var(--text);
    margin-bottom: 0.5rem;
}
.feature-desc { font-size: 0.82rem; color: var(--text-mid); line-height: 1.6; }

@keyframes pulse-border {
    0%   { box-shadow: 0 0 0 0 rgba(220,38,38,0.3); }
    70%  { box-shadow: 0 0 0 8px rgba(220,38,38,0); }
    100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
}
.pulse { animation: pulse-border 2s infinite; }
</style>
""", unsafe_allow_html=True)


# ── Load Models — cached so they load once only ───────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    rf_model  = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    lr_model  = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
    lr_scaler = joblib.load(os.path.join(MODELS_DIR, "lr_scaler.pkl"))
    return rf_model, lr_model, lr_scaler

try:
    rf_model, lr_model, lr_scaler = load_artifacts()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(model, input_df, scaler=None, model_type="rf", threshold=0.35):
    if model_type == "lr":
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        prob = model.predict_proba(input_scaled)[0][1]
    else:
        prob = model.predict_proba(input_df)[0][1]
    label = "High Risk" if prob >= threshold else "Low Risk"
    return prob, label


# ── SHAP — cached per input to avoid recomputing ─────────────────────────────
@st.cache_data(show_spinner=False)
def get_top_contributors_rf(_model, input_tuple, n=3):
    input_df = pd.DataFrame([input_tuple], columns=_model.feature_names_in_)
    try:
        explainer   = shap.TreeExplainer(_model)
        shap_values = explainer.shap_values(input_df)
        # shap_values[1] = class 1 (High Risk)
        vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0, :, 1]
        contrib_df = pd.DataFrame({
            "Feature": input_df.columns,
            "Impact" : vals,
            "Abs"    : np.abs(vals)
        }).dropna(subset=["Impact"]).sort_values("Abs", ascending=False).head(n)
        return contrib_df
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_top_contributors_lr(_model, _scaler, input_tuple, feature_names, n=3):
    input_df     = pd.DataFrame([input_tuple], columns=feature_names)
    input_scaled = pd.DataFrame(_scaler.transform(input_df), columns=feature_names)
    try:
        explainer   = shap.LinearExplainer(_model, input_scaled)
        shap_values = explainer(input_scaled)
        vals        = shap_values.values[0]
        contrib_df  = pd.DataFrame({
            "Feature": feature_names,
            "Impact" : vals,
            "Abs"    : np.abs(vals)
        }).dropna(subset=["Impact"]).sort_values("Abs", ascending=False).head(n)
        return contrib_df
    except Exception:
        return None


# ── Clinical Action ───────────────────────────────────────────────────────────
def get_clinical_action(label):
    if label == "High Risk":
        return {
            "class": "high",
            "title": "🚨 Immediate Clinical Action Required",
            "steps": [
                "Refer patient to specialist or higher-level facility immediately",
                "Escalate to obstetric emergency team if applicable",
                "Initiate close monitoring — vitals every 30 minutes",
                "Document findings and notify supervising clinician",
            ]
        }
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
        <div class="result-prob {label_class}">{bar_pct}<span style="font-size:1.1rem; font-weight:400">%</span></div>
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

    if contributors is not None and not contributors.empty:
        max_abs = contributors["Abs"].max()
    
        st.markdown("""
        <div class="contrib-card">
        <div class="contrib-title">🔬 Top Risk Contributors</div>
        """, unsafe_allow_html=True)

        for i, row in enumerate(contributors.itertuples(), 1):
            direction = "↑ Increases Risk" if row.Impact > 0 else "↓ Decreases Risk"
            dir_class = "dir-up" if row.Impact > 0 else "dir-down"
            bar_w     = int((row.Abs / max_abs) * 100) if max_abs > 0 else 0
            bar_color = "#dc2626" if row.Impact > 0 else "#059669"
            feat_lbl  = row.Feature.replace("_", " ").title()

            st.markdown(f"""
            <div class="contributor-item">
                <span class="contributor-rank">#{i}</span>
                <span class="contributor-name">{feat_lbl}</span>
                <div class="contributor-bar-wrap">
                    <div class="contributor-bar-fill" style="width:{bar_w}%;background:{bar_color}"></div>
                </div>
                <span class="contributor-direction {dir_class}">{direction}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="header-left">
        <h1>Materna<span>.</span></h1>
        <p>Intelligent Maternal Health Risk Stratification · Tanzania Clinical Dataset</p>
    </div>
    <div class="header-right">
        <span class="header-badge">🩺 TechCrush AI Bootcamp · Group 6</span><br>
        <span style="font-size:0.72rem;">ML-powered clinical decision support</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Could not load model files. Ensure `saved_models/` exists.\n\nError: {load_error}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.25rem 0;">
        <div style="font-family:'Playfair Display',serif; font-size:1.3rem; color:#0f172a;">
            Patient Input
        </div>
        <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.2rem; letter-spacing:1px; text-transform:uppercase;">
            Enter Clinical Parameters
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Model Selection</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Model", ["Both Models", "Random Forest", "Logistic Regression"],
        index=0, label_visibility="collapsed"
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Continuous Variables</div>', unsafe_allow_html=True)

    age             = st.number_input("Age (Years)",                     min_value=1,   max_value=99,   value=25,   step=1)
    systolic_bp     = st.number_input("Systolic Blood Pressure (mmHg)",  min_value=50,  max_value=250,  value=120,  step=1)
    diastolic_bp    = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30,  max_value=150,  value=80,   step=1)
    pulse_rate      = st.number_input("Pulse Rate (beats/min)",          min_value=30,  max_value=220,  value=80,   step=1)
    haemoglobin     = st.number_input("Haemoglobin Level (g/dL)",        min_value=1.0, max_value=25.0, value=11.5, step=0.1, format="%.1f")
    gestational_age = st.number_input("Gestational Age (Weeks)",         min_value=1,   max_value=45,   value=20,   step=1)

    st.markdown('<div class="section-label" style="margin-top:0.5rem;">Height & Weight</div>', unsafe_allow_html=True)
    height_cm = st.number_input("Height (cm)", min_value=50.0,  max_value=250.0, value=160.0, step=0.5, format="%.1f")
    weight_kg = st.number_input("Weight (kg)", min_value=20.0,  max_value=300.0, value=65.0,  step=0.5, format="%.1f")
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
    st.markdown(f'<div class="bmi-display">Calculated BMI: <strong>{bmi} kg/m²</strong></div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Categorical Variables</div>', unsafe_allow_html=True)

    miscarriage_history = st.selectbox("Previous Miscarriage",       ["No (0)", "Yes (1)"])
    malaria_rdt         = st.selectbox("Malaria Rapid Test (RDT)",   ["Negative (0)", "Positive (1)"])
    alcohol_use         = st.selectbox("Alcohol Use",                ["No (0)", "Yes (1)"])
    hiv_status          = st.selectbox("HIV Status",                 ["Negative (0)", "Positive (1)"])

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
    "alcohol_use"        : parse_binary(alcohol_use),
    "hiv_status"         : parse_binary(hiv_status),
}

feature_names = rf_model.feature_names_in_.tolist()
input_df      = pd.DataFrame([input_data])[feature_names]
input_tuple   = tuple(input_data[f] for f in feature_names)

# ── Main Content ──────────────────────────────────────────────────────────────
if not predict_btn:
    col1, col2, col3 = st.columns(3)
    cards = [
        ("🤖", "Dual Model Engine",     "Random Forest & Logistic Regression working in tandem for robust, cross-validated risk assessment."),
        ("🧬", "SHAP Explainability",   "Every prediction is backed by the top 3 clinical factors driving the risk classification."),
        ("🏥", "Actionable Guidance",   "High-risk predictions immediately trigger referral and escalation recommendations for clinicians."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:3rem 0 1rem 0;">
        <div style="font-size:2.5rem; margin-bottom:0.75rem;">🩺</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.35rem; color:#64748b;">
            Enter patient parameters in the sidebar and click
            <span style="color:#0891b2;">Generate Prediction</span>
        </div>
        <div style="font-size:0.82rem; color:#94a3b8; margin-top:0.5rem;">
            Trained on 8,509 Tanzanian maternal health records · AUC-ROC 0.944 (RF)
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    THRESHOLD = 0.35
    st.markdown('<div class="section-label" style="margin-bottom:1.25rem;">Prediction Results</div>', unsafe_allow_html=True)

    if model_choice == "Both Models":
        col1, col2 = st.columns(2)

        with col1:
            rf_prob, rf_label = predict(rf_model, input_df, model_type="rf", threshold=THRESHOLD)
            rf_contrib        = get_top_contributors_rf(rf_model, input_tuple)
            render_result("Random Forest", rf_prob, rf_label, rf_contrib)

        with col2:
            lr_prob, lr_label = predict(lr_model, input_df, scaler=lr_scaler, model_type="lr", threshold=THRESHOLD)
            lr_contrib        = get_top_contributors_lr(lr_model, lr_scaler, input_tuple, feature_names)
            render_result("Logistic Regression", lr_prob, lr_label, lr_contrib)

        if rf_label == lr_label:
            st.markdown(f"""
            <div class="consensus-banner agree">
                <strong>✓ Model Consensus</strong> — Both models agree:
                patient is <strong>{rf_label}</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="consensus-banner disagree">
                <strong>⚡ Model Disagreement</strong> — RF predicts <strong>{rf_label}</strong> ·
                LR predicts <strong>{lr_label}</strong> — defer to clinical judgement
            </div>
            """, unsafe_allow_html=True)

    elif model_choice == "Random Forest":
        col1, _ = st.columns([1, 1])
        with col1:
            rf_prob, rf_label = predict(rf_model, input_df, model_type="rf", threshold=THRESHOLD)
            rf_contrib        = get_top_contributors_rf(rf_model, input_tuple)
            render_result("Random Forest", rf_prob, rf_label, rf_contrib)

    else:
        col1, _ = st.columns([1, 1])
        with col1:
            lr_prob, lr_label = predict(lr_model, input_df, scaler=lr_scaler, model_type="lr", threshold=THRESHOLD)
            lr_contrib        = get_top_contributors_lr(lr_model, lr_scaler, input_tuple, feature_names)
            render_result("Logistic Regression", lr_prob, lr_label, lr_contrib)

    # ── Patient Summary ────────────────────────────────────────────────────────
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
            "alcohol_use"        : "Alcohol Use",
            "hiv_status"         : "HIV Status",
        }
        c1, c2 = st.columns(2)
        items  = list(input_data.items())
        half   = len(items) // 2
        for col, chunk in [(c1, items[:half]), (c2, items[half:])]:
            with col:
                for k, v in chunk:
                    lbl = display_labels.get(k, k.replace("_", " ").title())
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; padding:0.4rem 0;
                                border-bottom:1px solid #f1f5f9; font-size:0.84rem;">
                        <span style="color:#64748b;">{lbl}</span>
                        <span style="color:#0f172a; font-weight:500;">{v}</span>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <strong>⚕️ Clinical Disclaimer:</strong> This tool is intended to support — not replace — clinical decision-making.
        All predictions should be interpreted alongside full clinical assessment by a qualified healthcare provider.
        Developed for research and educational purposes · TechCrush AI Bootcamp Group 6 · Tanzania Maternal Health Dataset.
    </div>
    """, unsafe_allow_html=True)