import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------- CONFIG --------------------------------
st.set_page_config(page_title="🩺 Materna | Maternal Health Risk Predictor", layout="wide")

# ------------------------------ LOAD MODELS ------------------------------
log_model = joblib.load("../saved_models/logistic_regression_model.pkl")
scaler = joblib.load("../saved_models/lr_scaler.pkl")
rf_model = joblib.load("../saved_models/random_forest_model.pkl")

THRESHOLD = 0.35   # threshold used for both models

# ------------------------------ CUSTOM STYLING ---------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f9ff;
    }
    .stButton>button {
        background-color: #2a9d8f;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------ TITLE --------------------------------------
st.title("Materna.")
st.write("Enter patient details to predict maternal risk level")

# ------------------------------ SIDEBAR ------------------------------------
st.sidebar.header("Model Settings")
model_option = st.sidebar.radio(
    "Choose Model",
    ["Logistic Regression", "Random Forest", "Compare Both"]
)

# ------------------------------  DYNAMIC INPUT ----------------------------------
features = list(log_model.feature_names_in_)

st.subheader("Patient Information")

input_data = {}

units = {
    "age": "years",
    "systolic_bp": "mmHg",
    "diastolic_bp": "mmHg",
    "pulse_rate": "bpm",
    "haemoglobin": "g/dL",
    "gestational_age": "weeks"
}
for feature in features:
    if "_" in feature:
        base_label = feature.replace("_", " ").title()
    else:
        base_label = feature.title()
    
    if feature in units:
        label = f"{base_label} ({units[feature]})"
    else:
        label = base_label
    
    # BMI
    if feature == "bmi":
        height_cm = st.number_input("Height (cm)", min_value=50.0,  max_value=250.0, value=160.0, step=0.5, format="%.2f")
        weight_kg = st.number_input("Weight (kg)", min_value=20.0,  max_value=300.0, value=65.0,  step=0.5, format="%.2f")
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
        st.write(f"Bmi (kg/m²): {bmi:.2f}")
        input_data["bmi"] = bmi
        
    # Simple heuritic for categorical features
    if feature in ["alcohol_use"]:
        input_data[feature] = st.selectbox(label, ["No", "Yes"])
        input_data[feature] = 1 if input_data[feature] == "Yes" else 0
    elif feature in ["hiv_status"]:
        input_data[feature] = st.selectbox(label, ["Negative", "Positive"])
        input_data[feature] = 1 if input_data[feature] == "Positive" else 0
    elif feature in ["malaria_rdt"]:
        input_data[feature] = st.selectbox(label, ["Negative", "Positive"])
        input_data[feature] = 1 if input_data[feature] == "Positive" else 0
    elif feature in ["miscarriage_history"]:
        input_data[feature] = st.selectbox("Previous Miscarriage", ["No", "Yes"])
        input_data[feature] = 1 if input_data[feature] == "Yes" else 0
    elif feature != "bmi":
        input_data[feature] = st.number_input(label, value=0.0)
        
input_df = pd.DataFrame([input_data])

# Scale input for logistic regression
scaled_input = scaler.transform(input_df)

# ------------------------------ PREDICTION FUNCTION ------------------------
def display_result(prob, model_name):
    st.subheader(f"{model_name} Result")
    
    risk_prob = prob[0][1] # Probability of being high risk
    risk = 1 if risk_prob >= THRESHOLD else 0
    percentage = risk_prob * 100
    
    st.progress(int(percentage))
    
    if risk == 0:
        st.success(f"Low Risk ({percentage:.2f}%)")
        st.markdown("""
        ✅ Routine Antenatal Care Recommended
        
        Recommended next steps:
        - Continue standard antenatal care schedule
        - Reinforce nutrition, iron & folic acid adherence
        - Schedule next routine follow-up visit
        - Reassess at next visit or if symptoms change
        """)
    else:
        st.error(f"High Risk ({percentage:.2f}%)")
        st.warning("""
        ⚠️ Immediate medical attention recommended!
        
        Recommended next steps:
        - Refer patient to specialist or higher-level facility immediately
        - Escalate to obstetric emergency team if applicable
        - Initiate close monitoring — vitals every 30 minutes
        - Document findings and notify supervising clinician
        """)
        
# ------------------------------ PREDICT BUTTON -------------------------------
if st.button("Predict Risk"):
    if model_option == "Logistic Regression":
        prob = log_model.predict_proba(scaled_input)
        display_result(prob, "Logistic Regression")
        
    elif model_option == "Random Forest":
        prob = rf_model.predict_proba(input_df)
        display_result(prob, "Random Forest")
        
    elif model_option == "Compare Both":
        col1, col2 = st.columns(2)
        
        with col1:
            prob = log_model.predict_proba(scaled_input)
            display_result(prob, "Logistic Regression")
            
        with col2:
            prob = rf_model.predict_proba(input_df)
            display_result(prob, "Random Forest")
            
    # ----------------------------- FEATURE IMPORTANCE -----------------------
    st.markdown("---")
    st.subheader("Top Risk Contributors")

    if model_option == "Logistic Regression":
        st.markdown("### Logistic Regression")
        
        importance = np.abs(log_model.coef_[0])
        df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(3)
        
        st.bar_chart(df.set_index("Feature"))
    
    elif model_option == "Random Forest":
        st.markdown("### Random Forest")
        
        importance = rf_model.feature_importances_
        df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(3)
        
        st.bar_chart(df.set_index("Feature"))
    
    elif model_option == "Compare Both":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Logistic Regression")
            importance_log = np.abs(log_model.coef_[0])
            df_log = pd.DataFrame({
            "Feature": features,
            "Importance": importance_log
            }).sort_values(by="Importance", ascending=False).head(3)
    
            st.bar_chart(df_log.set_index("Feature"))    
        
        with col2:
            st.markdown("### Random Forest")
            importance_rf = rf_model.feature_importances_
            df_rf = pd.DataFrame({
            "Feature": features,
            "Importance": importance_rf
            }).sort_values(by="Importance", ascending=False).head(3)

            st.bar_chart(df_rf.set_index("Feature"))

# ----------------------------- FOOTER ------------------------------
st.markdown("---")
st.write("Built with Streamlit for Maternal Health Risk Prediction")
