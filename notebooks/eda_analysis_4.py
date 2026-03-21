import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# ==========================================
# STEP 1: LOADING THE DATA
# ==========================================
data_path = os.path.join(os.path.dirname(__file__), '../data/processed/maternal_health_clean.csv')
df = pd.read_csv(data_path)

# ==========================================
# STEP 2: DATA INTEGRITY & TARGET AUDIT
# ==========================================

# 1. Risk Distribution (Target Balance)
# We count how many "High Risk" (1) vs "Low Risk" (0) cases we have.
# This helps us know if our future AI will see enough examples of both.
risk_counts = df['risk'].value_counts().reset_index()
risk_counts.columns = ['Risk Level', 'Count']
risk_counts['Risk Level'] = risk_counts['Risk Level'].map({0: 'Low Risk', 1: 'High Risk'})

fig_risk = px.bar(risk_counts, x='Risk Level', y='Count', 
                   color='Risk Level', title="Class Balance: How many High vs Low Risk patients?",
                   color_discrete_map={'Low Risk': 'green', 'High Risk': 'red'})
fig_risk.write_image("risk_distribution.png")

# ==========================================
# STEP 3: UNIVARIATE ANALYSIS (Individual Profiles)
# ==========================================

# 1. Continuous Features (Numbers like Age, BMI, Blood Pressure)
cont_features = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'pulse_rate', 'haemoglobin', 'gestational_age']

for col in cont_features:
    fig = px.histogram(df, x=col, marginal="box", 
                       title=f"Distribution of {col.replace('_', ' ').title()}",
                       color_discrete_sequence=['skyblue'])
    fig.write_image(f"{col}_distribution.png")

# 2. Categorical Features (Yes/No questions like Malaria or HIV status)
cat_features = ['malaria_rdt', 'miscarriage_history', 'alcohol_use', 'hiv_status']

for col in cat_features:
    temp = df[col].value_counts().reset_index()
    temp.columns = [col, 'count']
    fig = px.bar(temp, x=col, y='count', title=f"Frequency of {col.replace('_', ' ').title()}",
                 color_discrete_sequence=['indianred'])
    fig.write_image(f"{col}_frequency.png")

# ==========================================
# STEP 4: BIVARIATE ANALYSIS (Finding Links)
# ==========================================

# We look specifically at BP, Haemoglobin, and BMI as they are key clinical markers.
clinical_focus = ['systolic_bp', 'haemoglobin', 'bmi']

for col in clinical_focus:
    fig = px.box(df, x='risk', y=col, color='risk',
                 title=f"Does {col.upper()} differ between Risk Groups?",
                 labels={'risk': 'Risk Level (0=Low, 1=High)'},
                 points=False)  # Remove points to avoid clutter
    fig.write_image(f"{col}_risk_comparison.png")

# ==========================================
# STEP 5: SUMMARY REPORT GENERATION
# ==========================================
summary = df[cont_features].describe().transpose()
summary['IQR'] = summary['75%'] - summary['25%']
print("--- CLINICAL SUMMARY STATISTICS ---")
print(summary[['mean', '50%', 'std', 'min', 'max', 'IQR']])