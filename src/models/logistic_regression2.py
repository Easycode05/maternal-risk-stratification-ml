#Maternal Health Risk Prediction — Logistic Regression Model
# Group 6 | TechCrush AI Bootcamp


#Import libraries
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = 'reports/lr_metrics'
os.makedirs(FIGURES_DIR, exist_ok=True)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)

#Load Cleaned Dataset
data = pd.read_csv("data/processed/maternal_health_clean.csv")
print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nStatistical Summary:")
print(data.describe())

#Separate Features and Target
print("\nActual columns in dataset:", data.columns.tolist())
X = data.drop("risk", axis=1)
Y = data["risk"]
print("\nFeatures used for training:")
print(X.columns.tolist())

# DATA SPLITTING — Three-way split with NO leakage
# Step 1: Hold out 20% as the untouched test set
# Step 2: From the remaining 80%, split 25% as validation (= 20% of total)
# Result: 60% train | 20% validation | 20% test

# CRITICAL: Scaler is fit ONLY on X_train — never on val or test.
# The model is retrained on X_train_full (80%) for final evaluation.
# The test set is touched ONCE at the very end — never during development.


# Step 1 — Hold out test set
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# Step 2 — Split training data into train + validation
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_full,
    Y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=Y_train_full
)

print("\n── Data Split Summary ──")
print(f"X_train      : {X_train.shape}      — 60% of data (model development)")
print(f"X_val        : {X_val.shape}       — 20% of data (threshold selection)")
print(f"X_test       : {X_test.shape}       — 20% of data (final evaluation only)")
print(f"X_train_full : {X_train_full.shape}  — 80% of data (final model training)")

print("\n── Class Distribution ──")
print("Train:", Y_train.value_counts().to_dict())
print("Val:  ", Y_val.value_counts().to_dict())
print("Test: ", Y_test.value_counts().to_dict())


# SCALING — Fit on X_train only, transform all splits
scaler = StandardScaler()
X_train_scaled      = scaler.fit_transform(X_train)        # fit + transform
X_val_scaled        = scaler.transform(X_val)              # transform only
X_train_full_scaled = scaler.transform(X_train_full)       # transform only
X_test_scaled       = scaler.transform(X_test)             # transform only

# Wrap back into DataFrames to preserve column names (needed for SHAP)
X_train_scaled      = pd.DataFrame(X_train_scaled,      columns=X.columns)
X_val_scaled        = pd.DataFrame(X_val_scaled,        columns=X.columns)
X_train_full_scaled = pd.DataFrame(X_train_full_scaled, columns=X.columns)
X_test_scaled       = pd.DataFrame(X_test_scaled,       columns=X.columns)

print("\nScaling complete — StandardScaler fit on X_train only.")


# PHASE 1 — DEVELOPMENT MODEL
# Train on X_train (60%) to evaluate on validation set and select threshold

print("PHASE 1 — Development Model (Train on 60%, Validate on 20%)")

lr_dev = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
    solver="lbfgs"
)
lr_dev.fit(X_train_scaled, Y_train)
print("Development model trained successfully.")

#Validation Evaluation
Y_val_prob = lr_dev.predict_proba(X_val_scaled)[:, 1]

print("\n── Validation Results at Different Thresholds ──")
for thresh in [0.50, 0.40, 0.35, 0.30]:
    Y_val_pred = (Y_val_prob >= thresh).astype(int)
    report = classification_report(Y_val, Y_val_pred,
                                   target_names=["Low Risk", "High Risk"],
                                   output_dict=True)
    recall    = report["High Risk"]["recall"]
    precision = report["High Risk"]["precision"]
    accuracy  = accuracy_score(Y_val, Y_val_pred)
    print(f"  Threshold {thresh:.2f} → "
          f"Recall: {recall:.3f} | Precision: {precision:.3f} | Accuracy: {accuracy:.3f}")

# Select threshold
# Review printed results above and set threshold accordingly
THRESHOLD = 0.35
print(f"\nSelected threshold from validation: {THRESHOLD}")

Y_val_final = (Y_val_prob >= THRESHOLD).astype(int)
print("\n── Validation Confusion Matrix (threshold = 0.35) ──")
print(confusion_matrix(Y_val, Y_val_final))
print("\n── Validation Classification Report ──")
print(classification_report(Y_val, Y_val_final, target_names=["Low Risk", "High Risk"]))


# PHASE 2 — FINAL MODEL
# Refit scaler on X_train_full, retrain model on full 80%

print("PHASE 2 — Final Model (Retrain on full 80% training data)")
print("="*60)

# Refit scaler on full training data before final model
final_scaler = StandardScaler()
X_train_full_scaled_final = final_scaler.fit_transform(X_train_full)
X_test_scaled_final       = final_scaler.transform(X_test)

X_train_full_scaled_final = pd.DataFrame(X_train_full_scaled_final, columns=X.columns)
X_test_scaled_final       = pd.DataFrame(X_test_scaled_final,       columns=X.columns)

lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
    solver="lbfgs"
)
lr_model.fit(X_train_full_scaled_final, Y_train_full)
print("Final model trained on X_train_full successfully.")
print(f"Training data size: {X_train_full.shape[0]} rows (80% of dataset)")


# PHASE 3 — FINAL TEST SET EVALUATION

print("PHASE 3 — Final Evaluation on Held-Out Test Set (20%)")
print("="*60)

Y_test_prob = lr_model.predict_proba(X_test_scaled_final)[:, 1]
Y_test_pred = (Y_test_prob >= THRESHOLD).astype(int)

#Accuracy
accuracy = accuracy_score(Y_test, Y_test_pred)
print(f"\nOverall Accuracy : {accuracy:.4f}")

#Classification Report
print("\n── Classification Report ──")
print(classification_report(Y_test, Y_test_pred, target_names=["Low Risk", "High Risk"]))

#Confusion Matrix
cm = confusion_matrix(Y_test, Y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low Risk", "High Risk"],
    yticklabels=["Low Risk", "High Risk"]
)
plt.title(f"Confusion Matrix — Logistic Regression (threshold = {THRESHOLD})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('reports/lr_metrics/lr_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

#ROC-AUC
fpr, tpr, _ = roc_curve(Y_test, Y_test_prob)
auc_score   = roc_auc_score(Y_test, Y_test_prob)
print(f"\nROC-AUC Score: {auc_score:.4f}")

plt.figure()
plt.plot(fpr, tpr, color="darkorange", label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], "r--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig('reports/lr_metrics/lr_ROC_Curve.png', dpi=300, bbox_inches='tight')
plt.close()


# ODDS RATIOS
# Exponentiated coefficients showing multiplicative effect on risk odds
print("\n── Odds Ratios ──")
odds_df = pd.DataFrame({
    "Feature"      : X.columns,
    "Coefficient"  : lr_model.coef_[0],
    "Odds Ratio"   : np.exp(lr_model.coef_[0])
}).sort_values("Odds Ratio", ascending=False)

print(odds_df.to_string(index=False))

#Odds Ratio Plot
plt.figure(figsize=(8, 5))
colors = ["crimson" if o > 1 else "steelblue" for o in odds_df["Odds Ratio"]]
sns.barplot(x="Odds Ratio", y="Feature", data=odds_df, palette=colors)
plt.axvline(x=1, color="black", linestyle="--", linewidth=1, label="OR = 1 (no effect)")
plt.title("Odds Ratios — Logistic Regression\n(OR > 1 increases risk | OR < 1 decreases risk)")
plt.legend()
plt.tight_layout()
plt.savefig('reports/lr_metrics/lr_odds_ratios.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nOdds ratio plot saved to reports/lr_metrics/lr_odds_ratios.png")

#Coefficient Plot
coef_df = odds_df.sort_values("Coefficient", ascending=False)
plt.figure(figsize=(8, 5))
colors = ["crimson" if c > 0 else "steelblue" for c in coef_df["Coefficient"]]
sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette=colors)
plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
plt.title("Feature Coefficients — Logistic Regression\n(Positive = increases risk | Negative = decreases risk)")
plt.tight_layout()
plt.savefig('reports/lr_metrics/lr_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()
print("Coefficient plot saved to reports/lr_metrics/lr_coefficients.png")


# SHAP EXPLAINABILITY
# Uses LinearExplainer — correct explainer for Logistic Regression
print("\n── SHAP Explainability ──")
explainer   = shap.LinearExplainer(lr_model, X_train_full_scaled_final)
shap_values = explainer(X_test_scaled_final)

# Summary plots
shap.summary_plot(shap_values, X_test_scaled_final, show=True)
shap.summary_plot(shap_values, X_test_scaled_final, max_display=10, show=True)

#Low Risk Waterfall 
low_risk_idx     = np.where(Y_test_pred == 0)[0][0]
low_risk_patient = X_test_scaled_final.iloc[low_risk_idx:low_risk_idx + 1]
low_prob         = lr_model.predict_proba(low_risk_patient)[0][1]
print(f"\nLow Risk Sample — Predicted probability: {low_prob * 100:.1f}%")
shap_values_low  = explainer(low_risk_patient)
plt.figure()
shap.plots.waterfall(shap_values_low[0], show=False)
plt.tight_layout()
plt.savefig('reports/lr_metrics/lr_waterfall_low_risk.png', dpi=300, bbox_inches='tight')
plt.close()

# ── High Risk Waterfall ───────────────────────────────────────────────────────
high_risk_idx     = np.where(Y_test_pred == 1)[0][0]
high_risk_patient = X_test_scaled_final.iloc[high_risk_idx:high_risk_idx + 1]
high_prob         = lr_model.predict_proba(high_risk_patient)[0][1]
print(f"\nHigh Risk Sample — Predicted probability: {high_prob * 100:.1f}%")
shap_values_high  = explainer(high_risk_patient)
plt.figure()
shap.plots.waterfall(shap_values_low[0], show=False)
plt.tight_layout()
plt.savefig('reports/lr_metrics/lr_waterfall_high_risk.png', dpi=300, bbox_inches='tight')
plt.close()

# ── Top Contributing Features for High Risk Sample ───────────────────────────
impact_df = pd.DataFrame({
    "Feature" : X_test_scaled_final.columns,
    "Impact"  : shap_values_high.values[0]
})
impact_df["abs"] = impact_df["Impact"].abs()
impact_df = impact_df.sort_values("abs", ascending=False)

print("\nTop Factors Influencing High Risk Prediction:")
print(impact_df[["Feature", "Impact"]].head(3).to_string(index=False))
print("\nFactors Increasing Risk:")
print(impact_df[impact_df["Impact"] > 0]["Feature"].head(3).tolist())
print("\nFactors Decreasing Risk:")
print(impact_df[impact_df["Impact"] < 0]["Feature"].head(3).tolist())


# SAVE FINAL MODEL AND SCALER
# Both must be saved — scaler is required at inference time
os.makedirs("saved_models", exist_ok=True)
joblib.dump(lr_model,    "saved_models/logistic_regression_model.pkl")
joblib.dump(final_scaler, "saved_models/lr_scaler.pkl")
print("\nFinal model saved to   saved_models/logistic_regression_model.pkl")
print("Scaler saved to        saved_models/lr_scaler.pkl")
print("Both files are required for deployment.")