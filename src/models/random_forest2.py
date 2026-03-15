# Maternal Health Risk Prediction — Random Forest Model
# Group 6 | TechCrush AI Bootcamp

#Importing the libaries
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = 'reports/'
os.makedirs(FIGURES_DIR, exist_ok=True)

from sklearn.ensemble import RandomForestClassifier
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
#
# Step 1: Hold out 20% as the untouched test set
# Step 2: From the remaining 80%, split 25% as validation (= 20% of total)
# Result: 60% train | 20% validation | 20% test
#
# CRITICAL: The model is trained on X_train_full (80%) for final evaluation.
# The validation split is used only for threshold selection and early checks.
# The test set is touched ONCE at the very end — never during development.


# Step 1 — Hold out test set (never touched until final evaluation)
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# Step 2 — Split training data into train + validation
# test_size=0.25 of 80% = 20% of total dataset
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


# PHASE 1 — DEVELOPMENT MODEL
# Train on X_train (60%) to evaluate on validation set and select threshold
# This model is NOT the final model — it is used for threshold selection only

print("PHASE 1 — Development Model (Train on 60%, Validate on 20%)")
print("="*60)

rf_dev = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_dev.fit(X_train, Y_train)
print("Development model trained successfully.")

#Validation Evaluation
Y_val_prob = rf_dev.predict_proba(X_val)[:, 1]

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

#Select threshold based on validation results
# Threshold 0.35 selected: best balance of recall ≥ 90% and precision ≥ 85%
THRESHOLD = 0.35
print(f"\nSelected threshold from validation: {THRESHOLD}")

Y_val_final = (Y_val_prob >= THRESHOLD).astype(int)
print("\n── Validation Confusion Matrix (threshold = 0.35) ──")
cm_val = confusion_matrix(Y_val, Y_val_final)
print(cm_val)
print("\n── Validation Classification Report ──")
print(classification_report(Y_val, Y_val_final, target_names=["Low Risk", "High Risk"]))


# PHASE 2 — FINAL MODEL
# Retrain on X_train_full (80%) using the threshold selected from validation
# This is the model that gets evaluated on the test set and saved for deployment
# No test data has been seen at any point before this step


print("PHASE 2 — Final Model (Retrain on full 80% training data)")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_full, Y_train_full)
print("Final model trained on X_train_full successfully.")
print(f"Training data size: {X_train_full.shape[0]} rows (80% of dataset)")


# PHASE 3 — FINAL TEST SET EVALUATION
# Test set is touched HERE for the first time
# All decisions (threshold, hyperparameters) were made using validation only

print("PHASE 3 — Final Evaluation on Held-Out Test Set (20%)")

Y_test_prob  = rf_model.predict_proba(X_test)[:, 1]
Y_test_pred  = (Y_test_prob >= THRESHOLD).astype(int)

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
plt.title(f"Confusion Matrix — Random Forest (threshold = {THRESHOLD})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

#ROC-AUC
fpr, tpr, _ = roc_curve(Y_test, Y_test_prob)
auc_score   = roc_auc_score(Y_test, Y_test_prob)
print(f"\nROC-AUC Score: {auc_score:.4f}")

plt.figure()
plt.plot(fpr, tpr, color="navy", label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], "r--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig('reports/ROC_Curve.png', dpi=300, bbox_inches='tight')
plt.close()

#Feature Importance
importance_df = pd.DataFrame({
    "Feature"    : X.columns,
    "Importance" : rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n── Feature Importance ──")
print(importance_df.to_string(index=False))

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df, color="steelblue")
plt.title("Feature Importance — Random Forest")
plt.tight_layout()
plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()


# SHAP EXPLAINABILITY
# Uses X_train_full as background — matches the final model's training data

print("\n── SHAP Explainability ──")
explainer   = shap.Explainer(rf_model, X_train_full)
shap_values = explainer(X_test, check_additivity=False)

# Summary plots
shap.summary_plot(shap_values[:, :, 1], X_test, show=True)
shap.summary_plot(shap_values[:, :, 1], X_test, max_display=10, show=True)

#Low Risk Waterfall
low_risk_idx     = np.where(Y_test_pred == 0)[0][0]
low_risk_patient = X_test.iloc[low_risk_idx:low_risk_idx + 1]
low_prob         = rf_model.predict_proba(low_risk_patient)[0][1]
print(f"\nLow Risk Sample — Predicted probability: {low_prob * 100:.1f}%")
shap_values_low  = explainer(low_risk_patient)
shap.plots.waterfall(shap_values_low[0, :, 1])

#High Risk Waterfall
high_risk_idx     = np.where(Y_test_pred == 1)[0][0]
high_risk_patient = X_test.iloc[high_risk_idx:high_risk_idx + 1]
high_prob         = rf_model.predict_proba(high_risk_patient)[0][1]
print(f"\nHigh Risk Sample — Predicted probability: {high_prob * 100:.1f}%")
shap_values_high  = explainer(high_risk_patient)
shap.plots.waterfall(shap_values_high[0, :, 1])

#Top Contributing Features for High Risk Sample
impact_df = pd.DataFrame({
    "Feature" : X_test.columns,
    "Impact"  : shap_values_high.values[0, :, 1]
})
impact_df["abs"] = impact_df["Impact"].abs()
impact_df = impact_df.sort_values("abs", ascending=False)

print("\nTop Factors Influencing High Risk Prediction:")
print(impact_df[["Feature", "Impact"]].head(5).to_string(index=False))
print("\nFactors Increasing Risk:")
print(impact_df[impact_df["Impact"] > 0]["Feature"].head(3).tolist())
print("\nFactors Decreasing Risk:")
print(impact_df[impact_df["Impact"] < 0]["Feature"].head(3).tolist())


# SAVE FINAL MODEL
# Saves the model trained on X_train_full — the deployment-ready model


os.makedirs("saved_models", exist_ok=True)
joblib.dump(rf_model, "saved_models/random_forest_model.pkl")
print("\nFinal model saved to saved_models/random_forest_model.pkl")
print("Model is trained on 80% of data and ready for deployment.")