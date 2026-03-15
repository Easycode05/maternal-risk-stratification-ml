import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

# ── 1. Load Cleaned Dataset ───────────────────────────────────────────
df = pd.read_csv('../../data/processed/maternal_health_clean.csv')

# ── 2. Define Features and Target ────────────────────────────────────
X = df.drop(columns=['risk'])
y = df['risk']

# ── 3. Data Splitting (60% Train, 20% Validation, 20% Test) ──────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set size  : {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size      : {X_test.shape[0]}")

# ── 4. Feature Scaling ────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ── 5. Train Logistic Regression Model ───────────────────────────────
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# ── 6. Validate on Validation Set ────────────────────────────────────
print("\n" + "=" * 55)
print("   VALIDATION SET RESULTS")
print("=" * 55)
y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]

print(f"Accuracy  : {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision : {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall    : {recall_score(y_val, y_val_pred):.4f}")
print(f"F1 Score  : {f1_score(y_val, y_val_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_val, y_val_prob):.4f}")

# ── 7. Threshold Tuning on Validation Set ────────────────────────────
print("\n" + "=" * 55)
print("   THRESHOLD TUNING ON VALIDATION SET")
print("=" * 55)
thresholds = [0.4, 0.35, 0.3]
best_threshold = 0.5
best_f1 = 0

for thresh in thresholds:
    y_pred_thresh = (y_val_prob >= thresh).astype(int)
    f1 = f1_score(y_val, y_pred_thresh)
    acc = accuracy_score(y_val, y_pred_thresh)
    print(f"Threshold {thresh} → Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\nBest Threshold: {best_threshold} (F1: {best_f1:.4f})")

# ── 8. Evaluate on Test Set Using Best Threshold ─────────────────────
print("\n" + "=" * 55)
print("   TEST SET RESULTS (Best Threshold)")
print("=" * 55)
y_test_prob = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)

print(f"Accuracy  : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_test_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_test_prob):.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_test_pred,
      target_names=['Low Risk', 'High Risk']))

# ── 9. Save Figures to reports/lr_figures ────────────────────────────
os.makedirs('../../reports/lr_figures', exist_ok=True)

# Confusion Matrix
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
plt.title('Confusion Matrix — Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('../../reports/lr_figures/lr_confusion_matrix.png')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue',
         label=f'AUC = {roc_auc_score(y_test, y_test_prob):.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve — Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('../../reports/lr_figures/lr_roc_curve.png')
plt.show()

# ── 10. Save Model and Scaler to saved_models ─────────────────────────
os.makedirs('../../saved_models', exist_ok=True)
joblib.dump(model, '../../saved_models/logistic_regression_model.pkl')
joblib.dump(scaler, '../../saved_models/lr_scaler.pkl')
print("Model and scaler saved to saved_models/")