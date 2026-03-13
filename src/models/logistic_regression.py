import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load cleaned dataset ──────────────────────────────────────────
df = pd.read_csv('../../data/processed/maternal_health_clean.csv')

# ── 2. Define features and target ────────────────────────────────────
X = df.drop(columns=['risk'])
y = df['risk']



# First split: 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Second split: temp split into 50% validation, 50% test (10% each of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ── 4. Scale features ─────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
X_val = scaler.transform(X_val) 
# ── 5. Train Logistic Regression ──────────────────────────────────────
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# ── 6. Predictions ────────────────────────────────────────────────────
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# ── 7. Evaluation Metrics ─────────────────────────────────────────────
print("=" * 50)
print("   LOGISTIC REGRESSION — EVALUATION METRICS")
print("=" * 50)
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_pred_prob):.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

# 7b. Validation Metrics
y_val_pred = model.predict(X_val)
y_val_pred_prob = model.predict_proba(X_val)[:, 1]
print()
print('=' * 50)
print('   VALIDATION SET METRICS')
print('=' * 50)
print(f'Accuracy  : {accuracy_score(y_val, y_val_pred):.4f}')
print(f'Precision : {precision_score(y_val, y_val_pred):.4f}')
print(f'Recall    : {recall_score(y_val, y_val_pred):.4f}')
print(f'F1 Score  : {f1_score(y_val, y_val_pred):.4f}')
print(f'ROC-AUC   : {roc_auc_score(y_val, y_val_pred_prob):.4f}')
print(classification_report(y_val, y_val_pred, target_names=['Low Risk', 'High Risk']))

# ── 8. Confusion Matrix ───────────────────────────────────────────────
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
plt.title('Confusion Matrix — Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('../../reports/figures/lr_confusion_matrix.png')
plt.show()

# ── 9. ROC Curve ──────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc_score(y_test, y_pred_prob):.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve — Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('../../reports/figures/lr_roc_curve.png')
plt.show()

import pickle
with open('logistic_regression.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model saved!')