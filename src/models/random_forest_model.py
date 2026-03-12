# Maternal Health Risk Prediction
# Random Forest Model

# import necessary libraries
import pandas as pd
import numpy as np
import shap

# Machine Learning tools
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
data = pd.read_csv("data/processed/maternal_health_clean.csv")
# Display first rows
print("First 5 rows of dataset:")
print(data.head())

# checking dataset structures
print("\nDataset information:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

# Separate features and target variable

print("Actual columns in dataset:", data.columns.tolist())
X = data.drop("risk", axis=1)
Y = data["risk"]
print("\nFeatures used for training:")
print(X.columns)

# split the data(training and testing)
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)
# train + validation
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_full,
    Y_train_full,
    test_size=0.25,
    stratify=Y_train_full,
    random_state=42
)
print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Test set:", X_test.shape)

# Verify Class distribution
print("\nTraining Class Distribution:")
print(Y_train.value_counts())

print("\nTesting Class Distribution:")
print(Y_test.value_counts())

# Train the random forset model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=300,       # number of trees
    class_weight="balanced",      # handles class imbalance
    random_state=42,
    n_jobs=-1     # use all CPU cores
)
# Train the model
rf_model.fit(X_train, Y_train)
print("Random Forest model trained successfully")

# Predict risk levels for the test data
Y_val_pred = rf_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(Y_val, Y_val_pred))
y_test_pred = rf_model.predict(X_test)
Y_prob = rf_model.predict_proba(X_test)
print("First 5 Probability Predictions:")
print(Y_prob[:5])
Y_prob_high = Y_prob[:, 1]
threshold = 0.35
Y_pred = (Y_prob_high >= threshold).astype(int)
print("First 10 Predictions:")
print(Y_pred[:10])

# Evaluate Model performance 
from  sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
print("\nModel Accuracy:", round(accuracy, 3))

# Classification Report
from sklearn.metrics import classification_report
print("\nClassification Report:\n")
print(classification_report(Y_test, Y_pred))

#confusion Matrix Visualization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"])
plt.title("Confusion Matrix - Random Forset (threshold = 0.35)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Get probabilities again
Y_prob = rf_model.predict_proba(X_test)[:,1]
#Calculate ROC values
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
# Calculate AUC
auc_score = roc_auc_score(Y_test, Y_prob)
print("ROC-AUC Score:", round(auc_score,3))
#plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.show()

# Feature importance
# this shows which health factors influence risk most
importance = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
print("\nFeature Importance:\n")
print(feature_importance)

# Plot Feature Importance
plt.figure(figsize=(8,5))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance
)
plt.title("Feature Importance - Random Forest")
plt.show()
# Create SHAP explainer
sample = X_test.sample(200)
explainer = shap.TreeExplainer(rf_model)
# Calculate SHAP values
shap_values = explainer(X_test)
# Summary plot
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, max_display=10)
print(X_test.shape)
print(shap_values.shape)
sample_index = 0
sample = X_test.iloc[sample_index:sample_index+1]
sample_prob = rf_model.predict_proba(sample)[0][1]
threshold = 0.35
prediction ="HIGH RISK" if sample_prob >= threshold else "LOW RISK"
print(f"\nPrediction: {prediction} ({sample_prob*100:.0f}%)")

high_risk_indices = np.where(Y_pred ==1)[0]
high_risk_index = high_risk_indices[0]
high_risk_sample = X_test.iloc[high_risk_index:high_risk_index+1]
high_risk_prob = rf_model.predict_proba(high_risk_sample)[0][1]
print(f"\nPrediction: HIGH RISK ({high_risk_prob*100:.0f}%)")                                                     
explainer = shap.Explainer(rf_model,X_train)
low_risk_index = np.where(Y_pred == 0)[0][0]
low_risk_patient = X_test.iloc[low_risk_index:low_risk_index+1]
low_prob = rf_model.predict_proba(low_risk_patient)[0][1]
print("Prediction: LOW RISK (", round(low_prob*100,2), "% )") 
shap_values_low = explainer(low_risk_patient)
shap.plots.waterfall(shap_values_low[0, :, 1]) 
high_risk_index = np.where(Y_pred == 1)[0][0]
high_risk_patient = X_test.iloc[high_risk_index:high_risk_index+1]
high_prob = rf_model.predict_proba(high_risk_patient)[0][1]
print("Prediction: HIGH RISK (", round(high_prob*100,2), "% )")
shap_values_high = explainer(high_risk_patient)
shap.plots.waterfall(shap_values_high[0, :, 1])
impact = shap_values_high.values[0, :, 1]
features = X_test.columns
df = pd.DataFrame({
    "Features": features,
    "Impact": impact
})

df["abs"] = df["Impact"].abs()
df = df.sort_values(by="abs", ascending=False)

print("\nTop Factors Influencing Prediction:\n")
print(df[["Features", "Impact"]].head(5))
print("\nFactors Increasing risk:")
print(df[df["Impact"] > 0]["Features"].head(3))
print("\nFactors decreasing risk:")
print(df[df["Impact"] < 0]["Features"].head(3))

# Save the trained model
import os
import joblib
os.makedirs("saved_models", exist_ok=True)
joblib.dump(rf_model, "saved_models/random_forest_model.pkl")
print("Model saved successfully!")