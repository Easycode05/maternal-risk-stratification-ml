# Maternal Health Risk Prediction
# Random Forest Model

# import necessary libraries
import pandas as pd
import numpy as np

# Machine Learning tools
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


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


# Save the trained model
import os
import joblib
os.makedirs("saved_models", exist_ok=True)
joblib.dump(rf_model, "saved_models/random_forest_model.pkl")
print("Model saved successfully!")