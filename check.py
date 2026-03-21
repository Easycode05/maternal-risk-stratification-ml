import joblib
import pandas as pd

# Load cleaned data
data = pd.read_csv("data/processed/maternal_health_clean.csv")
X = data.drop("risk", axis=1)

# Load saved models
rf_model = joblib.load("saved_models/random_forest_model.pkl")
lr_model = joblib.load("saved_models/logistic_regression_model.pkl")

# Check feature alignment
print("── CSV Columns ──")
print(X.columns.tolist())

print("\n── RF Feature Names ──")
print(rf_model.feature_names_in_.tolist())

print("\n── Match RF ──")
print(X.columns.tolist() == rf_model.feature_names_in_.tolist())

print("\n── LR Feature Names ──")
print(lr_model.feature_names_in_.tolist())

print("\n── Match LR ──")
print(X.columns.tolist() == lr_model.feature_names_in_.tolist())