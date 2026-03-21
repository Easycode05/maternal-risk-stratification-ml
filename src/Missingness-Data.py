



# %% [markdown]
# # Maternal Dataset Missingness Analysis
# This script analyzes missing data and visualizes it inline, fully compatible with VS Code interactive mode.

# %%
# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode
import seaborn as sns
from IPython.display import display
import warnings

warnings.filterwarnings("ignore")

# Show plots inline (like Jupyter)


# %%
# --- Features and rename mapping ---
FEATURES = [
    'age', 'height_cm', 'weight_kg', 'blood_pressure_v1',
    'pulse_rate_v1', 'hemoglobin_check_result_v1',
    'pregnant_week_number_v1', 'miscarriages_or_abortions',
    'malaria_rapid_test_result_v1', 'use_of_alcohol', 'hiv', 'Risk'
]

RENAME = {
    'pulse_rate_v1': 'pulse_rate',
    'hemoglobin_check_result_v1': 'haemoglobin',
    'pregnant_week_number_v1': 'gestational_age',
}

# %%
# --- Load dataset ---
df = pd.read_csv(
    r"C:\Users\Adeyanju Olu\Desktop\AI\maternal_dataset_csv.csv",
    dtype=str,
    low_memory=False
)

# Keep only selected features and rename columns
df = df[FEATURES].copy()
df.rename(columns=RENAME, inplace=True)

# Replace common missing value markers with NaN
missing_values = ["", " ", "NA", "N/A", "na", "null", "None", "unknown"]
df.replace(missing_values, np.nan, inplace=True)

# Convert numeric columns
numeric_cols = [
    'age', 'height_cm', 'weight_kg', 'blood_pressure_v1',
    'pulse_rate', 'haemoglobin', 'gestational_age',
    'miscarriages_or_abortions'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# %%
# --- Display first few rows inline ---
display(df.head())

# %%
# --- Separate columns with missing data ---
cols_no_missing = ['age', 'height_cm', 'hiv', 'Risk', 'use_of_alcohol']
df_missing = df.drop(columns=cols_no_missing)

# %%
# --- Missing Data Summary ---
missing_count = df_missing.isnull().sum().sort_values(ascending=False)
missing_percent = (df_missing.isnull().sum() / len(df)) * 100

missing_summary = pd.DataFrame({
    'Missing_Count': missing_count,
    'Missing_%': missing_percent
}).sort_values(by='Missing_%', ascending=False)

display(missing_summary)

# %%
# --- Missing Data Visualization ---
plt.figure(figsize=(12,6))
missing_summary['Missing_%'].plot(kind='bar', color='skyblue')
plt.title("Missing Percentage per Feature")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.show()

# Heatmap of missing values
plt.figure(figsize=(10,6))
sns.heatmap(df_missing.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap (Selected Features Only)")
plt.show()

# Correlation of missingness
plt.figure(figsize=(8,6))
sns.heatmap(df_missing.isnull().corr(), annot=True, cmap='coolwarm')
plt.title("Correlation of Missingness")
plt.show()

# %%
# --- Rows with missing values ---
missing_rows = df[df_missing.isnull().any(axis=1)]
display(f"Number of rows with missing values: {len(missing_rows)}")

# %%
# --- Boxplot: Age vs Haemoglobin Missingness ---
plt.figure(figsize=(8,6))
sns.boxplot(x=df['haemoglobin'].isnull(), y=df['age'])
plt.title("Age distribution vs Haemoglobin Missingness")
plt.xticks([0,1], ['Haemoglobin Present', 'Haemoglobin Missing'])
plt.show()