"""
src/preprocessing.py
Maternal Health Risk Stratification — Preprocessing Pipeline

Dataset : Tanzania MHRS Dataset (8,817 records, 683 columns)
Author  : Group 6 — TechCrush AI Bootcamp
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


#PATHS
INPUT_PATH  = 'data/raw/maternal_dataset_csv.csv'
OUTPUT_PATH = 'data/processed/maternal_health_clean.csv'


#FEATURES TO SELECT FROM RAW DATASET
FEATURES = [
    'age', 'height_cm', 'weight_kg', 'blood_pressure_v1',
    'pulse_rate_v1', 'hemoglobin_check_result_v1',
    'pregnant_week_number_v1', 'miscarriages_or_abortions',
    'malaria_rapid_test_result_v1', 'use_of_alcohol', 'hiv', 'Risk'
]


#COLUMN RENAME MAP
RENAME = {
    'pulse_rate_v1':              'pulse_rate',
    'hemoglobin_check_result_v1': 'haemoglobin',
    'pregnant_week_number_v1':    'gestational_age',
}


#IMPOSSIBLE VALUE THRESHOLDS
# Rows outside these ranges are unambiguous data entry errors.
# Height and weight are filtered before BMI derivation.
# Format: column → (min, max) — use None for no bound.
IMPOSSIBLE = {
    'age':         (15,   50),
    'height_cm':   (130, 180),
    'weight_kg':   (35,  150),
    'systolic_bp': (70,  180),
    'diastolic_bp':(45,  None),
    'pulse_rate_v1':(48, None),
    'hemoglobin_check_result_v1': (4,  20),
    'pregnant_week_number_v1':    (4,  42),
}

# BMI bounds applied after derivation from clean height/weight
BMI_MIN, BMI_MAX = 12, 50


#CONTINUOUS FEATURES (for imputation)
CONTINUOUS = [
    'age', 'bmi', 'systolic_bp', 'diastolic_bp',
    'pulse_rate', 'haemoglobin', 'gestational_age'
]


def load_and_select(filepath):
    """Load raw CSV and select the 13 relevant columns + target."""
    df = pd.read_csv(filepath, low_memory=False)
    print(f"[1] Loaded raw data: {df.shape}")
    return df[FEATURES].copy()


def clean_and_convert(df):
    """
    Convert all features to numeric and handle known
    non-numeric entries before any further processing.
    """
    # Known text entries that mean 'not recorded'
    df['weight_kg']     = df['weight_kg'].replace('no', np.nan)
    df['pulse_rate_v1'] = df['pulse_rate_v1'].replace('not_checked', np.nan)
    df.loc[df['height_cm'] == 0, 'height_cm'] = np.nan

    # Convert to numeric — any remaining non-numeric → NaN
    for col in ['age', 'height_cm', 'weight_kg', 'pulse_rate_v1',
                'hemoglobin_check_result_v1', 'pregnant_week_number_v1']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse blood pressure string "110/70" → two numeric columns
    bp = df['blood_pressure_v1'].str.split('/', expand=True)
    df['systolic_bp']  = pd.to_numeric(bp[0], errors='coerce')
    df['diastolic_bp'] = pd.to_numeric(bp[1], errors='coerce')
    df.drop(columns=['blood_pressure_v1'], inplace=True)

    print(f"[2] Converted to numeric")
    return df


def drop_impossible(df):
    """
    Drop rows with physiologically impossible values.
    Height and weight are filtered first, then BMI is derived
    so that BMI is computed only from valid source values.
    Impossible thresholds are defined in the IMPOSSIBLE constant.
    """
    initial = len(df)

    # Filter height and weight first (before BMI derivation)
    for col in ['height_cm', 'weight_kg']:
        min_val, max_val = IMPOSSIBLE[col]
        if min_val is not None:
            df = df[~(df[col].notna() & (df[col] < min_val))]
        if max_val is not None:
            df = df[~(df[col].notna() & (df[col] > max_val))]

    # Derive BMI from clean height and weight
    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    df.drop(columns=['height_cm', 'weight_kg'], inplace=True)

    # Filter impossible BMI values
    df = df[~(df['bmi'].notna() & (df['bmi'] < BMI_MIN))]
    df = df[~(df['bmi'].notna() & (df['bmi'] > BMI_MAX))]

    # Filter all other continuous features
    other_cols = [c for c in IMPOSSIBLE if c not in ('height_cm', 'weight_kg')]
    for col in other_cols:
        min_val, max_val = IMPOSSIBLE[col]
        if min_val is not None:
            df = df[~(df[col].notna() & (df[col] < min_val))]
        if max_val is not None:
            df = df[~(df[col].notna() & (df[col] > max_val))]

    print(f"[3] Dropped impossible rows : {initial - len(df)} rows : {len(df)} remaining")
    return df


def encode_features(df):
    """
    Encode all binary and target features to numeric (1/0).
    Standardises malaria RDT typos and alcohol entry typo.
    """
    # Malaria RDT: standardise typo variants → 0/1
    negative_vals = ['negative', 'ngative', 'nergative', 'negastive', 'nevative', 'negaive',
                        'negatve', 'negstav', 'megative', 'negarive', 'negativet', 'negarivr',
                        '-.ve', 'ngativeno', 'negstav', 'ne']
    positive_vals = ['positive', 'p0sitive']
    rdt = df['malaria_rapid_test_result_v1'].str.strip().str.lower()
    df['malaria_rdt'] = rdt.apply(
        lambda x: 0 if x in negative_vals else (1 if x in positive_vals else np.nan)
    )
    df.drop(columns=['malaria_rapid_test_result_v1'], inplace=True)

    # Alcohol: fix truncation typo 'ye' → 'yes'
    df['use_of_alcohol'] = (
        df['use_of_alcohol'].str.strip().str.lower().replace('ye', 'yes')
    )

    # Encode yes/no columns → 1/0 and rename
    yes_no_cols = {
        'miscarriages_or_abortions': 'miscarriage_history',
        'use_of_alcohol':            'alcohol_use',
        'hiv':                       'hiv_status',
    }
    for old, new in yes_no_cols.items():
        df[new] = df[old].map({'yes': 1, 'no': 0})
        df.drop(columns=[old], inplace=True)

    # Target variable: high → 1, low → 0
    df['risk'] = df['Risk'].map({'high': 1, 'low': 0})
    df.drop(columns=['Risk'], inplace=True)

    print(f"[4] Encoded binary features")
    return df


def impute_missing(df):
    """
    Impute remaining missing values:
        Continuous features → median
        Malaria RDT (binary) → mode
    """
    # Rename columns first so CONTINUOUS list matches
    df.rename(columns=RENAME, inplace=True)

    for col in CONTINUOUS:
        n = df[col].isna().sum()
        if n > 0:
            df[col] = df[col].fillna(df[col].median())

    # Malaria RDT
    n = df['malaria_rdt'].isna().sum()
    if n > 0:
        df['malaria_rdt'] = df['malaria_rdt'].fillna(df['malaria_rdt'].mode()[0])

    print(f"[5] Imputed missing values  : {df.isnull().sum().sum()} remaining")
    return df


def run_pipeline(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    """
    Master function: runs the full preprocessing pipeline.

    Parameters
    ----------
    input_path  : str — path to raw CSV (default: INPUT_PATH)
    output_path : str — path to save clean CSV (default: OUTPUT_PATH)

    Returns
    -------
    pd.DataFrame — fully clean dataset ready for modelling

    Example
    -------
    # Run with defaults:
    df = run_pipeline()
    
    """
   
    
    print("  MATERNAL HEALTH RISK — PREPROCESSING")
    

    # If clean file already exists, skip cleaning and load directly
    if os.path.exists(output_path):
        print(f"  Clean data already exists. Skipping pipeline.")
        print(f"  Loading from: {output_path}")
        return pd.read_csv(output_path)

    df = load_and_select(input_path)
    df = clean_and_convert(df)
    df = drop_impossible(df)
    df = encode_features(df)
    df = impute_missing(df)

    # Save clean dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[6] Saved clean dataset     : {output_path}")
    print(f"  Final shape  : {df.shape}")
    print(f"  Missing      : {df.isnull().sum().sum()}")
    print(f"  High risk (1): {df['risk'].sum()} ({round(df['risk'].mean()*100, 1)}%)")
    print(f"  Low risk  (0): {(df['risk']==0).sum()} ({round((df['risk']==0).mean()*100, 1)}%)")

    return df


if __name__ == "__main__":
    run_pipeline()
