# Machine Learning-Based Early Maternal Risk Stratification for Resource-Limited Settings (Using Tanzania as a Case Study)

## 🌍 Overview

Maternal mortality remains a critical public health challenge in sub-Saharan Africa, particularly in resource-constrained clinical settings where timely risk identification is difficult. This project develops and validates a machine learning pipeline to stratify maternal health risk at the point of care using routine clinical indicators.

We compare a **Random Forest classifier** (primary model) against a **Logistic Regression baseline**, evaluated on an indigenous Tanzanian clinical dataset. The pipeline is designed for deployment in low-resource environments where interpretability and recall for high-risk cases are paramount.

---

## ✅ What this project does

This repository contains a simple machine learning pipeline that:

1. **Preprocesses the raw dataset** (`data/raw/maternal_dataset_csv.csv`) into a clean dataset (`data/processed/maternal_health_clean.csv`).
2. **Trains and evaluates a Random Forest classifier** (primary model) and a **Logistic Regression baseline** on the cleaned data.
3. **Generates evaluation reports** such as confusion matrices, ROC curves, feature importance, and odds ratios.

---

## 🧰 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Easycode05/maternal-risk-stratification-ml.git
cd maternal-risk-stratification-ml
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run preprocessing

```bash
python src/preprocessing.py
```

This reads:

```
data/raw/maternal_dataset_csv.csv
```

and generates:

```
data/processed/maternal_health_clean.csv
```

---

### 4. Train and evaluate the models

**Random Forest (Primary Model):**
```bash
python src/models/random_forest2.py
```

**Logistic Regression (Baseline Model):**
```bash
python src/models/logistic_regression2.py
```

---

### 5. Output

- Trained models are saved to `saved_models/`
- Figures are saved to `reports/`

---

## 📁 Project Structure

```
maternal-risk-stratification-ml
│
├── data
│   ├── raw
│   │   └── maternal_dataset_csv.csv
│   └── processed
│       └── maternal_health_clean.csv
│
├── src
│   ├── preprocessing.py
│   ├── utils
│   └── models
│       ├── random_forest2.py
│       ├── logistic_regression2.py
│       └── requirements.txt
│
├── reports
│   ├── rf_metrics
│   ├── lr_metrics
│   └── eda_images
│       ├── age_distribution.png
│       ├── bmi_distribution.png
│       ├── diastolic_bp_distribution.png
│       ├── gestational_age_distribution.png
│       ├── haemoglobin_distribution.png
│       ├── pulse_rate_distribution.png
│       ├── risk_distribution.png
│       └── systolic_bp_distribution.png
│
├── saved_models
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   └── lr_scaler.pkl
│
├── deployment
│   ├── mhrs_app0.py
│   ├── mhrs_app1.py
│   └── mhrs_app2.py
│
├── notebooks
│   ├── eda_analysis_4.ipynb
│   ├── eda_analysis_4.py
│   └── requirements.txt
│
├── requirements.txt
└── README.md
```

---

## 📝 Notes

* If `data/processed/maternal_health_clean.csv` already exists, the preprocessing script will reuse it instead of recreating it.
* Additional models can be added inside `src/models/`.

---

## 📊 Dataset

| Attribute | Detail |
|---|---|
| Source | Zenodo — University of Dodoma & Muhimbili National Hospital |
| Collection | 5 districts, Tanzania |
| Records | 8,817 |
| Features | 11 clinical indicators |
| Target | Binary — risk (0 = Low Risk, 1 = High Risk) |
| Split | 60% Train / 20% Validation / 20% Test (Stratified, random_state=42) |

---

## 📊 Dataset Information

The model uses the following 11 clinical indicators as input features:

| Feature | Description |
|---|---|
| age | Age of the patient in years |
| pulse_rate | Resting pulse rate (bpm) |
| haemoglobin | Haemoglobin level (g/dL) |
| gestational_age | Gestational age (weeks) |
| systolic_bp | Upper blood pressure value (mmHg) |
| diastolic_bp | Lower blood pressure value (mmHg) |
| bmi | Body Mass Index (kg/m²) |
| malaria_rdt | Malaria rapid diagnostic test result (0/1) |
| miscarriage_history | History of miscarriage (0/1) |
| alcohol_use | Alcohol use (0/1) |
| hiv_status | HIV status (0/1) |

Target label: `risk` — Binary (0 = Low Risk, 1 = High Risk).

---

## 📈 Results Summary

| Metric | Random Forest | Logistic Regression |
|---|---|---|
| Model Role | Primary Model | Baseline Model |
| AUC-ROC | 0.994 | 0.914 |
| Accuracy | 87.1% | 83.0% |
| Recall (High Risk) | 90.5% | 90.4% |
| Precision (High Risk) | 89.1% | 83.4% |
| F1-Score (High Risk) | 89.8% | 86.8% |
| Specificity (Low Risk Recall) | ~90% | 69.8% |
| Classification Threshold | 0.35 | 0.35 |

> The Random Forest model outperforms the Logistic Regression baseline across all key metrics, achieving a near-perfect AUC-ROC of 0.994 while maintaining high recall for high-risk cases — critical for clinical deployment in resource-limited settings.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes** and should not be used for real clinical decision-making without proper validation.
