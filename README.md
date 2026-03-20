# Machine Learning-Based Early Maternal Risk Stratification for Resource-Limited Settings (Using Tanzania as a Case Study)

Machine learning–based **early maternal risk stratification for resource-limited healthcare settings**, using Tanzania as a case study.

This project demonstrates how machine learning can help **identify high-risk pregnancies early** using clinical indicators, supporting better monitoring and decision-making in healthcare systems with limited resources, with a focus on the Tanzanian healthcare context.

---

## ✅ What this project does

This repository contains a simple machine learning pipeline that:

1. **Preprocesses the raw dataset** (`data/raw/maternal_dataset_csv.csv`) into a clean dataset (`data/processed/maternal_health_clean.csv`).
2. **Trains and evaluates a Random Forest classifier** on the cleaned data.
3. **Generates evaluation reports** such as a confusion matrix and ROC curve.

---

## 🧰 Getting Started

### 1. Create and activate a Python environment

From the project root:

```bash
python -m venv venv
venv\Scripts\activate
```

> On macOS/Linux use:

```
source venv/bin/activate
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

### 4. Train and evaluate the model

```bash
python src/models/random_forest2.py
```

The script trains a **Random Forest model** and saves evaluation outputs in:

```
reports/confusion_matrix.png
reports/ROC_Curve.png
```

---

## 📁 Project Structure

```
maternal-risk-stratification-ml
│
├── data
│   ├── raw
│   └── processed
│
├── src
│   ├── preprocessing.py
│   └── models
│       └── random_forest2.py
│
├── reports
│
├── requirements.txt
└── README.md
```

---

## 📝 Notes

* If `data/processed/maternal_health_clean.csv` already exists, the preprocessing script will reuse it instead of recreating it.
* Additional models can be added inside `src/models/`.

---

## 📊 Dataset Features

The model uses the following clinical indicators as input features:

| Feature | Description |
|---|---|
| Age | Age of the patient in years |
| SystolicBP | Upper blood pressure value (mmHg) |
| DiastolicBP | Lower blood pressure value (mmHg) |
| BS | Blood glucose level (mmol/L) |
| BodyTemp | Body temperature (°F) |
| HeartRate | Resting heart rate (bpm) |

Target label: `RiskLevel` — `low risk`, `mid risk`, or `high risk`.

---

## 🤝 Contributing

1. Pull the latest `development` branch: `git pull origin development`
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes, commit, and push
4. Open a Pull Request targeting `development` — **not** `main`

---

## ⚠️ Disclaimer

This project is for **educational and research purposes** and should not be used for real clinical decision-making without proper validation.
