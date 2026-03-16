# Maternal Risk Stratification (ML)

Machine Learning–based early risk stratification for resource-limited settings.

---

## ✅ What this project does
This repository contains a full (simple) pipeline to:

1. **Preprocess the raw dataset** (`data/raw/maternal_dataset_csv.csv`) into a clean, model-ready CSV (`data/processed/maternal_health_clean.csv`).
2. **Train and evaluate a Random Forest classifier** on the cleaned data.
3. **Generate evaluation reports** (confusion matrix, ROC curve) under `reports/`.

---

## 🧰 Getting Started (Run the project)

### 1) Create & activate a Python environment
From the project root (`maternal-risk-stratification-ml`):

```bash
python -m venv venv
venv\Scripts\activate
```

> 💡 On macOS/Linux use `source venv/bin/activate` instead.


### 2) Install dependencies

```bash
pip install -r requirements.txt
```


### 3) Run preprocessing (generate clean dataset)

```bash
python src/preprocessing.py
```

This reads `data/raw/maternal_dataset_csv.csv`, cleans it, and writes:

- `data/processed/maternal_health_clean.csv`


### 4) Train + evaluate the Random Forest model

```bash
python src/models/random_forest2.py
```

This script trains a Random Forest model, evaluates it on a held-out test set, and saves evaluation plots to:

- `reports/confusion_matrix.png`
- `reports/ROC_Curve.png`


---

## 📁 Key Files / Structure

- `data/raw/` — original dataset (not modified)
- `data/processed/` — cleaned dataset used for training
- `src/preprocessing.py` — data cleaning + feature engineering pipeline
- `src/models/random_forest2.py` — training + evaluation script
- `reports/` — output plots (confusion matrix + ROC)

---

## 📝 Notes / Tips

- If you rerun `src/preprocessing.py` and `data/processed/maternal_health_clean.csv` already exists, the script will **reuse the existing cleaned data** (no need to re-run the clean step unless you want to regenerate it).
- If you want to change the model or add a new one, start by editing or copying `src/models/random_forest2.py`.
