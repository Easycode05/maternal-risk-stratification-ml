# Maternal Risk Stratification Using Machine Learning

## Overview

This project focuses on predicting maternal health risk levels using machine learning techniques. The goal is to identify pregnant women who may be at low,medium, or high risk based on specific medical indicators.

Early detection of maternal risk can help healthcare professionals make better decisions and improve maternal healthcare outcomes.


## Dataset

The dataset used in this project contains several health-related features, including:

- Age
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Blood Sugar
- Body Temperature
- Heart Rate

These variables are used to predict the maternal risk level.

Dataset location in the project:

data/processed/maternal_health_clean.csv


## Project Structure

maternal-risk-stratification-ml/

│  
├── data/  
│   └── Contains the dataset used for training and testing  

├── notebooks/  
│   └── Jupyter notebooks for data exploration and analysis  

├── src/  
│   └── Source code for building and training machine learning models  

├── reports/  
│   └── Generated results and project outputs  

├── requirements.txt  
│   └── List of Python libraries required for the project  

└── README.md  
    └── Project documentation  


## Installation

### 1. Clone the repository

git clone https://github.com/Centjoe/maternal-risk-stratification-ml.git

### 2. Navigate to the project directory

cd maternal-risk-stratification-ml

### 3. Create a virtual environment

python -m venv venv

### 4. Activate the virtual environment

For Windows:

venv\Scripts\activate

### 5. Install required libraries

pip install -r requirements.txt


## Running the Project

After installing the dependencies, you can run the project scripts or explore the notebooks to train and evaluate the model.

Example:

python src/models/logistic_regression.py

You can also open the **notebooks folder** to explore the data and model development using Jupyter Notebook.


## Tools and Libraries

The project was built using the following tools:

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook


## Author

Udochukwu Joseph