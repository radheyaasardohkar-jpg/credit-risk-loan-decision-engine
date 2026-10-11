# Credit Risk & Loan Decision Engine 💳

An end-to-end machine learning project that estimates loan default probability and translates model outputs into an explainable, configurable risk-triage workflow.

> **Disclaimer:** This is an educational portfolio prototype, not a production lending system or a recommendation to approve or reject real applicants.

## Overview

The project explores how historical loan application data can be used to study credit risk through data analysis, feature engineering, model training, probability calibration, risk scoring, explainability, and an interactive Streamlit dashboard.

## Features

- Data cleaning, quality checks, and exploratory data analysis
- Financial feature engineering and missing-value handling
- Logistic Regression baseline
- XGBoost classification and hyperparameter tuning
- Evaluation using ROC-AUC, average precision (PR-AUC), precision, recall, F1, and confusion matrices
- Probability calibration
- Project-defined risk score from 300 to 850
- Configurable approve / manual-review / reject triage policy
- SHAP-based global and individual model explanations
- Interactive Streamlit dashboard
- Downloadable prediction results
- Automated tests for scoring and decision logic

## Technology Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, SHAP, Streamlit, Joblib, Git, and GitHub.

## Dataset

The project uses the Home Credit Default Risk competition dataset.

Official source: https://www.kaggle.com/competitions/home-credit-default-risk/data

Download the data through the official source and follow the competition's applicable terms. Raw data and generated datasets are excluded from this repository.

## Project Structure

```text
credit-risk-loan-decision-engine/
├── app/
│   ├── components/
│   ├── utils/
│   └── streamlit_app.py
├── config/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── model_results/
├── src/
│   ├── data/
│   ├── decision/
│   ├── evaluation/
│   ├── explainability/
│   ├── features/
│   └── models/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Run Locally

1. Clone the repository and open its root directory.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Download the required dataset from the official source and place `application_train.csv` in `data/raw/`.
5. Run the notebooks in sequence to create the engineered dataset, train models, evaluate performance, and generate model artifacts.
6. Launch the dashboard from the project root:

   ```bash
   python -m streamlit run app/streamlit_app.py
   ```

The dashboard requires the model artifacts and engineered data produced by the training workflow.

## Model Evaluation

Logistic Regression is used as a baseline, followed by XGBoost and hyperparameter tuning. Average precision (PR-AUC) is a key metric because defaults are relatively uncommon in the dataset.

Insert the final selected model's measured results here after completing an independent final evaluation. Do not interpret development-set metrics as guaranteed future performance.

## Risk Score and Decision Policy

The risk score is a project-defined transformation of predicted probability of default. The approve / manual-review / reject thresholds are configurable demonstration settings.

The score is not a FICO score or a validated credit standard. The thresholds have not been established as appropriate for real lending.

## Explainability

SHAP is used to examine global feature importance and individual model outputs. Explanations describe model behavior; they do not prove causation or establish that a prediction is fair.

## Responsible AI and Limitations

Historical data may encode past inequalities, and seemingly neutral variables may act as proxies for protected characteristics. Model quality, probability calibration, and subgroup performance can change over time.

Before any operational lending use, the system would require independent validation, fair-lending and legal review, privacy and data governance controls, documented decision reasons, human oversight, monitoring, and drift detection.

## Project Status

Portfolio prototype with model training, evaluation, scoring, and a locally running Streamlit dashboard. Cloud deployment and further validation remain outstanding.