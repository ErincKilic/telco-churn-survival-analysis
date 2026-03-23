# Telco Churn Survival Analysis

A Streamlit app for predicting **customer churn risk over time** with a **Cox Proportional Hazards survival model**.

Unlike standard churn classifiers, this project estimates **forward-looking retention probabilities** conditioned on a customer’s **current tenure**.

## Features

- **Single Customer Analysis**
  - conditional survival curve from today
  - 6-month and 12-month retention estimates
  - expected remaining months
  - expected remaining revenue
  - model-based driver summary

- **Batch Risk Processor**
  - upload CSV/XLSX files
  - score 12-month retention probability and churn risk
  - rank customers by risk
  - download results as CSV

## Model

The project uses a penalized **Cox Proportional Hazards model** trained on telco customer data with:

- **Duration column:** `Tenure Months`
- **Event column:** `Churn Value`

Main input features include:

- Monthly Charges
- Senior Citizen
- Partner
- Dependents
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method

## Validation

Saved evaluation results:

- **Mean CV C-index:** 0.861
- **CV standard deviation:** 0.006
- **Holdout C-index:** 0.850
- **12-month calibration error:** 0.021
- **6-month calibration error:** 0.031

## Project Files

- `app.py` — Streamlit app
- `train_model.py` — training and evaluation pipeline
- `survival_model.pkl` — trained model bundle
- `model_evaluation_summary.json` — saved evaluation metrics
- `Telco_customer_churn.xlsx` — dataset
- `requirements.txt` — dependencies

## Installation

```bash
pip install -r requirements.txt

## Run the App
streamlit run app.py
