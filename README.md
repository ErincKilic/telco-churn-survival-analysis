# Telco Churn Survival Analysis

A Streamlit application for predicting **customer churn risk over time** using a **Cox Proportional Hazards survival model**.

Unlike standard churn classifiers that only predict whether a customer will churn, this project estimates **forward-looking retention probabilities conditioned on the customer’s current tenure**. The app supports both **single-customer analysis** and **batch risk scoring**. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

## Why this project matters

Churn is not only about **if** a customer leaves, but also **when** they are likely to leave.  
That timing matters for:

- retention campaigns
- customer prioritization
- expected revenue estimation
- intervention planning

This project uses survival analysis to model churn as a **time-to-event problem** instead of reducing it to a simple binary classification task. The training pipeline fits a penalized `CoxPHFitter` model on customer tenure and churn event data. :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

## Project Overview

The app provides two workflows.

### 1) Single Customer Analysis

For one customer profile, the app shows:

- a **conditional survival curve** from the customer’s current tenure
- probability of staying active over the next **6 months** and **12 months**
- **expected remaining months**
- **expected remaining revenue**
- a simple **risk driver breakdown** based on model coefficients :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

### 2) Batch Risk Processor

For a CSV or Excel file of active customers, the app:

- validates required columns
- predicts **12-month retention probability**
- calculates **12-month churn risk**
- ranks customers from highest to lowest risk
- allows downloading the full scored output as CSV :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

## Model Approach

This project uses a **penalized Cox Proportional Hazards model** trained on telco customer data. The training script defines:

- **Duration column:** `Tenure Months`
- **Event column:** `Churn Value` :contentReference[oaicite:13]{index=13}

### Model inputs

The model uses the following customer features:

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
- Payment Method :contentReference[oaicite:14]{index=14} :contentReference[oaicite:15]{index=15}

### Preprocessing

- missing values are dropped during training
- categorical features are one-hot encoded
- expected feature columns are aligned at inference time for app predictions :contentReference[oaicite:16]{index=16} :contentReference[oaicite:17]{index=17}

## Validation Summary

The model was evaluated with both **5-fold cross-validation** and a **train/holdout split**. The saved evaluation summary reports:

- **Mean cross-validated C-index:** 0.861
- **Cross-validation standard deviation:** 0.006
- **Holdout C-index:** 0.850
- **12-month weighted calibration error:** 0.021
- **6-month weighted calibration error:** 0.031 

These results suggest good ranking performance and reasonable holdout calibration on unseen data. :contentReference[oaicite:19]{index=19}

## Repository Structure

```text
.
├── app.py
├── train_model.py
├── model_evaluation_summary.json
├── requirements.txt
└── README.md
