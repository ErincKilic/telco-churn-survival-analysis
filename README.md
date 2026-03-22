# Telco Churn Survival Analysis

A Streamlit application for predicting **customer churn risk over time** using a **Cox Proportional Hazards survival model**.

Unlike standard churn classifiers that only predict whether a customer will churn, this project estimates **when churn risk is likely to occur** and produces **forward-looking retention probabilities conditioned on the customer's current tenure**.

## Why this project matters

Churn is not only about *if* a customer leaves, but also *when* they are likely to leave.  
That timing matters for:

- retention campaigns
- customer prioritization
- expected revenue estimation
- intervention planning

This project uses survival analysis to model churn as a **time-to-event problem** instead of reducing it to a simple binary classification task.

---

## Project Overview

The app provides two workflows:

### 1. Single Customer Analysis
For one customer profile, the app shows:

- a **conditional survival curve** from the customer's current tenure
- **probability of staying active** over the next 6 and 12 months
- **expected remaining months**
- **expected remaining revenue**
- a simple **risk driver breakdown** based on model coefficients

### 2. Batch Risk Processor
For a CSV or Excel file of active customers, the app:

- validates required columns
- predicts **12-month retention probability**
- calculates **12-month churn risk**
- ranks customers from highest to lowest risk
- allows downloading the full scored output as CSV

---

## Model Approach

This project uses a **penalized Cox Proportional Hazards model** trained on telco customer data.

### Target definition
- **Duration column:** `Tenure Months`
- **Event column:** `Churn Value`

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
- Payment Method

### Preprocessing
- missing values are dropped during training
- categorical features are one-hot encoded
- expected feature columns are aligned at inference time for app predictions

---

## Validation Summary

The model was evaluated with both **5-fold cross-validation** and a **train/holdout split**.

### Main results
- **Mean cross-validated C-index:** 0.861
- **Cross-validation standard deviation:** 0.006
- **Holdout C-index:** 0.850
- **12-month weighted calibration error:** 0.021
- **6-month weighted calibration error:** 0.031

These results suggest that the model has good ranking performance and reasonable calibration on unseen data.

---

## App Features

- interactive Streamlit UI
- tenure-conditioned survival predictions
- visual survival curve
- 6-month and 12-month retention estimates
- expected remaining revenue estimate
- single-customer analysis
- batch churn scoring from CSV/XLSX
- downloadable risk report
- built-in validation snapshot in the app

---

