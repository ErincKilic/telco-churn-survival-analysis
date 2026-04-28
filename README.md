# Telco Churn Survival Analysis

A Streamlit web app that predicts **customer churn risk over time** using a **Cox Proportional Hazards survival model**.

Unlike standard churn classification models that only predict whether a customer will churn, this project estimates **retention probabilities over future time horizons**, conditioned on the customer’s current tenure.

## Live Demo

[Open the Streamlit App](https://telco-churn-survival-analysis-a9.streamlit.app/)

---

## Project Overview

Customer churn is usually treated as a binary classification problem. However, businesses often need to know not only **whether** a customer is likely to churn, but also **when** that churn risk becomes significant.

This project uses survival analysis to estimate:

- customer retention probability over time
- 6-month and 12-month retention probability
- churn risk ranking for batch customer files
- expected remaining customer lifetime
- expected remaining revenue

The model is trained on telco customer data using a penalized **Cox Proportional Hazards model**.

---

## Features

### Single Customer Analysis

The app allows users to enter customer information manually and generate:

- conditional survival curve from the customer’s current tenure
- 6-month retention estimate
- 12-month retention estimate
- expected remaining months
- expected remaining revenue
- model-based churn driver summary

### Batch Risk Processor

The app also supports batch scoring through uploaded files.

Supported formats:

- `.csv`
- `.xlsx`

For each customer, the app calculates:

- 12-month retention probability
- churn risk score
- risk ranking
- downloadable scored results as CSV

---

## Model

The project uses a penalized **Cox Proportional Hazards survival model**.

### Target Setup

| Component | Column |
|---|---|
| Duration | `Tenure Months` |
| Event | `Churn Value` |

### Main Input Features

The model uses customer demographic, service, contract, and billing information, including:

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

---

## Validation Results

Saved model evaluation results:

| Metric | Value |
|---|---:|
| Mean CV C-index | 0.861 |
| CV Standard Deviation | 0.006 |
| Holdout C-index | 0.850 |
| 12-Month Calibration Error | 0.021 |
| 6-Month Calibration Error | 0.031 |

These results suggest that the model has strong ranking performance and reasonable calibration for short-to-medium-term retention estimates.

---

## Project Structure

```text
Telco_Project/
│
├── app.py                          # Streamlit application
├── train_model.py                  # Training and evaluation pipeline
├── survival_model.pkl              # Trained survival model bundle
├── model_evaluation_summary.json   # Saved evaluation metrics
├── Telco_customer_churn.xlsx       # Dataset
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ErincKilic/telco-churn-survival-analysis.git
cd telco-churn-survival-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app locally:

```bash
streamlit run app.py
```

If the `streamlit` command is not recognized, use:

```bash
python -m streamlit run app.py
```

---

## Example Use Cases

This project can be useful for:

- identifying high-risk customers before they churn
- prioritizing retention campaigns
- estimating customer lifetime value
- comparing churn risk across customer segments
- demonstrating survival analysis in a business analytics setting

---

## Why Survival Analysis?

Standard churn models usually answer:

> Will this customer churn?

Survival analysis answers a more useful business question:

> What is the probability that this customer remains active over the next 6 or 12 months?

This makes the model more suitable for retention planning, customer lifetime estimation, and time-aware risk analysis.

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Lifelines
- Scikit-learn
- Plotly / Matplotlib
- OpenPyXL

---

## Limitations

- The model is trained on historical telco customer data and may not generalize perfectly to other industries.
- Predictions should be interpreted as decision-support signals, not absolute outcomes.
- Survival models rely on assumptions such as proportional hazards, which should be checked when applying the model to new datasets.

---

## Author

**Erinç Kılıç**

BEMACS student at Bocconi University, interested in machine learning, artificial intelligence, and applied data science.
