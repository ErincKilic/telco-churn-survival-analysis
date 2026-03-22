import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model_bundle():
    data = joblib.load('survival_model.pkl')
    return {
        'model': data['model'],
        'expected_columns': data['columns'],
        'numeric_features': data.get('numeric_features', ['Monthly Charges']),
        'categorical_features': data.get(
            'categorical_features',
            ['Contract', 'Internet Service', 'Tech Support'],
        ),
        'duration_col': data.get('duration_col', 'Tenure Months'),
        'evaluation': data.get('evaluation', {}),
    }


bundle = load_model_bundle()
cph_model = bundle['model']
expected_columns = bundle['expected_columns']
numeric_features = bundle['numeric_features']
categorical_features = bundle['categorical_features']
duration_col = bundle['duration_col']
evaluation = bundle.get('evaluation', {})
MODEL_MAX_MONTH = int(np.floor(cph_model.baseline_survival_.index.max()))
SERVICE_COLUMNS = [
    'Online Security',
    'Online Backup',
    'Device Protection',
    'Tech Support',
    'Streaming TV',
    'Streaming Movies',
]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.get_dummies(df)
    for col in expected_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    return encoded[expected_columns]


def get_survival_at_month(curve: pd.DataFrame, month: float, column_position: int = 0) -> float:
    if month <= 0:
        return 1.0

    times = curve.index.to_numpy(dtype=float)
    values = curve.iloc[:, column_position].to_numpy(dtype=float)

    if month > times.max():
        return np.nan

    idx = np.searchsorted(times, month, side='right') - 1
    if idx < 0:
        return 1.0

    return float(values[idx])


def build_conditional_curve(curve: pd.DataFrame, current_tenure: int) -> pd.Series:
    max_forward_months = max(MODEL_MAX_MONTH - int(current_tenure), 0)
    months_ahead = np.arange(0, max_forward_months + 1)

    survival_now = get_survival_at_month(curve, current_tenure)
    if np.isnan(survival_now) or survival_now <= 0:
        return pd.Series([np.nan], index=[0])

    conditional_probs = []
    for months in months_ahead:
        future_survival = get_survival_at_month(curve, current_tenure + months)
        if np.isnan(future_survival):
            conditional_probs.append(np.nan)
        else:
            conditional_probs.append(min(max(future_survival / survival_now, 0.0), 1.0))

    return pd.Series(conditional_probs, index=months_ahead, name='Conditional Survival')




def render_model_validation_snapshot() -> None:
    if not evaluation:
        return

    cv_metrics = evaluation.get('cross_validated_c_index', {})
    holdout_metrics = evaluation.get('train_holdout', {})
    calibration = holdout_metrics.get('holdout_calibration', {})

    with st.expander('Model validation snapshot', expanded=False):
        st.caption(
            'These metrics are computed out-of-sample. Cross-validation checks ranking stability, and holdout '
            'calibration checks whether predicted churn probabilities match observed churn by horizon.'
        )

        c1, c2, c3, c4 = st.columns(4)
        cv_mean = cv_metrics.get('mean')
        cv_std = cv_metrics.get('std')
        holdout_c = holdout_metrics.get('holdout_c_index')
        cal_6 = calibration.get('6', {}).get('weighted_absolute_calibration_error')
        cal_12 = calibration.get('12', {}).get('weighted_absolute_calibration_error')

        c1.metric('CV C-index', f"{cv_mean:.3f}" if cv_mean is not None else 'N/A')
        c2.metric('CV Std. Dev.', f"{cv_std:.3f}" if cv_std is not None else 'N/A')
        c3.metric('Holdout C-index', f"{holdout_c:.3f}" if holdout_c is not None else 'N/A')
        c4.metric('12-Month Cal. Error', f"{cal_12:.3f}" if cal_12 is not None else 'N/A')

        if cal_6 is not None:
            st.caption(f'Weighted absolute calibration error at 6 months: {cal_6:.3f}')

def format_probability(probability: float, fallback: str = 'N/A') -> str:
    if pd.isna(probability):
        return fallback
    return f"{probability:.1%}"


def extract_batch_survival_at_targets(survival_matrix: pd.DataFrame, targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=float)
    times = survival_matrix.index.to_numpy(dtype=float)
    values = survival_matrix.to_numpy(dtype=float)

    n_customers = values.shape[1]
    probs = np.full(n_customers, np.nan, dtype=float)

    zero_mask = targets <= 0
    probs[zero_mask] = 1.0

    valid_mask = (targets > 0) & (targets <= times.max())
    if valid_mask.any():
        valid_positions = np.where(valid_mask)[0]
        idx = np.searchsorted(times, targets[valid_mask], side='right') - 1
        probs[valid_positions] = values[idx, valid_positions]

    return probs


def infer_monthly_charge_effect(monthly_charges: float):
    coefficients = cph_model.params_
    avg_charge = 65.0
    difference = monthly_charges - avg_charge
    mc_coef = coefficients.get('Monthly Charges', 0.0)

    if mc_coef == 0 or abs(difference) <= 5:
        return None

    charge_hr = float(np.exp(mc_coef * difference))
    compared_to_avg = f"(${monthly_charges:.0f} vs ${avg_charge:.0f} average)"
    magnitude = abs(charge_hr - 1)

    if charge_hr > 1.05:
        text = f"**Higher Monthly Charges** {compared_to_avg}: increases risk by {(charge_hr - 1):.0%}"
        return ("risk", magnitude, text)

    if charge_hr < 0.95:
        text = f"**Lower Monthly Charges** {compared_to_avg}: decreases risk by {(1 - charge_hr):.0%}"
        return ("protective", magnitude, text)

    return None


def collect_single_customer_inputs() -> tuple[int, float, pd.DataFrame]:
    st.sidebar.header('Customer Profile')

    with st.sidebar.expander('Tenure & Billing', expanded=True):
        current_tenure = st.slider(
            'Current Tenure (Months)',
            min_value=0,
            max_value=MODEL_MAX_MONTH,
            value=min(12, MODEL_MAX_MONTH),
            help='How long the customer has already been active. All future-looking risk is conditioned on this.',
        )
        monthly_charges = st.slider('Monthly Charges ($)', 18.0, 120.0, 70.0)
        contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'], index=1)
        payment_method = st.selectbox(
            'Payment Method',
            ['Electronic check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check'],
        )

    with st.sidebar.expander('Household', expanded=False):
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
        partner = st.selectbox('Partner', ['No', 'Yes'])
        dependents = st.selectbox('Dependents', ['No', 'Yes'])

    with st.sidebar.expander('Phone Services', expanded=False):
        phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
        if phone_service == 'No':
            multiple_lines = 'No phone service'
            st.caption('Multiple Lines is fixed to **No phone service** because the customer has no phone plan.')
        else:
            multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes'])

    with st.sidebar.expander('Internet & Add-ons', expanded=True):
        internet_service = st.selectbox('Internet Service', ['Fiber optic', 'DSL', 'No'])
        internet_dependent_values = {}
        if internet_service == 'No':
            for service_col in SERVICE_COLUMNS:
                internet_dependent_values[service_col] = 'No internet service'
            st.caption('All internet-based add-ons are fixed to **No internet service**.')
        else:
            for service_col in SERVICE_COLUMNS:
                internet_dependent_values[service_col] = st.selectbox(service_col, ['No', 'Yes'])

    input_df = pd.DataFrame([
        {
            'Monthly Charges': monthly_charges,
            'Senior Citizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'Phone Service': phone_service,
            'Multiple Lines': multiple_lines,
            'Internet Service': internet_service,
            'Online Security': internet_dependent_values['Online Security'],
            'Online Backup': internet_dependent_values['Online Backup'],
            'Device Protection': internet_dependent_values['Device Protection'],
            'Tech Support': internet_dependent_values['Tech Support'],
            'Streaming TV': internet_dependent_values['Streaming TV'],
            'Streaming Movies': internet_dependent_values['Streaming Movies'],
            'Contract': contract,
            'Paperless Billing': paperless_billing,
            'Payment Method': payment_method,
        }
    ])
    return current_tenure, monthly_charges, input_df


st.title('Customer Survival Predictor 📉')
st.markdown(
    'Estimate forward-looking churn risk with a Cox survival model. '
    'Predictions are conditioned on each customer\'s **current tenure** and now use a broader set of '
    '**service, billing, and household** features.'
)

st.caption(
    'Current model inputs: Monthly Charges, Senior Citizen, Partner, Dependents, Phone Service, Multiple Lines, '
    'Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, '
    'Streaming Movies, Contract, Paperless Billing, and Payment Method.'
)

render_model_validation_snapshot()

tab1, tab2 = st.tabs(['👤 Single Customer Analysis', '📁 Batch Risk Processor'])

with tab1:
    current_tenure, monthly_charges, input_df = collect_single_customer_inputs()
    input_final = encode_features(input_df)

    absolute_survival_curve = cph_model.predict_survival_function(input_final)
    conditional_survival_curve = build_conditional_curve(absolute_survival_curve, current_tenure)

    st.subheader('Conditional Survival Probability From Today')
    st.caption(
        f'This curve starts at month {current_tenure} of the customer lifecycle and shows the probability '
        f'of remaining active for future months only. The model horizon ends at month {MODEL_MAX_MONTH}.'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        conditional_survival_curve.index,
        conditional_survival_curve.values,
        linewidth=2.5,
    )
    ax.set_xlabel('Months From Today')
    ax.set_ylabel('Probability of Remaining Active')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.fill_between(
        conditional_survival_curve.index,
        conditional_survival_curve.values,
        alpha=0.2,
    )
    st.pyplot(fig)

    st.markdown('### Risk & Revenue Analysis')

    valid_curve = conditional_survival_curve.dropna()
    expected_remaining_months = (
        float(np.trapezoid(valid_curve.values, valid_curve.index)) if len(valid_curve) > 1 else 0.0
    )
    expected_remaining_revenue = expected_remaining_months * monthly_charges

    stay_6_more = get_survival_at_month(absolute_survival_curve, current_tenure + 6)
    stay_12_more = get_survival_at_month(absolute_survival_curve, current_tenure + 12)
    survival_now = get_survival_at_month(absolute_survival_curve, current_tenure)

    prob_6_months = (
        min(max(stay_6_more / survival_now, 0.0), 1.0)
        if not pd.isna(stay_6_more) and not pd.isna(survival_now) and survival_now > 0
        else np.nan
    )
    prob_12_months = (
        min(max(stay_12_more / survival_now, 0.0), 1.0)
        if not pd.isna(stay_12_more) and not pd.isna(survival_now) and survival_now > 0
        else np.nan
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Current Tenure', f'{current_tenure} mo')
    col2.metric('Expected Remaining Months', f'{expected_remaining_months:,.1f}')
    col3.metric('Stay Active Next 6 Months', format_probability(prob_6_months))
    col4.metric('Stay Active Next 12 Months', format_probability(prob_12_months))

    st.metric('Expected Remaining Revenue', f'${expected_remaining_revenue:,.2f}')

    if current_tenure + 12 > MODEL_MAX_MONTH:
        st.warning(
            f'A full 12-month forward estimate is not available for this customer because their current tenure '
            f'already reaches month {current_tenure}, while the trained model only has observed history through month {MODEL_MAX_MONTH}.'
        )

    st.markdown('---')
    st.markdown('### 🔍 Risk Drivers')
    st.write('These drivers come from the model coefficients for the selected profile.')

    coefficients = cph_model.params_
    risk_factors = []
    protective_factors = []

    for col in input_final.columns:
        if input_final[col].iloc[0] == 1:
            coef = coefficients[col]
            hr = float(np.exp(coef))
            clean_name = col.replace('_', ': ')
            if hr > 1.05:
                risk_factors.append((hr - 1, f'**{clean_name}**: increases risk by {(hr - 1):.0%}'))
            elif hr < 0.95:
                protective_factors.append((1 - hr, f'**{clean_name}**: decreases risk by {(1 - hr):.0%}'))

    charge_effect = infer_monthly_charge_effect(monthly_charges)

    mc_coef = float(cph_model.params_.get('Monthly Charges', 0.0))
    difference = monthly_charges - 65.0
    charge_hr = float(np.exp(mc_coef * difference)) if mc_coef != 0 else None

    st.write("DEBUG monthly charges")
    st.write({
        "monthly_charges": monthly_charges,
        "mc_coef": mc_coef,
        "difference_vs_avg": difference,
        "charge_hr": charge_hr,
        "charge_effect": charge_effect,
    })

    if charge_effect is not None:
        effect_type, score, text = charge_effect
        if effect_type == "risk":
            risk_factors.append((score, text))
        else:
            protective_factors.append((score, text))

    risk_factors = [text for _, text in sorted(risk_factors, key=lambda x: x[0], reverse=True)[:6]]
    protective_factors = [text for _, text in sorted(protective_factors, key=lambda x: x[0], reverse=True)[:6]]

    col_risk, col_safe = st.columns(2)

    with col_risk:
        st.error('🚨 **Factors Increasing Churn Risk**')
        if risk_factors:
            for factor in risk_factors:
                st.write(f'- {factor}')
        else:
            st.write('No major risk factors identified.')

    with col_safe:
        st.success('🛡️ **Factors Protecting Customer**')
        if protective_factors:
            for factor in protective_factors:
                st.write(f'- {factor}')
        else:
            st.write('No major protective factors identified.')

with tab2:
    st.header('Batch Risk Predictor')
    st.write(
        'Upload a CSV or Excel file of active customers to identify who is at the highest risk of churning in the next 12 months. '
        f'Your file must include **{duration_col}** plus all model input columns.'
    )

    required_columns = [duration_col, *numeric_features, *categorical_features]
    st.caption('Required columns: ' + ', '.join(required_columns))

    uploaded_file = st.file_uploader('Upload Customer Data', type=['csv', 'xlsx'])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f'Successfully loaded {len(batch_df)} customers.')

        missing_columns = [col for col in required_columns if col not in batch_df.columns]

        if missing_columns:
            st.error('⚠️ Your uploaded file is missing required columns: ' + ', '.join(missing_columns))
        else:
            batch_df = batch_df.copy()
            batch_df[duration_col] = pd.to_numeric(batch_df[duration_col], errors='coerce')
            for col in numeric_features:
                batch_df[col] = pd.to_numeric(batch_df[col], errors='coerce')

            invalid_duration_mask = batch_df[duration_col].isna() | (batch_df[duration_col] < 0)
            invalid_numeric_mask = batch_df[numeric_features].isna().any(axis=1)

            if invalid_duration_mask.any():
                st.error(
                    f'⚠️ The **{duration_col}** column contains missing or negative values. Please clean that column and try again.'
                )
            elif invalid_numeric_mask.any():
                st.error(
                    '⚠️ One or more numeric feature columns contain missing or non-numeric values. '
                    'Please clean those rows and try again.'
                )
            else:
                batch_features = batch_df[numeric_features + categorical_features]
                batch_final = encode_features(batch_features)
                survival_matrix = cph_model.predict_survival_function(batch_final)

                current_tenures = batch_df[duration_col].to_numpy(dtype=float)
                stay_now = extract_batch_survival_at_targets(survival_matrix, current_tenures)
                stay_12_ahead = extract_batch_survival_at_targets(survival_matrix, current_tenures + 12)

                valid_horizon_mask = current_tenures + 12 <= MODEL_MAX_MONTH
                conditional_retention_12 = np.full(len(batch_df), np.nan, dtype=float)

                computable_mask = valid_horizon_mask & (~np.isnan(stay_now)) & (~np.isnan(stay_12_ahead)) & (stay_now > 0)
                conditional_retention_12[computable_mask] = np.clip(
                    stay_12_ahead[computable_mask] / stay_now[computable_mask],
                    0.0,
                    1.0,
                )

                results_df = batch_df.copy()
                results_df['12-Month Retention Probability'] = conditional_retention_12
                results_df['12-Month Churn Risk'] = 1 - conditional_retention_12
                results_df['Prediction Status'] = np.where(
                    valid_horizon_mask,
                    'OK',
                    f'Unavailable: exceeds month {MODEL_MAX_MONTH} training horizon',
                )

                available_predictions = results_df['Prediction Status'].eq('OK').sum()
                unavailable_predictions = len(results_df) - available_predictions
                st.caption(f'Computed valid 12-month forward estimates for {available_predictions} of {len(results_df)} customers.')

                if unavailable_predictions > 0:
                    st.warning(
                        f'{unavailable_predictions} customer(s) could not receive a full 12-month forward estimate because '
                        f'their current tenure pushes the target month beyond the model\'s observed horizon of {MODEL_MAX_MONTH} months.'
                    )

                ranked_df = results_df[results_df['Prediction Status'] == 'OK'].copy()
                ranked_df = ranked_df.sort_values(by='12-Month Churn Risk', ascending=False)

                if ranked_df.empty:
                    st.info('No rows have enough remaining model horizon for a valid 12-month forward risk estimate.')
                else:
                    display_df = ranked_df.copy()
                    display_df['12-Month Retention Probability'] = display_df['12-Month Retention Probability'].apply(
                        lambda x: f'{x:.1%}'
                    )
                    display_df['12-Month Churn Risk'] = display_df['12-Month Churn Risk'].apply(
                        lambda x: f'{x:.1%}'
                    )

                    st.markdown('### 🚨 Top At-Risk Customers')
                    st.dataframe(display_df.head(100), use_container_width=True)

                full_display_df = results_df.copy()
                full_display_df['12-Month Retention Probability'] = full_display_df['12-Month Retention Probability'].apply(
                    lambda x: f'{x:.1%}' if pd.notna(x) else 'N/A'
                )
                full_display_df['12-Month Churn Risk'] = full_display_df['12-Month Churn Risk'].apply(
                    lambda x: f'{x:.1%}' if pd.notna(x) else 'N/A'
                )

                with st.expander('See all rows, including unavailable predictions'):
                    st.dataframe(full_display_df, use_container_width=True)

                csv_output = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label='📥 Download Full Risk Report (CSV)',
                    data=csv_output,
                    file_name='customer_risk_report.csv',
                    mime='text/csv',
                )
