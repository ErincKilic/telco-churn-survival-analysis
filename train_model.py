import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_PATH = Path('Telco_customer_churn.xlsx')
MODEL_PATH = Path('survival_model.pkl')
RANDOM_STATE = 42
PENALIZER = 0.1
HOLDOUT_SIZE = 0.2
CV_FOLDS = 5
CALIBRATION_HORIZONS = (6, 12)
CALIBRATION_BINS = 5

NUMERIC_FEATURES = ['Monthly Charges']
CATEGORICAL_FEATURES = [
    'Senior Citizen',
    'Partner',
    'Dependents',
    'Phone Service',
    'Multiple Lines',
    'Internet Service',
    'Online Security',
    'Online Backup',
    'Device Protection',
    'Tech Support',
    'Streaming TV',
    'Streaming Movies',
    'Contract',
    'Paperless Billing',
    'Payment Method',
]

DURATION_COL = 'Tenure Months'
EVENT_COL = 'Churn Value'
MODEL_FEATURES = [DURATION_COL, EVENT_COL, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Select modeling columns, clean missing data, and one-hot encode categoricals."""
    subset = df[MODEL_FEATURES].copy()
    subset.dropna(inplace=True)

    encoded = pd.get_dummies(
        subset,
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
    )
    return encoded


def build_model() -> CoxPHFitter:
    return CoxPHFitter(penalizer=PENALIZER)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[DURATION_COL, EVENT_COL])


def extract_survival_at_horizon(survival_matrix: pd.DataFrame, horizon: float) -> np.ndarray:
    times = survival_matrix.index.to_numpy(dtype=float)
    if horizon <= 0:
        return np.ones(survival_matrix.shape[1], dtype=float)
    if horizon > times.max():
        return np.full(survival_matrix.shape[1], np.nan, dtype=float)

    idx = np.searchsorted(times, horizon, side='right') - 1
    if idx < 0:
        return np.ones(survival_matrix.shape[1], dtype=float)
    return survival_matrix.iloc[idx].to_numpy(dtype=float)


def c_index_for_split(model: CoxPHFitter, df: pd.DataFrame) -> float:
    partial_hazard = model.predict_partial_hazard(feature_frame(df))
    return float(
        concordance_index(
            df[DURATION_COL],
            -partial_hazard.to_numpy().ravel(),
            df[EVENT_COL],
        )
    )


def cross_validated_concordance(encoded_df: pd.DataFrame) -> dict:
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores: list[float] = []

    for train_idx, val_idx in splitter.split(encoded_df, encoded_df[EVENT_COL]):
        train_df = encoded_df.iloc[train_idx]
        val_df = encoded_df.iloc[val_idx]

        model = build_model()
        model.fit(train_df, duration_col=DURATION_COL, event_col=EVENT_COL)
        fold_scores.append(c_index_for_split(model, val_df))

    return {
        'fold_scores': [float(score) for score in fold_scores],
        'mean': float(np.mean(fold_scores)),
        'std': float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
    }


def calibration_table(
    durations: pd.Series,
    events: pd.Series,
    predicted_event_prob: np.ndarray,
    horizon: int,
    n_bins: int = CALIBRATION_BINS,
) -> tuple[list[dict], float]:
    calibration_df = pd.DataFrame(
        {
            'duration': durations.to_numpy(dtype=float),
            'event': events.to_numpy(dtype=int),
            'predicted_event_prob': predicted_event_prob,
        }
    ).dropna()

    if calibration_df.empty:
        return [], float('nan')

    unique_scores = calibration_df['predicted_event_prob'].nunique()
    if unique_scores < 2:
        kmf = KaplanMeierFitter()
        kmf.fit(calibration_df['duration'], event_observed=calibration_df['event'])
        observed_event_prob = 1.0 - float(kmf.predict(horizon))
        mean_pred = float(calibration_df['predicted_event_prob'].mean())
        abs_error = abs(mean_pred - observed_event_prob)
        table = [
            {
                'bin': 'all',
                'n': int(len(calibration_df)),
                'mean_predicted_event_prob': mean_pred,
                'observed_event_prob_km': observed_event_prob,
                'absolute_error': abs_error,
            }
        ]
        return table, abs_error

    bins = min(n_bins, int(unique_scores))
    calibration_df['bin'] = pd.qcut(
        calibration_df['predicted_event_prob'],
        q=bins,
        labels=False,
        duplicates='drop',
    )

    table: list[dict] = []
    weighted_error = 0.0
    total_n = len(calibration_df)

    for bin_id, group in calibration_df.groupby('bin', sort=True):
        kmf = KaplanMeierFitter()
        kmf.fit(group['duration'], event_observed=group['event'])
        observed_event_prob = 1.0 - float(kmf.predict(horizon))
        mean_pred = float(group['predicted_event_prob'].mean())
        abs_error = abs(mean_pred - observed_event_prob)
        weighted_error += abs_error * (len(group) / total_n)
        table.append(
            {
                'bin': int(bin_id) + 1,
                'n': int(len(group)),
                'mean_predicted_event_prob': mean_pred,
                'observed_event_prob_km': observed_event_prob,
                'absolute_error': abs_error,
            }
        )

    return table, float(weighted_error)


def evaluate_train_holdout(encoded_df: pd.DataFrame) -> dict:
    train_df, holdout_df = train_test_split(
        encoded_df,
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
        stratify=encoded_df[EVENT_COL],
    )

    model = build_model()
    model.fit(train_df, duration_col=DURATION_COL, event_col=EVENT_COL)

    holdout_survival = model.predict_survival_function(feature_frame(holdout_df))
    holdout_calibration: dict[str, dict] = {}

    for horizon in CALIBRATION_HORIZONS:
        survival_probs = extract_survival_at_horizon(holdout_survival, horizon)
        predicted_event_prob = 1.0 - survival_probs
        table, ice = calibration_table(
            durations=holdout_df[DURATION_COL],
            events=holdout_df[EVENT_COL],
            predicted_event_prob=predicted_event_prob,
            horizon=horizon,
        )
        holdout_calibration[str(horizon)] = {
            'weighted_absolute_calibration_error': ice,
            'bin_summary': table,
        }

    return {
        'train_rows': int(len(train_df)),
        'holdout_rows': int(len(holdout_df)),
        'train_event_rate': float(train_df[EVENT_COL].mean()),
        'holdout_event_rate': float(holdout_df[EVENT_COL].mean()),
        'train_c_index': c_index_for_split(model, train_df),
        'holdout_c_index': c_index_for_split(model, holdout_df),
        'holdout_calibration': holdout_calibration,
    }


def train_final_model(encoded_df: pd.DataFrame) -> CoxPHFitter:
    """Fit the production model on the full encoded dataset."""
    model = build_model()
    model.fit(encoded_df, duration_col=DURATION_COL, event_col=EVENT_COL)
    return model


def main() -> None:
    df = pd.read_excel(DATA_PATH)
    encoded_df = prepare_dataset(df)

    cv_metrics = cross_validated_concordance(encoded_df)
    split_metrics = evaluate_train_holdout(encoded_df)

    print(f"Cross-val C-index: {cv_metrics['mean']:.3f} ± {cv_metrics['std']:.3f}")
    print(f"Train C-index: {split_metrics['train_c_index']:.3f}")
    print(f"Holdout C-index: {split_metrics['holdout_c_index']:.3f}")
    for horizon, metrics in split_metrics['holdout_calibration'].items():
        print(
            f"Holdout calibration error @ {horizon} months: "
            f"{metrics['weighted_absolute_calibration_error']:.3f}"
        )

    final_model = train_final_model(encoded_df)
    print('Model trained successfully on full dataset. Saving...')

    model_data = {
        'model': final_model,
        'columns': feature_frame(encoded_df).columns.tolist(),
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'duration_col': DURATION_COL,
        'event_col': EVENT_COL,
        'evaluation': {
            'cross_validated_c_index': cv_metrics,
            'train_holdout': split_metrics,
        },
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

    report_path = MODEL_PATH.with_name('model_evaluation_summary.json')
    report_path.write_text(json.dumps(model_data['evaluation'], indent=2))
    print(f'Evaluation summary saved to {report_path}')


if __name__ == '__main__':
    main()
