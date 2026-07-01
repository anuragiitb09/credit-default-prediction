import cloudpickle
import json
import numpy as np
import pandas as pd
import shap
from pathlib import Path

APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR   = PROJECT_ROOT / 'models'

REFERENCE_DATE = pd.Timestamp('2018-12-31')
LGD = 0.45


def load_artifacts():
    with open(MODELS_DIR / 'xgboost_best.pkl', 'rb') as f:
        model = cloudpickle.load(f)

    with open(MODELS_DIR / 'preprocessor.pkl', 'rb') as f:
        preprocessor = cloudpickle.load(f)

    with open(MODELS_DIR / 'xgboost_metadata.json') as f:
        meta = json.load(f)

    with open(MODELS_DIR / 'explainability_summary.json') as f:
        explain_meta = json.load(f)

    base_xgb = model.estimator
    explainer = shap.TreeExplainer(base_xgb, feature_perturbation='tree_path_dependent')

    return model, base_xgb, preprocessor, meta, explain_meta, explainer


def build_raw_input_df(inputs: dict) -> pd.DataFrame:
    row = {
        'loan_amnt'        : inputs['loan_amnt'],
        'int_rate'         : inputs['int_rate'],
        'grade'            : inputs['grade'],
        'sub_grade'        : inputs['sub_grade'],
        'emp_length'       : inputs['emp_length'],
        'home_ownership'   : inputs['home_ownership'],
        'annual_inc'       : inputs['annual_inc'],
        'dti'              : inputs['dti'],
        'fico_range_low'   : inputs['fico_range_low'],
        'revol_util'       : inputs['revol_util'],
        'open_acc'         : inputs['open_acc'],
        'purpose'          : inputs['purpose'],
        'addr_state'       : inputs['addr_state'],
        'earliest_cr_line' : inputs['earliest_cr_line'],
    }
    df = pd.DataFrame([row])
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
    df['credit_age_months'] = (
        (REFERENCE_DATE - df['earliest_cr_line']).dt.days / 30.44
    ).round(0).astype('float32')
    df = df.drop(columns=['earliest_cr_line'])
    return df


def score_applicant(raw_df: pd.DataFrame, model, preprocessor, threshold: float):
    X_processed = preprocessor.transform(raw_df)
    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    prob = model.predict_proba(X_df)[:, 1][0]
    decision = 'DECLINE' if prob >= threshold else 'APPROVE'
    risk_score = int(round((1 - prob) * 100))
    return prob, decision, risk_score, X_df


def get_shap_explanation(X_df, explainer, feature_names_clean):
    shap_values = explainer(X_df)
    shap_values.feature_names = feature_names_clean
    return shap_values


def clean_feature_name(name: str) -> str:
    prefixes = [
        'log_features__', 'num_features__', 'ohe_features__',
        'grade__', 'subgrade__', 'state_target__', 'emp_length__'
    ]
    for p in prefixes:
        name = name.replace(p, '')
    name = name.replace('home_ownership_', 'ownership: ')
    name = name.replace('purpose_', 'purpose: ')
    if 'annual_inc' in name:
        name = name + ' (log)' if '(log)' not in name else name
    return name


def expected_loss(prob_default: float, loan_amnt: float, lgd: float = LGD) -> float:
    return prob_default * lgd * loan_amnt


def batch_score(df_raw: pd.DataFrame, model, preprocessor, threshold: float):
    df = df_raw.copy()
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
    df['credit_age_months'] = (
        (REFERENCE_DATE - df['earliest_cr_line']).dt.days / 30.44
    ).round(0).astype('float32')
    df_features = df.drop(columns=['earliest_cr_line'])
    X_processed = preprocessor.transform(df_features)
    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    probs = model.predict_proba(X_df)[:, 1]
    decisions = np.where(probs >= threshold, 'DECLINE', 'APPROVE')
    risk_scores = np.round((1 - probs) * 100).astype(int)
    el = probs * LGD * df['loan_amnt'].values
    result = df_raw.copy()
    result['default_probability'] = probs.round(4)
    result['risk_score']          = risk_scores
    result['decision']            = decisions
    result['expected_loss']       = el.round(2)
    return result.sort_values('default_probability', ascending=False).reset_index(drop=True)