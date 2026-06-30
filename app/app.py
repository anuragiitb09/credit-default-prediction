import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin


# ── Custom classes required to unpickle models/preprocessor.pkl and
#    models/xgboost_best.pkl. These must be defined here (in whatever
#    script is run as __main__) because that's where joblib looks for
#    them when unpickling. ──────────────────────────────────────────

class CalibratedXGB:
    def __init__(self, base_model, scaler):
        self.base_model = base_model
        self.scaler = scaler
        self.estimator = base_model

    def predict_proba(self, X):
        raw_probs = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        cal_probs = self.scaler.predict_proba(raw_probs)
        return cal_probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class EmpLengthEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def parse(val):
            if pd.isnull(val):
                return -1
            val = str(val).strip().lower()
            if '10+' in val:
                return 10
            if '< 1' in val:
                return 0
            digits = ''.join(filter(str.isdigit, val.split('year')[0]))
            return int(digits) if digits else -1

        col = X.iloc[:, 0] if hasattr(X, 'iloc') else pd.Series(X.flatten())
        return col.map(parse).values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(['emp_length'])


# ── App config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Default Risk Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Credit Risk Platform")
st.sidebar.markdown("XGBoost + SHAP + Fairlearn")

page = st.sidebar.radio(
    "Navigate",
    ["Single applicant scorer", "Portfolio batch upload"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

from utils import load_artifacts

@st.cache_resource
def get_artifacts():
    return load_artifacts()

model, base_xgb, preprocessor, meta, explain_meta, explainer = get_artifacts()

with st.sidebar.expander("Model info"):
    st.markdown(f"""
    **Model**: XGBoost (calibrated)
    **AUC-ROC (test)**: {meta['test_metrics']['auc_roc']:.3f}
    **KS statistic**: {meta['test_metrics']['ks']:.1f}
    **Gini**: {meta['test_metrics']['gini']:.3f}
    **Decision threshold**: {meta['decision_threshold']:.2f}
    """)

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Decision threshold override",
    min_value=0.05, max_value=0.95,
    value=float(meta['decision_threshold']),
    step=0.01,
    help="Lower = more conservative (rejects more applicants). Default is the F1-optimal threshold from validation."
)

if page == "Single applicant scorer":
    import page_single
    page_single.render(model, preprocessor, explainer, meta, threshold)
else:
    import page_portfolio
    page_portfolio.render(model, preprocessor, threshold)