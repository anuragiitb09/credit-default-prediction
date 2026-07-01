import logging
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import ApplicantInput, PredictionResponse, ShapFeature, HealthResponse
from utils import (
    load_artifacts, build_raw_input_df, score_applicant,
    get_shap_explanation, clean_feature_name, expected_loss
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit-api")


# ── Custom classes required to unpickle models/xgboost_best.pkl and
#    models/preprocessor.pkl. Must be defined here because uvicorn runs
#    this file as the entry point, and joblib looks for these classes
#    in whatever module is currently __main__/__mp_main__. ────────────

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



ARTIFACTS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model, preprocessor, and SHAP explainer once
    logger.info("Loading model artifacts...")
    model, base_xgb, preprocessor, meta, explain_meta, explainer = load_artifacts()
    ARTIFACTS['model'] = model
    ARTIFACTS['base_xgb'] = base_xgb
    ARTIFACTS['preprocessor'] = preprocessor
    ARTIFACTS['meta'] = meta
    ARTIFACTS['explain_meta'] = explain_meta
    ARTIFACTS['explainer'] = explainer
    logger.info(f"Model loaded. Test AUC-ROC: {meta['test_metrics']['auc_roc']:.4f}")
    yield
    # Shutdown: nothing to clean up — artifacts are released with the process
    logger.info("Shutting down.")


app = FastAPI(
    title="Credit Default Prediction API",
    description="XGBoost credit risk scoring with SHAP explainability",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your dashboard's domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health():
    """Liveness/readiness check. Returns model status and key metrics."""
    if not ARTIFACTS:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_version="unknown",
            test_auc_roc=0.0
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version="xgboost_best_v1",
        test_auc_roc=ARTIFACTS['meta']['test_metrics']['auc_roc']
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(applicant: ApplicantInput):
    """
    Score a single credit applicant.
    Returns risk score, default probability, decision, expected loss,
    and the top 3 SHAP features driving the prediction.
    """
    if not ARTIFACTS:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        inputs = applicant.model_dump()
        inputs['earliest_cr_line'] = inputs['earliest_cr_line'].isoformat()

        raw_df = build_raw_input_df(inputs)

        model        = ARTIFACTS['model']
        preprocessor = ARTIFACTS['preprocessor']
        meta         = ARTIFACTS['meta']
        explainer    = ARTIFACTS['explainer']
        threshold    = meta['decision_threshold']

        prob, decision, risk_score, X_processed = score_applicant(
            raw_df, model, preprocessor, threshold
        )

        feature_names_clean = [clean_feature_name(c) for c in X_processed.columns]
        shap_exp = get_shap_explanation(X_processed, explainer, feature_names_clean)

        shap_vals = shap_exp.values[0]
        top3_idx = sorted(
            range(len(shap_vals)),
            key=lambda i: abs(shap_vals[i]),
            reverse=True
        )[:3]

        top3_features = [
            ShapFeature(
                feature=feature_names_clean[i],
                shap_value=round(float(shap_vals[i]), 4),
                direction='increases_risk' if shap_vals[i] > 0 else 'decreases_risk'
            )
            for i in top3_idx
        ]

        el = expected_loss(prob, applicant.loan_amnt)

        return PredictionResponse(
            risk_score=risk_score,
            default_probability=round(float(prob), 4),
            decision=decision,
            decision_threshold=threshold,
            expected_loss=round(el, 2),
            top_shap_features=top3_features
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/", tags=["monitoring"])
def root():
    return {
        "service": "Credit Default Prediction API",
        "docs": "/docs",
        "health": "/health"
    }