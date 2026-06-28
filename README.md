# Credit Default Prediction with Explainability

End-to-end ML system for credit default prediction using LendingClub
loan data (2007–2018, 2.2M records), with SHAP-based explainability,
fairness auditing (Fairlearn), and a live lender dashboard.

## Target metrics
| Metric | Target | Achieved |
|---|---|---|
| AUC-ROC | ≥ 0.80 | TBD |
| PR-AUC | ≥ 0.65 | TBD |
| KS Statistic | ≥ 40 | TBD |
| Gini Coefficient | ≥ 0.60 | TBD |

## Live demo
[Streamlit app — deploying Week 4]

## Stack
XGBoost · SHAP · Fairlearn · Optuna · Streamlit · FastAPI · Docker

## Structure
- `notebooks/` — EDA, modeling, SHAP, fairness audit notebooks
- `src/` — reusable pipeline modules
- `app/` — Streamlit dashboard + FastAPI backend
- `reports/` — SHAP figures, fairness report, model card

## Run locally
conda activate credit-default
streamlit run app/app.py

## Model card
See reports/model_card.md