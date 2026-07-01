# Credit Default Prediction with Explainability & Fairness Auditing

An end-to-end credit risk system that predicts loan default probability,
explains every decision with SHAP, and audits the model for geographic
fairness — built to mirror the model governance standards used in bank
credit risk teams.

[**Live demo**](https://credit-default-prediction-anurag.streamlit.app) · [Fairness report](reports/fairness_section.md) · [Explainability report](reports/explainability_section.md)

---

## Overview

Banks and lenders need credit scoring models that are accurate, explainable,
and demonstrably fair across borrower demographics — not just high-AUC
black boxes. This project builds that full pipeline on LendingClub's public
loan data (2007–2018, 2.2M originated loans):

- A calibrated **XGBoost** classifier tuned via **Optuna** (50-trial Bayesian search)
- **SHAP** explainability at both the global and per-applicant level, aligned with ECOA adverse-action notice requirements
- A **Fairlearn** fairness audit across US Census regions, with an `ExponentiatedGradient`-constrained model variant and a quantified accuracy-fairness tradeoff curve
- A live **Streamlit** dashboard (single-applicant scorer + portfolio batch scorer) and a **FastAPI** + **Docker** backend for programmatic scoring

---

## Key results

| Metric | Value |
|---|---|
| AUC-ROC (test, 2017–2018) | **0.708** |
| PR-AUC (test) | **0.376** |
| KS statistic | **30.5** |
| Gini coefficient | **0.416** |
| Decision threshold | **0.49** |
| Demographic parity difference (unconstrained) | **0.014** |
| Demographic parity ratio (unconstrained) | **0.749** |
| EEOC four-fifths rule (≥ 0.80) | Fails unconstrained → passes after ExponentiatedGradient mitigation |

Evaluation uses a strict **temporal train/val/test split** (train: 2007–2015,
val: 2016, test: 2017–2018) rather than a random split, to avoid the future
information leakage common in naive treatments of this dataset. Metrics are
intentionally conservative — the temporal split produces lower AUC than
a random split on the same data, which is the correct approach for a
production credit model.

---

## Architecture

```
Raw LendingClub CSV (2.2M loans)
        │
        ▼
  Leakage audit + target definition  ──►  data/processed/loans_clean.parquet
        │
        ▼
  sklearn ColumnTransformer pipeline       (log transform, ordinal/target/
  (preprocessing)                           one-hot encoding, custom parsers)
        │
        ▼
  XGBoost + Optuna (50 trials)         ──►  models/xgboost_best.pkl
  + Platt calibration
        │
        ├──►  SHAP TreeExplainer       ──►  reports/shap/  (global + per-applicant)
        │
        ├──►  Fairlearn audit          ──►  reports/fairness/  (tradeoff curve)
        │
        └──►  Serving layer
                  ├── Streamlit dashboard (app/app.py)      — human-facing UI
                  └── FastAPI + Docker (app/main.py)        — programmatic API
```

Full pipeline code: [`notebooks/`](notebooks/) (01–07, run in order) and reusable modules in [`app/`](app/).

---

## Stack

`XGBoost` · `SHAP` · `Fairlearn` · `Optuna` · `scikit-learn` · `Streamlit` · `FastAPI` · `Docker` · `pandas` · `cloudpickle`

---

## Run locally

### Option A — Streamlit dashboard

```bash
conda create -n credit-default python=3.10 -y
conda activate credit-default
pip install -r requirements.txt

cd app
streamlit run app.py
```

Opens at `http://localhost:8501`. Two pages: single-applicant scorer with an
embedded SHAP waterfall, and a portfolio batch CSV scorer with expected loss
calculation.

### Option B — FastAPI via Docker

```bash
docker build -t credit-default-api .
docker run -p 8000:8000 credit-default-api
```

Test it:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000, "int_rate": 13.5, "grade": "C", "sub_grade": "C3",
    "emp_length": "10+ years", "home_ownership": "MORTGAGE", "annual_inc": 65000,
    "dti": 18.0, "fico_range_low": 695, "revol_util": 45.0, "open_acc": 8,
    "purpose": "debt_consolidation", "addr_state": "CA",
    "earliest_cr_line": "2010-06-01"
  }'
```

Interactive API docs at `http://localhost:8000/docs`.

---

## Project structure

```
├── data/                   # raw + processed datasets (gitignored, regenerate via notebooks)
├── notebooks/              # 01_eda → 07_fairness_audit, run in order
├── app/
│   ├── app.py              # Streamlit entry point
│   ├── page_single.py      # single-applicant scorer + SHAP waterfall
│   ├── page_portfolio.py   # batch CSV scorer
│   ├── main.py             # FastAPI app — /predict, /health
│   ├── schemas.py          # Pydantic request/response models
│   └── utils.py            # shared model loading + inference logic
├── models/                 # fitted preprocessor.pkl, xgboost_best.pkl, metadata
├── reports/
│   ├── shap/               # beeswarm, dependence plots, waterfall cases
│   ├── fairness/           # accuracy-fairness tradeoff, region breakdowns
│   └── fairness_section.md
├── Dockerfile
└── requirements.txt
```

---

## Methodology notes

**Leakage prevention.** All post-origination columns (payment history,
recoveries, last credit pull date) are identified and dropped before
modeling — see the leakage audit in `notebooks/02_data_loading.ipynb`.
Only the 32 features known at loan application time are used.

**Why temporal split, not random split.** Credit models deployed in
production score applicants who haven't been seen yet. A random train/test
split leaks future macroeconomic conditions into training and overstates
performance. This project trains on 2007–2015, validates on 2016, and
tests on 2017–2018 exclusively. The resulting AUC is lower than a random
split would produce — this is intentional and correct.

**Calibration.** Raw XGBoost probabilities are Platt-scaled so the output
can be used directly as a risk score, not just a binary classification.

**Fairness.** Borrowers are grouped by US Census region (a geographic
proxy, since the dataset has no direct demographic attributes). The
unconstrained model fails the EEOC four-fifths rule (DP ratio: 0.749).
Applying `ExponentiatedGradient` with a `DemographicParity` constraint
brings the model into compliance, with the accuracy cost quantified across
an epsilon sweep in `reports/fairness/accuracy_fairness_tradeoff.png`.

**Model serialization.** Both the XGBoost model and preprocessing pipeline
are serialized with `cloudpickle` rather than `joblib`, ensuring custom
transformer classes are embedded directly in the artifact and loadable
without environment-specific class definitions.

---

## Fairness and explainability reports

- [`reports/fairness_section.md`](reports/fairness_section.md) — full fairness audit with ECOA/EEOC framing
- [`reports/explainability_section.md`](reports/explainability_section.md) — SHAP methodology and key findings

---

## Disclaimer

Built on public LendingClub data for portfolio and research purposes. Not
intended for production lending decisions without further validation,
expanded protected-attribute coverage, and independent regulatory review.