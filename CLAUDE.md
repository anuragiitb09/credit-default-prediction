# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

End-to-end credit risk ML system that predicts loan default probability on LendingClub
public loan data (2007-2018, 2.2M originated loans), with SHAP explainability and a
Fairlearn fairness audit layered on top — built to mirror bank credit-risk model
governance practices (leakage audits, temporal validation, ECOA/EEOC-aligned reporting).
Python 3.10, MIT licensed. Live demo: https://credit-default-prediction-anurag.streamlit.app

Two git remotes exist: `origin` (GitHub) and `huggingface` (a Hugging Face Space) — check
`git remote -v` before assuming a single push destination.

Related planning docs (historical/roadmap reference, not code) live one level up:
`../credit_default_project_plan.html` (phased 6-7 week plan) and
`../credit_default_30day_plan.html` (daily execution plan).

## Setup & run

    conda create -n credit-default python=3.10 -y && conda activate credit-default
    pip install -r requirements.txt
    pip install optuna fastapi uvicorn lime   # used in code, NOT in requirements.txt — see Gotchas

Streamlit dashboard (primary UI):

    cd app && streamlit run app.py       # localhost:8501

FastAPI (programmatic API):

    cd app && uvicorn main:app --reload  # localhost:8000, interactive docs at /docs

There is no Dockerfile despite the README documenting a `docker build` workflow — don't
go looking for one; build/run locally, or write the Dockerfile if the user asks for it.

## Architecture / pipeline

    Raw LendingClub CSV (2.2M loans, data/raw/, gitignored)
      -> leakage audit + target def (notebook 02) -> data/processed/loans_clean.parquet
      -> sklearn ColumnTransformer preprocessing (notebook 04) -> models/preprocessor.pkl
      -> XGBoost + Optuna 50-trial HPO + Platt calibration (notebook 05) -> models/xgboost_best.pkl
      -> SHAP TreeExplainer (notebook 06) -> reports/shap/
      -> Fairlearn audit + ExponentiatedGradient mitigation (notebook 07) -> reports/fairness/
      -> serving: Streamlit dashboard (app/app.py) and FastAPI (app/main.py),
         both reconstructing the calibrated model from models/xgboost_model.json +
         models/platt_scaler.pkl at startup

Notebooks are the pipeline (run 01->07 in order to regenerate everything from raw data);
`app/` is the serving layer that consumes the notebooks' output artifacts. Separate
concerns — don't expect to find preprocessing/training logic inside `app/`.

## Directory structure — what's real vs dead

- `app/` — **the actual serving code.** `app.py` (Streamlit entry), `main.py` (FastAPI
  entry), `page_single.py` / `page_portfolio.py` (Streamlit pages), `schemas.py`
  (Pydantic v2 request/response models), `utils.py` (shared inference helpers:
  `load_artifacts`, `score_applicant`, `get_shap_explanation`, `expected_loss`, etc.)
- `notebooks/` — **the actual ML pipeline**, meant to run in order:
  01 (trivial import sanity check) -> 02 (load/clean/leakage-audit/temporal split) ->
  03 (deep EDA) -> 04 (preprocessing pipeline) -> 05 (XGBoost + Optuna + calibration) ->
  06 (SHAP explainability + LIME cross-check) -> 07 (Fairlearn fairness audit).
- `src/` — **dead scaffold, ignore it.** Only empty `__init__.py` files under
  `src/`, `src/features/`, `src/models/`, `src/evaluation/`. Nothing imports from
  `src`. Don't look here for pipeline logic, and don't wire code into it "to clean up"
  unless the user asks for a real refactor into a package structure.
- `models/` — pickled/JSON artifacts. Only `models/xgboost_fair_constrained.pkl` is
  gitignored and absent on a fresh clone (rerun notebook 07 to regenerate it); everything
  else (`xgboost_best.pkl`, `preprocessor.pkl`, `platt_scaler.pkl`, `xgboost_model.json`,
  metadata JSONs) is actually committed despite the `models/*.pkl` gitignore rule, because
  they were tracked before that rule was added.
- `data/raw/`, `data/processed/` — gitignored (raw CSV is ~450MB). Absent on a fresh
  clone; regenerate via notebooks 02/04.
- `reports/` — generated plots, `reports/shap/` and `reports/fairness/` subfolders, and
  two narrative writeups (`explainability_section.md`, `fairness_section.md`) meant for
  a model card.

## Key results (so you don't have to re-derive them)

- Test AUC-ROC 0.708, PR-AUC 0.376, KS 30.5, Gini 0.416, decision threshold 0.49
  (F1-optimized on validation). Evaluated on a temporal split (train 2007-2015,
  val 2016, test 2017-2018) — intentionally more conservative than a random split.
- These fall short of the project's own stated targets (AUC >=0.80, KS >=40,
  PR-AUC >=0.65) — documented honestly in the notebooks, not hidden. Don't "fix" this
  by silently changing the eval methodology.
- Fairness: unconstrained model fails the EEOC four-fifths rule (demographic parity
  ratio 0.749 < 0.80 threshold) across US Census regions (a proxy protected attribute,
  since LendingClub has no direct demographic fields). `ExponentiatedGradient`
  mitigation reaches compliance at ~14pp accuracy cost (`xgboost_fair_constrained.pkl`).
- SHAP/LIME concordance is only 13% (well below the 70% target) — a documented, known
  limitation, not a bug to silently patch over.
- Top SHAP features: sub_grade, grade, loan_amnt, annual_inc, int_rate.

Full hyperparameters and per-fold metrics live in `models/*_metadata.json` and
`models/*_summary.json` — read those directly rather than expecting this file to
enumerate them.

## Gotchas

- **`CalibratedXGB` is intentionally defined three times** (`app/utils.py`, `app/app.py`,
  `app/main.py`) instead of imported once. Required so `pickle`/`cloudpickle` can resolve
  the class when unpickling in each entrypoint's own module namespace. Don't
  "deduplicate" this into a shared import — it will likely break unpickling.
- **`requirements.txt` is incomplete.** `optuna`, `fastapi`, `uvicorn`, and `lime` are all
  used in the code/notebooks but not listed. A fresh `pip install -r requirements.txt`
  is not enough to run the full pipeline or serve the API.
- **`src/` is dead** — see Directory structure above. Don't waste time searching it.
- **No tests, no CI.** No pytest, no test directory, no `.github/workflows`. Verify
  behavior manually (run Streamlit/FastAPI locally) when making changes.
- **No Dockerfile**, despite the README documenting a `docker build`/`docker run` workflow.
- **`xgboost_fair_constrained.pkl` is the only model artifact not committed to git**
  (everything else in `models/` is tracked despite the `*.pkl` gitignore rule — see
  Directory structure). Rerun notebook 07 to regenerate it if it's missing locally.
- FastAPI CORS in `app/main.py` currently allows all origins — fine for local dev,
  flag it if asked about productionizing.

## Conventions observed in this codebase

- snake_case functions/variables, PascalCase classes, UPPER_SNAKE_CASE constants
  (e.g. `REFERENCE_DATE`, `LGD`).
- Type hints are inconsistent — present on some signatures, absent on others. Match
  the style of the function you're editing rather than imposing hints everywhere.
- Docstrings are minimal throughout; match that terseness rather than adding verbose
  docstrings to every function.
- Pydantic v2 is used for all API-boundary validation (`app/schemas.py`).
- Notebooks use `print()` for logging; `app/main.py` uses the `logging` module. Keep
  that split rather than introducing a logging framework into notebooks.
