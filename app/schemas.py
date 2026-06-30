from pydantic import BaseModel, Field, field_validator
from typing import Literal
from datetime import date


class ApplicantInput(BaseModel):
    """
    Schema for a single credit applicant.
    Matches the 14 fields used in the Streamlit single-applicant scorer.
    """
    loan_amnt: float = Field(..., ge=500, le=40000, description="Requested loan amount in USD")
    int_rate: float = Field(..., ge=5.0, le=31.0, description="Interest rate assigned (%)")
    grade: Literal['A','B','C','D','E','F','G'] = Field(..., description="LendingClub grade")
    sub_grade: str = Field(..., description="LendingClub sub-grade, e.g. 'C3'")
    emp_length: str = Field(..., description="Employment length, e.g. '10+ years', '< 1 year'")
    home_ownership: Literal['RENT', 'MORTGAGE', 'OWN', 'OTHER'] = Field(..., description="Home ownership status")
    annual_inc: float = Field(..., ge=0, le=2_000_000, description="Self-reported annual income (USD)")
    dti: float = Field(..., ge=0, le=60, description="Debt-to-income ratio (%)")
    fico_range_low: int = Field(..., ge=300, le=850, description="FICO score at origination")
    revol_util: float = Field(..., ge=0, le=150, description="Revolving line utilization rate (%)")
    open_acc: int = Field(..., ge=0, le=60, description="Number of open credit accounts")
    purpose: str = Field(..., description="Loan purpose category")
    addr_state: str = Field(..., min_length=2, max_length=2, description="Two-letter US state code")
    earliest_cr_line: date = Field(..., description="Date earliest credit line was opened")

    @field_validator('sub_grade')
    @classmethod
    def validate_sub_grade(cls, v: str, info):
        grade = info.data.get('grade')
        if grade and not v.startswith(grade):
            raise ValueError(f"sub_grade '{v}' must start with grade '{grade}'")
        if len(v) != 2 or not v[1].isdigit() or not (1 <= int(v[1]) <= 5):
            raise ValueError(f"sub_grade must be like 'C3' (grade letter + digit 1-5), got '{v}'")
        return v

    @field_validator('addr_state')
    @classmethod
    def validate_state(cls, v: str):
        return v.upper()

    @field_validator('earliest_cr_line')
    @classmethod
    def validate_credit_line_date(cls, v: date):
        if v.year < 1950:
            raise ValueError("earliest_cr_line year looks invalid (before 1950)")
        if v > date.today():
            raise ValueError("earliest_cr_line cannot be in the future")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 15000,
                "int_rate": 13.5,
                "grade": "C",
                "sub_grade": "C3",
                "emp_length": "10+ years",
                "home_ownership": "MORTGAGE",
                "annual_inc": 65000,
                "dti": 18.0,
                "fico_range_low": 695,
                "revol_util": 45.0,
                "open_acc": 8,
                "purpose": "debt_consolidation",
                "addr_state": "CA",
                "earliest_cr_line": "2010-06-01"
            }
        }


class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    direction: Literal['increases_risk', 'decreases_risk']


class PredictionResponse(BaseModel):
    risk_score: int = Field(..., description="0-100, where 100 = lowest risk")
    default_probability: float = Field(..., description="Predicted probability of default")
    decision: Literal['APPROVE', 'DECLINE']
    decision_threshold: float
    expected_loss: float = Field(..., description="PD x LGD(0.45) x EAD, in USD")
    top_shap_features: list[ShapFeature]

    class Config:
        json_schema_extra = {
            "example": {
                "risk_score": 78,
                "default_probability": 0.0823,
                "decision": "APPROVE",
                "decision_threshold": 0.42,
                "expected_loss": 555.53,
                "top_shap_features": [
                    {"feature": "int_rate", "shap_value": -0.42, "direction": "decreases_risk"},
                    {"feature": "fico_range_low", "shap_value": -0.31, "direction": "decreases_risk"},
                    {"feature": "dti", "shap_value": 0.18, "direction": "increases_risk"}
                ]
            }
        }


class HealthResponse(BaseModel):
    status: Literal['ok', 'degraded']
    model_loaded: bool
    model_version: str
    test_auc_roc: float