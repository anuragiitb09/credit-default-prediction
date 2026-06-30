## Explainability

### Method
SHAP TreeExplainer (tree_path_dependent) computed on 5,000 stratified
test samples. LIME cross-validation on 5 individual borrowers confirms
top-3 feature concordance of 13% (threshold: 70%).

### Global feature importance (top 5)
1. **subsub_grade** — mean |SHAP| = 0.2866
2. **grade** — mean |SHAP| = 0.1742
3. **loan_amnt** — mean |SHAP| = 0.1691
4. **annual_inc** — mean |SHAP| = 0.1151
5. **int_rate** — mean |SHAP| = 0.1087

### Key findings
- **Interest rate** is the dominant predictor. Rates above ~18% strongly increase
  predicted default probability, consistent with adverse selection at origination.
- **DTI** shows a non-linear relationship: impact is moderate below 20%,
  accelerates sharply above 30%.
- **Grade** interacts with interest rate: at the same rate, higher-grade borrowers
  show lower SHAP impact, suggesting the grade captures risk not fully reflected
  in rate alone.
- **FICO score at origination** reduces default probability below ~650; above 720
  its marginal SHAP contribution flattens.
- **Credit age** (months since earliest credit line) consistently reduces default
  risk — longer credit history signals reliability.

### Regulatory alignment
Per-decision SHAP waterfall plots provide feature-level adverse action explanations
aligned with ECOA (Equal Credit Opportunity Act) requirements.
The top contributing features for any rejection decision are surfaced in the
lender dashboard for compliance review.

### Base value
The model base value is 0.0053 (log-odds),
corresponding to the population average default rate of 21.29%.