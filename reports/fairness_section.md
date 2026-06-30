## Fairness audit

### Protected attribute
Borrowers grouped by US Census region (Northeast, Midwest, South, West),
derived from `addr_state`, used as a geographic proxy in the absence of
direct demographic data in the LendingClub dataset.

### Unconstrained model — fairness metrics
| Metric | Value | Interpretation |
|---|---|---|
| Demographic parity difference | 0.0137 | 0 = equal approval rates across regions |
| Demographic parity ratio | 0.7485 | Fails EEOC four-fifths rule (≥0.80) |
| Equalized odds difference | 0.0227 | 0 = equal TPR/FPR across regions |
| Predictive parity difference | 0.0241 | 0 = equal precision across regions |

### Mitigation
Applied Fairlearn's `ExponentiatedGradient` with a `DemographicParity`
constraint (eps=0.02). This reduces demographic parity difference from
0.0137 to 0.0314, at a cost of
14.17 percentage points
of accuracy (0.7851 → 0.6433).

### Accuracy–fairness tradeoff
The tradeoff curve (see `reports/fairness/accuracy_fairness_tradeoff.png`) shows
a smooth, near-linear cost of fairness across the epsilon sweep — there is no
single eps value that achieves both perfect fairness and zero accuracy cost,
consistent with established fairness-accuracy tradeoff theory.

### Regulatory framing
Findings are evaluated against the Equal Credit Opportunity Act (ECOA) and
Fair Housing Act standards. The four-fifths rule (selection rate ratio ≥ 0.80
between any two groups) is the primary regulatory threshold referenced.
The constrained model variant (xgboost_fair_constrained.pkl) is provided as an
audit-ready alternative when regulatory fairness constraints take precedence
over marginal accuracy gains.