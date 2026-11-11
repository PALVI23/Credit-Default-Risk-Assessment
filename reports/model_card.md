# Model card: Home Credit default-risk model

**Model:** LightGBM gradient-boosted trees (5-fold ensemble, Optuna-tuned), followed by Platt calibration and a profit-optimal threshold policy (approve if PD < 0.165).
**Intended use:** An educational and research demonstration of an explainable, fairness-audited credit-scoring pipeline. **Not for real lending decisions.**
**Data:** Home Credit Default Risk (Kaggle, 2018): 307,511 applicants from a consumer lender serving people with thin credit files. The population may not represent other markets.
**Target:** `TARGET = 1` means the client had payment difficulties on the loan (8.07% of applicants).
**Features:** 242 per-applicant features from 7 relational tables. Gender (`CODE_GENDER`) is excluded from the inputs; age is used.

## Performance (holdout, n = 61,503)
| AUC | Gini | KS | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|
| 0.7934 | 0.587 | 0.445 | 0.297 | 0.0651 | 0.0026 |

At the deployed threshold: 87.1% approval; the default rate among approved loans is 5.01% (vs 8.07% overall).

## Fairness (holdout, single threshold)
| Group | Default rate | Approval rate | Creditworthy applicants rejected | AUC |
|---|---|---|---|---|
| Female | 7.0% | 89.0% | 8.6% | 0.789 |
| Male | 10.2% | 83.3% | 12.8% | 0.791 |
| Age < 30 | 11.3% | 77.6% | 18.2% | 0.770 |
| Age 60+ | 5.0% | 95.8% | 3.2% | 0.748 |

* Gender can be predicted from the model's inputs with AUC 0.903 (proxy features), so excluding it does not guarantee gender-neutral outcomes.
* Equal-opportunity post-processing (thresholds F 0.150 / M 0.193) cuts the gender false-rejection gap from 4.2 to 0.5 percentage points for −0.55% profit. It requires the protected attribute at decision time, which may be unlawful in many jurisdictions.

## Limitations
* Trained on approved loans only (selection bias; no reject inference).
* Static snapshot; no drift monitoring.
* The profit model is simplified (fixed 10% margin and 50% LGD, no time value of money).
* SHAP reason codes describe the model's behaviour, not causal effects.
* Lower accuracy for the youngest and oldest applicants (AUC 0.75–0.77 vs about 0.80 for ages 30–49).
