# Model Card — Customer Churn Prediction

## Model details
- **Type:** scikit-learn `Pipeline` (feature engineering → impute/scale + one-hot → `LogisticRegression`).
- **Version:** 2.0 · serialized with scikit-learn 1.8.0 (`models/churn_pipeline.pkl`).
- **Selection:** best of Logistic Regression vs Random Forest by 5-fold cross-validated ROC-AUC on the training split.
- **Owner:** Davide Infantino ([@dinfalabs](https://github.com/dinfalabs)).

## Intended use
- **Use:** estimate the probability that a telecom customer will churn, to prioritize **retention outreach**.
- **Users:** retention / CRM teams, analysts.
- **Out of scope:** automated punitive or pricing decisions about individuals without human review; non-telecom domains; legally protected decisions.

## Training data
- **Source:** Telco Customer Churn (IBM sample data, via Kaggle).
- **Size:** 7,043 customers; churn rate ≈ 26.5% (imbalanced, handled with `class_weight`).
- **Split:** 80/20 stratified; the test set is used once for the reported metrics.
- **Features (19):** demographics, account, services and billing fields (see README). `gender` is included as a feature — see *Ethical considerations*.

## Metrics (held-out test set, n = 1,409)
| Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|
| 0.731 | 0.496 | 0.797 | 0.611 | 0.847 | 0.851 |

- **Decision threshold:** 0.5 (default). Tuning to a business cost ratio is recommended.
- Live values are persisted in `models/metadata.json`.

## Limitations
- Trained on a single point-in-time snapshot of one telecom; **no temporal validation or drift handling**.
- Precision ≈ 0.50: roughly half of flagged customers would not actually churn — acceptable for outreach, **not** for punitive actions.
- Performance on populations that differ from the training distribution is unknown.

## Ethical considerations
- `gender` is an input; monitor predictions for **disparate impact** before any real deployment, and consider removing protected attributes.
- The model must not be used for discriminatory pricing or denial of service.
- Every prediction is **explainable** per customer (`src/explain.py`), supporting transparency and contestability.

## Maintenance
- Retrain: `python train_model.py` (regenerates the pipeline, metadata and reports).
- Quality gate: CI fails if test ROC-AUC < 0.78 or recall < 0.45 (`tests/test_model_performance.py`).
