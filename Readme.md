# Credit Scoring Model — Credit Default Prediction

**CodeAlpha Machine Learning Internship — Task 1**

## Problem

Credit-card issuers need to know in advance whether a customer is likely to **default** on their payment next month. Approving credit for someone who defaults causes a financial loss; wrongly restricting a customer who would have paid causes lost business. This project builds a machine-learning classifier that estimates a customer's default risk from their demographic profile and six months of billing/payment history.

## Dataset

**Default of Credit Card Clients** — 30,000 customers from a major bank in Taiwan (April–September 2005), with 23 features covering:
- Demographics: `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- Credit: `LIMIT_BAL`
- Repayment status for the last 6 months: `PAY_0`, `PAY_2`–`PAY_6`
- Bill amounts for the last 6 months: `BILL_AMT1`–`BILL_AMT6`
- Payment amounts for the last 6 months: `PAY_AMT1`–`PAY_AMT6`

**Target:** `default.payment.next.month` (1 = default, 0 = no default). The dataset is imbalanced — about 22% of customers default.

## What's in this project

`Credit_Scoring_Model.ipynb` walks through the full workflow:

1. Data loading, quality checks, and cleanup (consolidating undocumented category codes)
2. Exploratory data analysis (target balance, demographics, repayment history vs. default, correlations)
3. Leakage-safe preprocessing with a scikit-learn `ColumnTransformer` + `Pipeline`
4. An 80/20 stratified train/test split, with the test set untouched until final evaluation
5. A Logistic Regression baseline
6. Comparison of four models: Logistic Regression, Decision Tree, Random Forest, and HistGradientBoostingClassifier
7. Hyperparameter tuning of the best model (Random Forest) with `RandomizedSearchCV` and stratified cross-validation, optimizing ROC-AUC
8. A single, final, honest evaluation on the held-out test set — accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, and precision-recall curve
9. Feature importance and error analysis (false positives vs. false negatives, and why they matter for credit risk)
10. Saving the complete deployment-ready pipeline and a reusable prediction function

## Results

The tuned Random Forest was selected as the final model based on cross-validated ROC-AUC. Its test-set performance is reported in full inside the notebook (Section 12 — Final Model Evaluation), including the classification report, confusion matrix, and ROC/PR curves. Accuracy alone is not a reliable metric here because of class imbalance, so precision, recall, F1, and ROC-AUC are reported together.

The strongest predictors of default were the customer's most recent repayment status (`PAY_0`) and their credit limit (`LIMIT_BAL`), followed by recent bill amounts.

## Files

| File | Description |
|---|---|
| `Credit_Scoring_Model.ipynb` | Full notebook: EDA, modeling, tuning, evaluation, deployment prep |
| `credit_default_model.joblib` | Complete deployment-ready pipeline (preprocessing + trained model) |
| `feature_columns.json` | Feature schema / expected column order for prediction |
| `default_of_credit_card_clients.csv` | Source dataset |
| `README.md` | This file |

## Using the saved model

```python
import joblib
import pandas as pd

model = joblib.load("credit_default_model.joblib")

def predict(customer_dict):
    df = pd.DataFrame([customer_dict])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {"prediction": int(prediction), "probability": float(probability)}
```

This can be wrapped in a **Streamlit** app, a **Flask** route, or a **FastAPI** endpoint for deployment — the notebook's final sections show a minimal example.

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib.

## Disclaimer

This model is for educational/demonstration purposes as part of a machine learning internship task. It is not a production credit-decisioning system and should not be used to make real lending decisions without further validation, fairness auditing, and regulatory review.
