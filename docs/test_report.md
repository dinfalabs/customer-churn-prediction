# Customer Churn Prediction - Debugging & Testing Session

## 🐛 The Problem
During prediction, the machine learning model was receiving an unexpected number of features. The `StandardScaler` was trained on **12 numeric features**, but the UI was passing **36 features** (including the one-hot encoded categorical variables), leading to a dimensional mismatch error:
`ValueError: The feature names should match those that were passed during fit. Feature names unseen at fit time[...]`

## 🛠️ The Fix
We isolated the numerical features inside the Streamlit `app.py` script specifically for scaling, before concatenating them back dynamically.
```python
# Data alignment and feature extraction
X = prepare_prediction_data(customer_data, template_df=template_df, expected_columns=feature_names)

# Scale ONLY the numeric features that the scaler expects
scaler_features = list(scaler.feature_names_in_)
X_numeric = X[scaler_features].copy()
X_scaled_numeric = pd.DataFrame(scaler.transform(X_numeric), columns=scaler_features)

# Combine for final prediction
X_scaled = X.copy()
X_scaled[scaler_features] = X_scaled_numeric

prediction = model.predict(X_scaled)[0]
```

## ✅ Continuous Integration & Testing
We wrote automated test cases via `pytest` to validate feature engineering and data alignment:

### `tests/test_feature_engineering.py`
✔️ Validated `engineer_features()` constructs aggregate features (like `TotalServices`).
✔️ Checked that categorical features appropriately translate into numerical risk mappings (`ContractRisk`).

### `tests/test_app_integration.py`
✔️ End-to-end simulated form submission to verify `prepare_prediction_data()`.
✔️ Confirmed categorical columns successfully match training columns using the saved Template DataFrame.
✔️ Asserted that `scaler_features` perfectly handles its exact 12 expected columns without throwing dimensional mismatch errors.

## 🚀 Results
```
============================= test session starts ==============================
collected 3 items                                                              

tests/test_app_integration.py::test_prepare_prediction_data_alignment PASSED [ 33%]
tests/test_feature_engineering.py::test_engineer_features_creates_total_services PASSED [ 66%]
tests/test_feature_engineering.py::test_contract_risk_mapping PASSED     [100%]

========================= 3 passed in 1.22s =========================
```
