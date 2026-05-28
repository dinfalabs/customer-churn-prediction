import pytest
import pandas as pd
import joblib
import sys
import os

# Insert src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import prepare_prediction_data

def test_prepare_prediction_data_alignment():
    """Test that customer data is correctly aligned with the model's expected features"""
    # Load what the scaler expects
    scaler = joblib.load('models/best_churn_model_scaler.pkl')
    features = joblib.load('models/best_churn_model_features.pkl')
    template_df = joblib.load('models/best_churn_model_template.pkl')
    
    # Mock customer data from the web form
    customer_data = {
        'gender': 'Male',
        'SeniorCitizen': 'No',
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 12,
        'PhoneService': 'Yes',
        'PaperlessBilling': 'Yes',
        'MonthlyCharges': 65.0,
        'TotalCharges': 1500.0,
        'Contract': 'Month-to-month'
    }
    
    # Run the function
    try:
        X = prepare_prediction_data(customer_data, template_df=template_df, expected_columns=features)
        
        # Check alignment
        assert len(X.columns) == len(features), f"Expected {len(features)} columns, got {len(X.columns)}"
        assert list(X.columns) == features, "Column order mismatch"
        
        # Extract numerical features for the scaler
        scaler_features = list(scaler.feature_names_in_)
        X_numeric = X[scaler_features]
        assert len(X_numeric.columns) == 12, "Scaler expects exactly 12 numeric features"
        
        # This shouldn't raise any ValueError
        scaled_values = scaler.transform(X_numeric)
        assert scaled_values.shape == (1, 12)
        
        print("Integration test passed: Data alignment and scaling workflow works perfectly.")
    except Exception as e:
        pytest.fail(f"Integration pipeline failed: {e}")
