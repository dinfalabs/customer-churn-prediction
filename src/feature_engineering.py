"""
Feature engineering module for Customer Churn Prediction project.

This module handles:
- Encoding categorical variables
- Feature scaling
- Feature creation and transformation
- Feature selection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings('ignore')


def separate_features_and_target(df: pd.DataFrame, target_col: str = 'Churn') -> tuple:
    """
    Separate features and target variable.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column
        
    Returns:
        tuple: (X features dataframe, y target series)
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col].map({'Yes': 1, 'No': 0})
    
    return X, y


def encode_categorical_features(X: pd.DataFrame, X_test: pd.DataFrame = None, fit_encoders: bool = True):
    """
    Encode categorical features using appropriate encoding methods.
    
    Args:
        X (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features (optional)
        fit_encoders (bool): Whether to fit new encoders or use existing ones
        
    Returns:
        tuple: (X_encoded, X_test_encoded, encoders_dict)
    """
    X = X.copy()
    if X_test is not None:
        X_test = X_test.copy()
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    encoders_dict = {}
    
    # Binary categorical features - use binary encoding
    binary_cols = [col for col in categorical_cols if X[col].nunique() == 2]
    for col in binary_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        
        if X_test is not None:
            X_test[col] = le.transform(X_test[col])
        
        encoders_dict[col] = le
    
    # Multi-class categorical features - use one-hot encoding
    multiclass_cols = [col for col in categorical_cols if col not in binary_cols]
    if multiclass_cols:
        # One-hot encode
        X = pd.get_dummies(X, columns=multiclass_cols, drop_first=True)
        
        if X_test is not None:
            X_test = pd.get_dummies(X_test, columns=multiclass_cols, drop_first=True)
            
            # Ensure same columns between train and test
            train_cols = set(X.columns)
            test_cols = set(X_test.columns)
            
            # Add missing columns to test set
            for col in train_cols - test_cols:
                X_test[col] = 0
            
            # Remove extra columns from test set
            X_test = X_test[X.columns]
        
        encoders_dict['multiclass'] = multiclass_cols
    
    return X, X_test, encoders_dict


def engineer_features(X: pd.DataFrame, X_test: pd.DataFrame = None) -> tuple:
    """
    Create and engineer new features.
    
    Feature engineering operations:
    - Total services: count of active services
    - Contract risk: based on contract type
    - Charge ratio: monthly vs total charges
    - Tenure segments: categorize tenure into groups
    
    Args:
        X (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features (optional)
        
    Returns:
        tuple: (X_engineered, X_test_engineered)
    """
    X = X.copy()
    if X_test is not None:
        X_test = X_test.copy()
    
    # 1. Count total services
    service_cols = [col for col in X.columns if 'Online' in col or 'Streaming' in col or 'Device' in col or 'Tech' in col]
    if service_cols:
        X['TotalServices'] = (X[service_cols] == 'Yes').sum(axis=1)
        if X_test is not None:
            X_test['TotalServices'] = (X_test[service_cols] == 'Yes').sum(axis=1)
    
    # 2. Contract risk feature
    if 'Contract' in X.columns:
        contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
        X['ContractRisk'] = X['Contract'].map(contract_risk).fillna(2)
        if X_test is not None:
            X_test['ContractRisk'] = X_test['Contract'].map(contract_risk).fillna(2)
    
    # 3. Monthly to total charges ratio
    if 'MonthlyCharges' in X.columns and 'TotalCharges' in X.columns:
        X['ChargeRatio'] = X['MonthlyCharges'] / (X['TotalCharges'] + 1)  # +1 to avoid division by zero
        if X_test is not None:
            X_test['ChargeRatio'] = X_test['MonthlyCharges'] / (X_test['TotalCharges'] + 1)
    
    # 4. Tenure segments
    if 'tenure' in X.columns:
        X['TenureSegment'] = pd.cut(X['tenure'], bins=[0, 12, 24, 36, 72], 
                                     labels=['0-1 year', '1-2 years', '2-3 years', '3+ years'])
        X['TenureSegment'] = X['TenureSegment'].astype(str)
        
        if X_test is not None:
            X_test['TenureSegment'] = pd.cut(X_test['tenure'], bins=[0, 12, 24, 36, 72], 
                                              labels=['0-1 year', '1-2 years', '2-3 years', '3+ years'])
            X_test['TenureSegment'] = X_test['TenureSegment'].astype(str)
    
    return X, X_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
                   numeric_cols: list = None) -> tuple:
    """
    Scale numeric features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features (optional)
        numeric_cols (list): List of numeric columns to scale (if None, auto-detect)
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    X_train = X_train.copy()
    if X_test is not None:
        X_test = X_test.copy()
    
    # Auto-detect numeric columns if not provided
    if numeric_cols is None:
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Filter to only columns that exist in the dataframe
    numeric_cols = [col for col in numeric_cols if col in X_train.columns]
    
    if not numeric_cols:
        return X_train, X_test, None
    
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    
    if X_test is not None:
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, scaler


def get_feature_importance_dataframe(feature_names: list, importance_values: np.ndarray, 
                                    top_n: int = 15) -> pd.DataFrame:
    """
    Create a dataframe of feature importance scores.
    
    Args:
        feature_names (list): List of feature names
        importance_values (np.ndarray): Array of importance values
        top_n (int): Number of top features to return
        
    Returns:
        pd.DataFrame: Feature importance dataframe sorted by importance
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=False)
    
    return importance_df.head(top_n)


if __name__ == "__main__":
    # Example usage
    print("Feature engineering module loaded successfully")
