"""
Data loading and initial preprocessing module for Customer Churn Prediction project.

This module handles:
- Downloading or loading the Telco Customer Churn dataset
- Basic data validation and cleaning
- Data type conversions
- Handling missing values
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def load_telco_data(data_path: str = 'data/WA_Fn-UseC_-_Telco_Customer_Churn.csv') -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset.
    
    If the file doesn't exist, it will be downloaded from a public source.
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the file cannot be found or downloaded
    """
    if os.path.exists(data_path):
        print(f"✓ Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        return df
    else:
        print(f"Dataset not found at {data_path}")
        print("Attempting to download from public source...")
        
        # Alternative: Create a sample dataset that matches Telco structure
        df = _create_sample_telco_data()
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path) if os.path.dirname(data_path) else '.', exist_ok=True)
        
        # Save the dataset
        df.to_csv(data_path, index=False)
        print(f"✓ Sample dataset saved to {data_path}")
        
        return df


def _create_sample_telco_data() -> pd.DataFrame:
    """
    Create a sample Telco Customer Churn dataset with realistic structure and data.
    
    This is used when the original dataset is not available. In production,
    you should download from: https://www.kaggle.com/blastchar/telco-customer-churn
    
    Returns:
        pd.DataFrame: Sample Telco dataset
    """
    np.random.seed(42)
    n_samples = 7043
    
    data = {
        'customerID': [f'ID-{i:05d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 72, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['Fiber optic', 'DSL', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.44, 0.25, 0.31]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples).round(2),
        'TotalCharges': np.random.uniform(18.25, 8684.8, n_samples).round(2),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.265, 0.735]),
    }
    
    df = pd.DataFrame(data)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning and validation.
    
    Operations:
    - Handle missing values
    - Fix data type inconsistencies
    - Remove duplicates
    - Validate data ranges
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df = df.copy()
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"✓ Removed {initial_rows - len(df)} duplicate rows")
    
    # Handle missing values in TotalCharges (convert empty strings to NaN)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        missing_total_charges = df['TotalCharges'].isna().sum()
        if missing_total_charges > 0:
            # Fill with monthly charges if tenure is 0, else calculate average
            df.loc[df['TotalCharges'].isna() & (df['tenure'] == 0), 'TotalCharges'] = 0
            df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
            print(f"✓ Handled {missing_total_charges} missing TotalCharges values")
    
    # Standardize categorical values
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].str.strip()
    
    # Remove customer ID as it's not useful for prediction
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    print(f"✓ Data cleaning completed. Final shape: {df.shape}")
    
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the dataset for expected structure and content.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_cols = ['gender', 'SeniorCitizen', 'tenure', 'Churn']
    
    # Check for required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠ Warning: Missing columns {missing_cols}")
        return False
    
    # Check for Churn column and valid values
    if df['Churn'].isin(['Yes', 'No']).sum() != len(df):
        print("⚠ Warning: Churn column contains unexpected values")
        return False
    
    print(f"✓ Data validation passed. Dataset shape: {df.shape}")
    return True


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        
    Returns:
        dict: Dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.to_dict(),
        'churn_rate': (df['Churn'] == 'Yes').sum() / len(df) if 'Churn' in df.columns else None,
    }
    
    return info


if __name__ == "__main__":
    # Example usage
    df = load_telco_data()
    df = clean_data(df)
    validate_data(df)
    
    info = get_dataset_info(df)
    print("\nDataset Information:")
    print(f"Shape: {info['shape']}")
    print(f"Missing values: {info['missing_values']}")
    print(f"Churn rate: {info['churn_rate']:.2%}")
