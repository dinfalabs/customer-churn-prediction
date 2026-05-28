"""
Main model training script.

Run this script to train models and save the best one.

Usage:
    python train_model.py
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_telco_data, clean_data, validate_data
from src.feature_engineering import (
    separate_features_and_target, encode_categorical_features, 
    engineer_features, scale_features, get_feature_importance_dataframe
)
from src.model_utils import (
    train_logistic_regression, train_random_forest, evaluate_model, 
    compare_models, select_best_model, save_model
)


def main():
    """
    Main training pipeline.
    """
    print("\n" + "="*80)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING PIPELINE")
    print("="*80)
    
    # 1. Load data
    print("\n[1/8] Loading dataset...")
    df = load_telco_data('data/WA_Fn-UseC_-_Telco_Customer_Churn.csv')
    
    # 2. Clean data
    print("\n[2/8] Cleaning data...")
    df = clean_data(df)
    validate_data(df)
    
    # 3. Feature preparation
    print("\n[3/8] Preparing features...")
    X, y = separate_features_and_target(df)
    
    # 4. Feature engineering
    print("\n[4/8] Engineering features...")
    X, _ = engineer_features(X, None)
    
    # 5. Train-test split
    print("\n[5/8] Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    - Training samples: {len(X_train)}")
    print(f"    - Testing samples: {len(X_test)}")
    
    # 6. Encode and scale
    print("\n[6/8] Encoding categorical features and scaling...")
    X_train_encoded, X_test_encoded, _ = encode_categorical_features(X_train, X_test)
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_encoded, X_test_encoded
    )
    
    feature_names = X_train_scaled.columns.tolist()
    
    # 7. Train models
    print("\n[7/8] Training models...")
    
    # Logistic Regression
    print("\n    Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    lr_results = evaluate_model(
        lr_model, X_train_scaled, y_train, X_test_scaled, y_test, 
        model_name='Logistic Regression'
    )
    
    # Random Forest
    print("\n    Training Random Forest...")
    rf_model = train_random_forest(X_train_scaled, y_train, n_estimators=100)
    rf_results = evaluate_model(
        rf_model, X_train_scaled, y_train, X_test_scaled, y_test, 
        model_name='Random Forest'
    )
    
    # 8. Compare and select best model
    print("\n[8/8] Comparing models and selecting best...")
    models_results = {
        'Logistic Regression': lr_results,
        'Random Forest': rf_results
    }
    
    comparison_df = compare_models(models_results)
    best_model, best_model_name = select_best_model(models_results, metric='f1')
    
    # Save best model
    model_path = 'models/best_churn_model.pkl'
    save_model(best_model, model_path, scaler=scaler, feature_names=feature_names)
    
    # Save a template DataFrame for prediction (with ALL possible columns from one-hot encoding)
    import joblib
    from sklearn.preprocessing import LabelEncoder
    
    X_template = X_train.copy()
    categorical_cols = X_template.select_dtypes(include=['object']).columns.tolist()
    
    # Binary categorical features - label encode
    binary_cols = [col for col in categorical_cols if X_template[col].nunique() == 2]
    for col in binary_cols:
        le = LabelEncoder()
        X_template[col] = le.fit_transform(X_template[col])
    
    # Multi-class categorical features - one-hot encode (drop_first=False for ALL columns)
    multiclass_cols = [col for col in categorical_cols if col not in binary_cols]
    if multiclass_cols:
        X_template = pd.get_dummies(X_template, columns=multiclass_cols, drop_first=False)
    
    # Save as empty template (0 rows, but correct columns and types)
    template_df = X_template.iloc[:0].copy()
    joblib.dump(template_df, 'models/best_churn_model_template.pkl')
    
    # Save comparison
    comparison_df.to_csv('reports/model_comparison.csv', index=False)
    
    # Save feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance_df = get_feature_importance_dataframe(
            feature_names, best_model.feature_importances_, top_n=15
        )
        importance_df.to_csv('reports/feature_importance.csv', index=False)
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test F1-Score: {best_model_name}: {best_model_name}: {lr_results['test_metrics']['f1']:.4f}")
    print(f"\nModel saved to: {model_path}")
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app.py")
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
