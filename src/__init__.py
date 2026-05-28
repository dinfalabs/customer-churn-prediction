"""
Customer Churn Prediction - src package

This package contains all the modules needed for the customer churn prediction project:
- data_loader: Data loading and preprocessing
- feature_engineering: Feature creation and transformation
- model_utils: Model training and evaluation
- config: Configuration settings
- utils: Utility functions
"""

from .data_loader import load_telco_data, clean_data, validate_data, get_dataset_info
from .feature_engineering import (separate_features_and_target, encode_categorical_features, 
                                 engineer_features, scale_features, get_feature_importance_dataframe)
from .model_utils import (train_logistic_regression, train_random_forest, evaluate_model, 
                         compare_models, select_best_model, save_model, load_model, 
                         cross_validate_model, ModelEvaluator)
from . import config
from . import utils

__version__ = "1.0.0"
__author__ = "Data Science Team"
__description__ = "Customer Churn Prediction - A comprehensive ML project"

__all__ = [
    # Data loading
    'load_telco_data',
    'clean_data',
    'validate_data',
    'get_dataset_info',
    # Feature engineering
    'separate_features_and_target',
    'encode_categorical_features',
    'engineer_features',
    'scale_features',
    'get_feature_importance_dataframe',
    # Model training and evaluation
    'train_logistic_regression',
    'train_random_forest',
    'evaluate_model',
    'compare_models',
    'select_best_model',
    'save_model',
    'load_model',
    'cross_validate_model',
    'ModelEvaluator',
    # Modules
    'config',
    'utils',
]
