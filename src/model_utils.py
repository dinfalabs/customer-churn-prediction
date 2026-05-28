"""
Model training and evaluation module for Customer Churn Prediction project.

This module handles:
- Model training
- Model evaluation
- Metrics calculation
- Model comparison
- Model serialization
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            roc_curve, auc, classification_report)
import joblib
import os
from typing import Dict, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Class for evaluating classification models."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities (for ROC-AUC)
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            np.ndarray: Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                                    model_name: str = "Model"):
        """
        Print detailed classification report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
        """
        print(f"\n{'='*60}")
        print(f"Classification Report - {model_name}")
        print(f"{'='*60}")
        print(classification_report(y_true, y_pred, 
                                   target_names=['No Churn', 'Churn'],
                                   digits=4))


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, 
                             random_state: int = 42) -> LogisticRegression:
    """
    Train a Logistic Regression model.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        random_state (int): Random state for reproducibility
        
    Returns:
        LogisticRegression: Trained model
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs',
        verbose=0
    )
    
    model.fit(X_train, y_train)
    print("✓ Logistic Regression model trained")
    
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, 
                       n_estimators: int = 100, random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest model.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        n_estimators (int): Number of trees
        random_state (int): Random state for reproducibility
        
    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4
    )
    
    model.fit(X_train, y_train)
    print(f"✓ Random Forest model trained ({n_estimators} trees)")
    
    return model


def evaluate_model(model, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  model_name: str = "Model") -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Testing features
        y_test (np.ndarray): Testing labels
        model_name (str): Name of the model
        
    Returns:
        dict: Evaluation results
    """
    evaluator = ModelEvaluator()
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probability predictions
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)
    
    # Metrics
    train_metrics = evaluator.calculate_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = evaluator.calculate_metrics(y_test, y_test_pred, y_test_proba)
    
    # Confusion matrix
    cm = evaluator.get_confusion_matrix(y_test, y_test_pred)
    
    # Classification report
    evaluator.print_classification_report(y_test, y_test_pred, model_name)
    
    results = {
        'model': model,
        'model_name': model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'confusion_matrix': cm,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba,
    }
    
    return results


def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple models' performance.
    
    Args:
        results_dict (dict): Dictionary with model names as keys and evaluation results as values
        
    Returns:
        pd.DataFrame: Comparison dataframe
    """
    comparison_data = []
    
    for model_name, results in results_dict.items():
        metrics = results['test_metrics']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics.get('roc_auc', np.nan),
        })
    
    comparison_df = pd.DataFrame(comparison_data).sort_values('F1-Score', ascending=False)
    
    print("\n" + "="*80)
    print("Model Comparison Summary")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    return comparison_df


def select_best_model(results_dict: Dict[str, Dict[str, Any]], 
                     metric: str = 'f1') -> Tuple[Any, str]:
    """
    Select the best model based on a specific metric.
    
    Args:
        results_dict (dict): Dictionary with model names and evaluation results
        metric (str): Metric to use for comparison ('f1', 'accuracy', 'roc_auc', etc.)
        
    Returns:
        tuple: (best_model, best_model_name)
    """
    best_score = -1
    best_model = None
    best_model_name = None
    
    for model_name, results in results_dict.items():
        score = results['test_metrics'].get(metric, 0)
        if score > best_score:
            best_score = score
            best_model = results['model']
            best_model_name = model_name
    
    print(f"\n✓ Best model: {best_model_name} ({metric}: {best_score:.4f})")
    
    return best_model, best_model_name


def save_model(model, model_path: str, scaler=None, feature_names: list = None):
    """
    Save trained model and associated objects.
    
    Args:
        model: Trained model
        model_path (str): Path to save the model
        scaler: Scaler object (optional)
        feature_names (list): List of feature names (optional)
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")
    
    if scaler is not None:
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"✓ Scaler saved to {scaler_path}")
    
    if feature_names is not None:
        features_path = model_path.replace('.pkl', '_features.pkl')
        joblib.dump(feature_names, features_path)
        print(f"✓ Feature names saved to {features_path}")


def load_model(model_path: str):
    """
    Load saved model.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        model: Loaded model
    """
    model = joblib.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    return model


def cross_validate_model(model, X: np.ndarray, y: np.ndarray, 
                        cv: int = 5) -> Dict[str, list]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model to validate
        X (np.ndarray): Features
        y (np.ndarray): Labels
        cv (int): Number of folds
        
    Returns:
        dict: Cross-validation scores
    """
    metrics = {
        'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
        'precision': cross_val_score(model, X, y, cv=cv, scoring='precision'),
        'recall': cross_val_score(model, X, y, cv=cv, scoring='recall'),
        'f1': cross_val_score(model, X, y, cv=cv, scoring='f1'),
    }
    
    print(f"\nCross-Validation Results (CV={cv}):")
    for metric_name, scores in metrics.items():
        print(f"  {metric_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return metrics


if __name__ == "__main__":
    print("Model training and evaluation module loaded successfully")
