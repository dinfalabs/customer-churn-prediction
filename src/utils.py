"""
Utility functions for the Customer Churn Prediction project.

This module contains helper functions for common tasks like data validation,
logging, and utility operations.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: str = None, log_level: str = 'INFO') -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file (str): Path to log file (optional)
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


logger = setup_logging()


# ============================================================================
# DATA VALIDATION FUNCTIONS
# ============================================================================

def validate_dataframe(df: pd.DataFrame, 
                      min_rows: int = 1, 
                      required_columns: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        min_rows (int): Minimum required rows
        required_columns (list): Required column names
        
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    # Check if DataFrame is empty
    if df.empty:
        errors.append("DataFrame is empty")
        return False, errors
    
    # Check minimum rows
    if len(df) < min_rows:
        errors.append(f"DataFrame has {len(df)} rows, requires minimum {min_rows}")
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for all NaN columns
    nan_cols = df.columns[df.isnull().all()].tolist()
    if nan_cols:
        errors.append(f"Columns with all NaN values: {nan_cols}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def check_data_quality(df: pd.DataFrame, 
                      max_missing_pct: float = 50) -> Dict[str, Any]:
    """
    Check data quality metrics.
    
    Args:
        df (pd.DataFrame): DataFrame to check
        max_missing_pct (float): Maximum allowed missing value percentage
        
    Returns:
        dict: Quality metrics
    """
    metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'total_cells': len(df) * len(df.columns),
        'missing_cells': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    # Check if quality is acceptable
    metrics['is_acceptable'] = metrics['missing_percentage'] <= max_missing_pct
    
    return metrics


def get_column_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Get statistics for a specific column.
    
    Args:
        df (pd.DataFrame): DataFrame
        column (str): Column name
        
    Returns:
        dict: Column statistics
    """
    if column not in df.columns:
        return {'error': f"Column '{column}' not found"}
    
    col = df[column]
    
    stats = {
        'column_name': column,
        'data_type': str(col.dtype),
        'non_null_count': col.notna().sum(),
        'null_count': col.isnull().sum(),
        'null_percentage': (col.isnull().sum() / len(col)) * 100,
        'unique_values': col.nunique(),
        'unique_percentage': (col.nunique() / len(col)) * 100,
    }
    
    # Add type-specific statistics
    if col.dtype in ['int64', 'float64']:
        stats.update({
            'mean': col.mean(),
            'median': col.median(),
            'std': col.std(),
            'min': col.min(),
            'max': col.max(),
            'q25': col.quantile(0.25),
            'q75': col.quantile(0.75),
        })
    elif col.dtype == 'object':
        stats.update({
            'most_common': col.value_counts().index[0] if len(col) > 0 else None,
            'least_common': col.value_counts().index[-1] if len(col) > 0 else None,
        })
    
    return stats


# ============================================================================
# PRINTING AND FORMATTING FUNCTIONS
# ============================================================================

def print_section_header(title: str, width: int = 80, char: str = "="):
    """
    Print a formatted section header.
    
    Args:
        title (str): Section title
        width (int): Width of the header
        char (str): Character to use for the border
    """
    border = char * width
    print(f"\n{border}")
    print(f"{title.center(width)}")
    print(f"{border}\n")


def print_summary_table(data: Dict[str, Any], title: str = None):
    """
    Print a formatted summary table.
    
    Args:
        data (dict): Data to print
        title (str): Table title (optional)
    """
    if title:
        print(f"\n{'='*60}")
        print(f"{title.center(60)}")
        print(f"{'='*60}")
    
    for key, value in data.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:.4f}")
        else:
            print(f"{key:.<40} {value}")
    
    print()


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value (float): Value to format
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number.
    
    Args:
        value (float): Value to format
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted number string
    """
    return f"{value:,.{decimals}f}"


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare two DataFrames.
    
    Args:
        df1 (pd.DataFrame): First DataFrame
        df2 (pd.DataFrame): Second DataFrame
        
    Returns:
        dict: Comparison results
    """
    return {
        'shapes_match': df1.shape == df2.shape,
        'df1_shape': df1.shape,
        'df2_shape': df2.shape,
        'columns_match': list(df1.columns) == list(df2.columns),
        'df1_columns': list(df1.columns),
        'df2_columns': list(df2.columns),
        'values_match': df1.equals(df2),
        'row_count_match': len(df1) == len(df2),
        'column_count_match': len(df1.columns) == len(df2.columns),
    }


# ============================================================================
# ARRAY/LIST UTILITY FUNCTIONS
# ============================================================================

def flatten_list(nested_list: List) -> List:
    """
    Flatten a nested list.
    
    Args:
        nested_list (list): Nested list to flatten
        
    Returns:
        list: Flattened list
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def get_unique_sorted(items: List) -> List:
    """
    Get unique items in sorted order.
    
    Args:
        items (list): List of items
        
    Returns:
        list: Sorted list of unique items
    """
    return sorted(list(set(items)))


def split_list(items: List, n: int) -> List[List]:
    """
    Split a list into n chunks.
    
    Args:
        items (list): List to split
        n (int): Number of chunks
        
    Returns:
        list: List of chunks
    """
    chunk_size = len(items) // n
    remainder = len(items) % n
    
    chunks = []
    start = 0
    
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(items[start:end])
        start = end
    
    return chunks


# ============================================================================
# TIME/DATE UTILITY FUNCTIONS
# ============================================================================

def get_timestamp() -> str:
    """
    Get current timestamp.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


# ============================================================================
# FILE UTILITY FUNCTIONS
# ============================================================================

def ensure_directory(directory: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory (str): Directory path
        
    Returns:
        Path: Path object
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size(file_path: str) -> str:
    """
    Get file size in human-readable format.
    
    Args:
        file_path (str): Path to file
        
    Returns:
        str: Formatted file size
    """
    file_size_bytes = Path(file_path).stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if file_size_bytes < 1024:
            return f"{file_size_bytes:.2f} {unit}"
        file_size_bytes /= 1024
    
    return f"{file_size_bytes:.2f} TB"


# ============================================================================
# DICTIONARY UTILITY FUNCTIONS
# ============================================================================

def merge_dicts(*dicts: Dict) -> Dict:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Variable number of dictionaries
        
    Returns:
        dict: Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def filter_dict(d: Dict, keys: List[str]) -> Dict:
    """
    Filter dictionary to only include specified keys.
    
    Args:
        d (dict): Dictionary to filter
        keys (list): Keys to keep
        
    Returns:
        dict: Filtered dictionary
    """
    return {k: v for k, v in d.items() if k in keys}


def invert_dict(d: Dict) -> Dict:
    """
    Invert a dictionary (swap keys and values).
    
    Args:
        d (dict): Dictionary to invert
        
    Returns:
        dict: Inverted dictionary
    """
    return {v: k for k, v in d.items()}


if __name__ == "__main__":
    print("Utility functions module loaded successfully")
    print_section_header("Utility Functions Available", width=60)
    
    functions = [
        "setup_logging",
        "validate_dataframe",
        "check_data_quality",
        "get_column_stats",
        "print_section_header",
        "format_percentage",
        "flatten_list",
        "get_timestamp",
        "ensure_directory",
    ]
    
    for i, func in enumerate(functions, 1):
        print(f"{i:2d}. {func}")
