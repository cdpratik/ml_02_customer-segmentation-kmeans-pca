"""
Utility Helper Functions
Author: Pratik
Date: 2024
Description: Common utility functions for customer segmentation project
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def print_header(text: str, width: int = 80, char: str = "=") -> None:
    """
    Print formatted section header
    
    Parameters:
    -----------
    text : str
        Header text
    width : int
        Total width of header
    char : str
        Character to use for border
        
    Example:
    --------
    >>> print_header("DATA OVERVIEW")
    ================================================================================
                                    DATA OVERVIEW                                    
    ================================================================================
    """
    print(f"\n{char * width}")
    print(text.center(width))
    print(f"{char * width}\n")


def print_subheader(text: str, width: int = 80) -> None:
    """
    Print formatted subsection header
    
    Parameters:
    -----------
    text : str
        Subheader text
    width : int
        Total width
    """
    print(f"\n{text}")
    print("-" * width)


def load_data(file_path: str, sheet_name: str = 'Year 2009-2010') -> pd.DataFrame:
    """
    Load Online Retail II dataset from Excel file
    
    Parameters:
    -----------
    file_path : str
        Path to Excel file
    sheet_name : str
        Name of sheet to load
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
        
    Example:
    --------
    >>> df = load_data('data/raw/online_retail_II.xlsx')
    """
    try:
        print(f"Loading dataset from: {file_path}")
        print(f"Sheet: {sheet_name}\n")
        
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        
        return df
        
    except FileNotFoundError:
        print(f"✗ Error: File not found at {file_path}")
        print("  Please ensure the dataset is downloaded to data/raw/")
        return None
        
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        return None


def save_dataframe(df: pd.DataFrame, 
                   file_path: str, 
                   file_format: str = 'csv',
                   **kwargs) -> None:
    """
    Save DataFrame to file
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    file_path : str
        Output file path
    file_format : str
        Format: 'csv', 'excel', 'parquet', 'pickle'
    **kwargs : dict
        Additional arguments for save function
        
    Example:
    --------
    >>> save_dataframe(df, 'data/processed/cleaned.csv', file_format='csv')
    """
    try:
        if file_format == 'csv':
            df.to_csv(file_path, index=kwargs.get('index', False))
        elif file_format == 'excel':
            df.to_excel(file_path, index=kwargs.get('index', False))
        elif file_format == 'parquet':
            df.to_parquet(file_path, index=kwargs.get('index', False))
        elif file_format == 'pickle':
            df.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        file_size = pd.read_csv(file_path).memory_usage(deep=True).sum() / 1024**2 if file_format == 'csv' else 0
        print(f"✓ Data saved to: {file_path}")
        if file_size > 0:
            print(f"  File size: {file_size:.2f} MB")
            
    except Exception as e:
        print(f"✗ Error saving data: {str(e)}")


def get_dataframe_info(df: pd.DataFrame) -> Dict:
    """
    Get comprehensive DataFrame information
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    Dict
        Dictionary with dataset statistics
        
    Example:
    --------
    >>> info = get_dataframe_info(df)
    >>> print(f"Total rows: {info['n_rows']}")
    """
    info = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'duplicates': int(df.duplicated().sum()),
        'total_missing': int(df.isnull().sum().sum()),
        'missing_percentage': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2)
    }
    
    return info


def check_missing_values(df: pd.DataFrame, sort_by: str = 'Missing_Percentage') -> pd.DataFrame:
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': np.round(
            (df.isnull().sum().values / len(df)) * 100,
            2
        )
    })

    missing = missing[missing['Missing_Count'] > 0]

    return missing.sort_values(sort_by, ascending=False)


def get_duplicate_info(df: pd.DataFrame, 
                       subset: Optional[List[str]] = None) -> Dict:
    """
    Get detailed duplicate information
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    subset : List[str], optional
        Columns to consider for duplicates
        
    Returns:
    --------
    Dict
        Duplicate statistics
        
    Example:
    --------
    >>> dup_info = get_duplicate_info(df)
    >>> print(f"Duplicates: {dup_info['count']}")
    """
    if subset:
        duplicates = df.duplicated(subset=subset).sum()
        duplicate_rows = df[df.duplicated(subset=subset, keep=False)]
    else:
        duplicates = df.duplicated().sum()
        duplicate_rows = df[df.duplicated(keep=False)]
    
    return {
        'count': int(duplicates),
        'percentage': round(duplicates / len(df) * 100, 2),
        'duplicate_rows': duplicate_rows
    }


def save_json(data: Dict, file_path: str, indent: int = 4) -> None:
    """
    Save dictionary to JSON file
    
    Parameters:
    -----------
    data : Dict
        Dictionary to save
    file_path : str
        Output file path
    indent : int
        JSON indentation
        
    Example:
    --------
    >>> save_json(analysis_results, 'reports/results.json')
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        print(f"✓ JSON saved to: {file_path}")
        
    except Exception as e:
        print(f"✗ Error saving JSON: {str(e)}")


def load_json(file_path: str) -> Optional[Dict]:
    """
    Load JSON file into dictionary
    
    Parameters:
    -----------
    file_path : str
        Path to JSON file
        
    Returns:
    --------
    Dict or None
        Loaded dictionary
        
    Example:
    --------
    >>> config = load_json('config/settings.json')
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✓ JSON loaded from: {file_path}")
        return data
        
    except FileNotFoundError:
        print(f"✗ Error: File not found at {file_path}")
        return None
        
    except Exception as e:
        print(f"✗ Error loading JSON: {str(e)}")
        return None


def display_value_counts(df: pd.DataFrame, 
                        column: str, 
                        top_n: int = 10,
                        show_percentage: bool = True) -> pd.DataFrame:
    """
    Display value counts for a column
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name
    top_n : int
        Number of top values to show
    show_percentage : bool
        Whether to show percentages
        
    Returns:
    --------
    pd.DataFrame
        Value counts with optional percentages
        
    Example:
    --------
    >>> display_value_counts(df, 'Country', top_n=5)
    """
    counts = df[column].value_counts().head(top_n)
    
    if show_percentage:
        result = pd.DataFrame({
            'Value': counts.index,
            'Count': counts.values,
            'Percentage': np.round((counts.values / len(df)) * 100, 2)
        })
    else:
        result = pd.DataFrame({
            'Value': counts.index,
            'Count': counts.values
        })
    
    return result.reset_index(drop=True)


def get_numeric_summary(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get enhanced summary statistics for numeric columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : List[str], optional
        Specific columns to analyze
        
    Returns:
    --------
    pd.DataFrame
        Enhanced summary statistics
        
    Example:
    --------
    >>> summary = get_numeric_summary(df, ['Quantity', 'UnitPrice'])
    """
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[columns]
    
    summary = numeric_df.describe().T
    summary['missing'] = numeric_df.isnull().sum()
    summary['missing_pct'] = round((numeric_df.isnull().sum() / len(df)) * 100, 2)
    summary['zeros'] = (numeric_df == 0).sum()
    summary['negatives'] = (numeric_df < 0).sum()
    
    # Reorder columns
    cols = ['count', 'missing', 'missing_pct', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'zeros', 'negatives']
    summary = summary[[col for col in cols if col in summary.columns]]
    
    return summary


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format number with thousand separators
    
    Parameters:
    -----------
    num : float
        Number to format
    decimals : int
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted number
        
    Example:
    --------
    >>> format_number(1234567.89)
    '1,234,567.89'
    """
    return f"{num:,.{decimals}f}"


def create_timestamp() -> str:
    """
    Create formatted timestamp string
    
    Returns:
    --------
    str
        Timestamp in format: YYYY-MM-DD_HH-MM-SS
        
    Example:
    --------
    >>> timestamp = create_timestamp()
    >>> print(timestamp)  # '2024-01-15_14-30-45'
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def memory_usage_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate memory usage report by column
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Memory usage by column
        
    Example:
    --------
    >>> mem_report = memory_usage_report(df)
    """
    mem = df.memory_usage(deep=True)
    mem_mb = mem / 1024**2
    
    report = pd.DataFrame({
        'Column': mem.index,
        'Memory_MB': mem_mb.values,
        'Percentage': round((mem_mb.values / mem_mb.sum()) * 100, 2)
    })
    
    report = report.sort_values('Memory_MB', ascending=False).reset_index(drop=True)
    
    return report


if __name__ == "__main__":
    print("Utility helpers module loaded successfully!")
    print("\nAvailable functions:")
    print("  - print_header()")
    print("  - load_data()")
    print("  - save_dataframe()")
    print("  - get_dataframe_info()")
    print("  - check_missing_values()")
    print("  - get_duplicate_info()")
    print("  - save_json() / load_json()")
    print("  - display_value_counts()")
    print("  - get_numeric_summary()")
    print("  - and more...")