"""
Data Validation Functions
Author: Pratik
Date: 2024
Description: Validation functions for Online Retail II dataset
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has expected Online Retail II schema
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
        
    Example:
    --------
    >>> is_valid, issues = validate_schema(df)
    >>> if not is_valid:
    ...     print("Issues found:", issues)
    """
    expected_columns = [
        'InvoiceNo', 'StockCode', 'Description', 'Quantity',
        'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'
    ]
    
    issues = []
    
    # Check if all expected columns exist
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check extra columns
    extra_cols = set(df.columns) - set(expected_columns)
    if extra_cols:
        issues.append(f"Unexpected columns: {extra_cols}")
    
    # Validate data types (if columns exist)
    if 'Quantity' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Quantity']):
            issues.append("'Quantity' should be numeric type")
    
    if 'UnitPrice' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['UnitPrice']):
            issues.append("'UnitPrice' should be numeric type")
    
    if 'InvoiceDate' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
            # Try to convert
            try:
                pd.to_datetime(df['InvoiceDate'])
            except:
                issues.append("'InvoiceDate' cannot be converted to datetime")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues


def identify_data_quality_issues(df: pd.DataFrame, 
                                 verbose: bool = True) -> Dict:
    """
    Identify common data quality issues in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    verbose : bool
        Whether to print findings
        
    Returns:
    --------
    Dict
        Dictionary of issues with counts and percentages
        
    Example:
    --------
    >>> issues = identify_data_quality_issues(df)
    >>> print(f"Missing Customer IDs: {issues['missing_customer_id']['count']}")
    """
    issues = {}
    total_rows = len(df)
    
    # 1. Missing CustomerID
    if 'CustomerID' in df.columns:
        missing_cust = df['CustomerID'].isnull().sum()
        issues['missing_customer_id'] = {
            'count': int(missing_cust),
            'percentage': round(missing_cust / total_rows * 100, 2),
            'severity': 'HIGH' if (missing_cust / total_rows) > 0.1 else 'MEDIUM'
        }
    
    # 2. Cancelled transactions (InvoiceNo starts with 'C')
    if 'InvoiceNo' in df.columns:
        cancelled = df[df['InvoiceNo'].astype(str).str.startswith('C', na=False)].shape[0]
        issues['cancelled_transactions'] = {
            'count': int(cancelled),
            'percentage': round(cancelled / total_rows * 100, 2),
            'severity': 'MEDIUM'
        }
    
    # 3. Negative quantities
    if 'Quantity' in df.columns:
        neg_qty = (df['Quantity'] < 0).sum()
        issues['negative_quantities'] = {
            'count': int(neg_qty),
            'percentage': round(neg_qty / total_rows * 100, 2),
            'severity': 'MEDIUM',
            'note': 'Likely represents returns or corrections'
        }
    
    # 4. Zero quantities
    if 'Quantity' in df.columns:
        zero_qty = (df['Quantity'] == 0).sum()
        issues['zero_quantities'] = {
            'count': int(zero_qty),
            'percentage': round(zero_qty / total_rows * 100, 2),
            'severity': 'LOW'
        }
    
    # 5. Negative or zero prices
    if 'UnitPrice' in df.columns:
        invalid_price = (df['UnitPrice'] <= 0).sum()
        issues['invalid_prices'] = {
            'count': int(invalid_price),
            'percentage': round(invalid_price / total_rows * 100, 2),
            'severity': 'HIGH'
        }
    
    # 6. Missing descriptions
    if 'Description' in df.columns:
        missing_desc = df['Description'].isnull().sum()
        issues['missing_descriptions'] = {
            'count': int(missing_desc),
            'percentage': round(missing_desc / total_rows * 100, 2),
            'severity': 'LOW'
        }
    
    # 7. Duplicate records
    duplicates = df.duplicated().sum()
    issues['duplicate_records'] = {
        'count': int(duplicates),
        'percentage': round(duplicates / total_rows * 100, 2),
        'severity': 'MEDIUM'
    }
    
    # 8. Missing values (any column)
    total_missing = df.isnull().sum().sum()
    issues['total_missing_values'] = {
        'count': int(total_missing),
        'percentage': round(total_missing / (total_rows * len(df.columns)) * 100, 2),
        'severity': 'INFO'
    }
    
    if verbose:
        print("="*80)
        print("DATA QUALITY ISSUES IDENTIFIED")
        print("="*80)
        
        for issue_name, issue_data in issues.items():
            severity = issue_data.get('severity', 'INFO')
            severity_symbol = {
                'HIGH': '🔴',
                'MEDIUM': '🟡',
                'LOW': '🟢',
                'INFO': 'ℹ️'
            }.get(severity, 'ℹ️')
            
            print(f"\n{severity_symbol} {issue_name.replace('_', ' ').title()}")
            print(f"   Count: {issue_data['count']:,}")
            print(f"   Percentage: {issue_data['percentage']}%")
            print(f"   Severity: {severity}")
            
            if 'note' in issue_data:
                print(f"   Note: {issue_data['note']}")
        
        print(f"\n{'='*80}\n")
    
    return issues


def validate_date_range(df: pd.DataFrame, 
                       date_column: str = 'InvoiceDate',
                       expected_start: Optional[str] = None,
                       expected_end: Optional[str] = None) -> Dict:
    """
    Validate date range in dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    date_column : str
        Name of date column
    expected_start : str, optional
        Expected start date (YYYY-MM-DD)
    expected_end : str, optional
        Expected end date (YYYY-MM-DD)
        
    Returns:
    --------
    Dict
        Date range information and validation results
        
    Example:
    --------
    >>> date_info = validate_date_range(df, expected_start='2009-12-01')
    """
    if date_column not in df.columns:
        return {'error': f"Column '{date_column}' not found"}
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    date_range_days = (max_date - min_date).days
    
    result = {
        'min_date': str(min_date),
        'max_date': str(max_date),
        'range_days': date_range_days,
        'range_months': round(date_range_days / 30, 1),
        'range_years': round(date_range_days / 365, 1)
    }
    
    # Validate against expected range
    if expected_start:
        expected_start_dt = pd.to_datetime(expected_start)
        result['start_matches_expected'] = min_date >= expected_start_dt
        
    if expected_end:
        expected_end_dt = pd.to_datetime(expected_end)
        result['end_matches_expected'] = max_date <= expected_end_dt
    
    return result


def validate_numeric_ranges(df: pd.DataFrame,
                            column_ranges: Dict[str, Tuple[float, float]]) -> Dict:
    """
    Validate that numeric columns fall within expected ranges
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column_ranges : Dict[str, Tuple[float, float]]
        Dictionary mapping column names to (min, max) tuples
        
    Returns:
    --------
    Dict
        Validation results for each column
        
    Example:
    --------
    >>> ranges = {'UnitPrice': (0, 10000), 'Quantity': (1, 10000)}
    >>> validation = validate_numeric_ranges(df, ranges)
    """
    results = {}
    
    for column, (expected_min, expected_max) in column_ranges.items():
        if column not in df.columns:
            results[column] = {'error': 'Column not found'}
            continue
        
        actual_min = df[column].min()
        actual_max = df[column].max()
        
        out_of_range = df[
            (df[column] < expected_min) | (df[column] > expected_max)
        ].shape[0]
        
        results[column] = {
            'actual_min': float(actual_min),
            'actual_max': float(actual_max),
            'expected_min': expected_min,
            'expected_max': expected_max,
            'within_range': (actual_min >= expected_min) and (actual_max <= expected_max),
            'out_of_range_count': int(out_of_range),
            'out_of_range_percentage': round(out_of_range / len(df) * 100, 2)
        }
    
    return results


def check_referential_integrity(df: pd.DataFrame) -> Dict:
    """
    Check relationships between columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    Dict
        Referential integrity check results
        
    Example:
    --------
    >>> integrity = check_referential_integrity(df)
    """
    results = {}
    
    # Check InvoiceNo and CustomerID relationship
    if 'InvoiceNo' in df.columns and 'CustomerID' in df.columns:
        # Each invoice should belong to one customer
        invoice_customer = df.groupby('InvoiceNo')['CustomerID'].nunique()
        multi_customer_invoices = (invoice_customer > 1).sum()
        
        results['invoice_customer_relationship'] = {
            'total_invoices': len(invoice_customer),
            'multi_customer_invoices': int(multi_customer_invoices),
            'is_valid': multi_customer_invoices == 0
        }
    
    # Check if same StockCode has multiple descriptions
    if 'StockCode' in df.columns and 'Description' in df.columns:
        stock_descriptions = df.groupby('StockCode')['Description'].nunique()
        multi_desc_stocks = (stock_descriptions > 1).sum()
        
        results['stockcode_description_relationship'] = {
            'total_products': len(stock_descriptions),
            'products_with_multiple_descriptions': int(multi_desc_stocks),
            'is_consistent': multi_desc_stocks == 0
        }
    
    return results


def generate_validation_report(df: pd.DataFrame, 
                               save_path: Optional[str] = None) -> Dict:
    """
    Generate comprehensive validation report
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    save_path : str, optional
        Path to save JSON report
        
    Returns:
    --------
    Dict
        Complete validation report
        
    Example:
    --------
    >>> report = generate_validation_report(df, 'reports/validation.json')
    """
    from datetime import datetime
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'dataset_shape': {
            'rows': len(df),
            'columns': len(df.columns)
        },
        'schema_validation': {},
        'data_quality_issues': {},
        'date_validation': {},
        'referential_integrity': {}
    }
    
    # Schema validation
    is_valid, issues = validate_schema(df)
    report['schema_validation'] = {
        'is_valid': is_valid,
        'issues': issues
    }
    
    # Data quality issues
    report['data_quality_issues'] = identify_data_quality_issues(df, verbose=False)
    
    # Date validation
    if 'InvoiceDate' in df.columns:
        report['date_validation'] = validate_date_range(df)
    
    # Referential integrity
    report['referential_integrity'] = check_referential_integrity(df)
    
    # Save if path provided
    if save_path:
        import json
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        print(f"✓ Validation report saved to: {save_path}")
    
    return report


if __name__ == "__main__":
    print("Data validation module loaded successfully!")
    print("\nAvailable functions:")
    print("  - validate_schema()")
    print("  - identify_data_quality_issues()")
    print("  - validate_date_range()")
    print("  - validate_numeric_ranges()")
    print("  - check_referential_integrity()")
    print("  - generate_validation_report()")