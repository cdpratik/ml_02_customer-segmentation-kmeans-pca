"""
RFM Analysis Module
Author: Pratik
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional


class RFMAnalyzer:
    """
    RFM (Recency, Frequency, Monetary) Analysis for customer segmentation
    """
    
    def __init__(self, reference_date: Optional[pd.Timestamp] = None):
        self.reference_date = reference_date
        self.rfm_data = None
        
    def calculate_rfm(self, df: pd.DataFrame, 
                     customer_col: str = 'Customer ID',
                     date_col: str = 'InvoiceDate',
                     revenue_col: str = 'Revenue',
                     invoice_col: str = 'Invoice') -> pd.DataFrame:
        
        if self.reference_date is None:
            self.reference_date = df[date_col].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby(customer_col).agg({
            date_col: lambda x: (self.reference_date - x.max()).days,
            invoice_col: 'nunique',
            revenue_col: 'sum'
        }).reset_index()
        
        rfm.columns = [customer_col, 'Recency', 'Frequency', 'Monetary']
        
        self.rfm_data = rfm
        
        return rfm
    
    def _safe_qcut(self, series: pd.Series, labels: list, q: int = 5):
        """
        Safe qcut that handles duplicate edges by ranking first.
        """
        # rank(method='first') breaks ties by order of appearance -> unique values
        ranked = series.rank(method='first')
        try:
            return pd.qcut(ranked, q=q, labels=labels, duplicates='drop')
        except ValueError:
            # Ultimate fallback: if still fails (e.g., < q unique values)
            n_bins = min(q, series.nunique())
            if n_bins <= 1:
                mid = labels[len(labels)//2]
                return pd.Series([mid]*len(series), index=series.index)
            # Adjust labels to match n_bins
            is_desc = labels[0] > labels[-1]
            if is_desc:
                adj = np.linspace(labels[0], labels[-1], n_bins).astype(int).tolist()
                adj = sorted(adj, reverse=True)
            else:
                adj = np.linspace(labels[0], labels[-1], n_bins).astype(int).tolist()
                adj = sorted(adj)
            return pd.qcut(ranked, q=n_bins, labels=adj, duplicates='drop')
    
    def score_rfm(self, rfm_df: pd.DataFrame, 
                  recency_labels: list = [5, 4, 3, 2, 1],
                  frequency_labels: list = [1, 2, 3, 4, 5],
                  monetary_labels: list = [1, 2, 3, 4, 5]) -> pd.DataFrame:
        
        rfm_scored = rfm_df.copy()
        
        # FIX: rank first to avoid duplicate bin edges
        rfm_scored['R_Score'] = self._safe_qcut(rfm_scored['Recency'], recency_labels)
        rfm_scored['F_Score'] = self._safe_qcut(rfm_scored['Frequency'], frequency_labels)
        rfm_scored['M_Score'] = self._safe_qcut(rfm_scored['Monetary'], monetary_labels)
        
        rfm_scored['R_Score'] = rfm_scored['R_Score'].astype(int)
        rfm_scored['F_Score'] = rfm_scored['F_Score'].astype(int)
        rfm_scored['M_Score'] = rfm_scored['M_Score'].astype(int)
        
        rfm_scored['RFM_Score'] = (rfm_scored['R_Score'].astype(str) + 
                                   rfm_scored['F_Score'].astype(str) + 
                                   rfm_scored['M_Score'].astype(str))
        
        rfm_scored['RFM_Total'] = rfm_scored['R_Score'] + rfm_scored['F_Score'] + rfm_scored['M_Score']
        
        return rfm_scored
    
    def get_rfm_statistics(self) -> pd.DataFrame:
        
        if self.rfm_data is None:
            raise ValueError("RFM data not calculated yet")
        
        stats = self.rfm_data[['Recency', 'Frequency', 'Monetary']].describe()
        
        return stats


def create_additional_features(df: pd.DataFrame, 
                               customer_col: str = 'Customer ID') -> pd.DataFrame:
    
    additional_features = df.groupby(customer_col).agg({
        'Quantity': 'sum',
        'Invoice': 'nunique',
        'Revenue': ['sum', 'mean'],
        'StockCode': 'nunique',
        'InvoiceDate': ['min', 'max']
    }).reset_index()
    
    additional_features.columns = [
        customer_col, 'Total_Items', 'Total_Orders', 
        'Total_Revenue', 'Avg_Order_Value', 'Unique_Products', 
        'First_Purchase', 'Last_Purchase'
    ]
    
    additional_features['Avg_Items_Per_Order'] = (
        additional_features['Total_Items'] / additional_features['Total_Orders']
    )
    
    additional_features['Customer_Lifespan_Days'] = (
        additional_features['Last_Purchase'] - additional_features['First_Purchase']
    ).dt.days
    
    additional_features['Purchase_Frequency'] = (
        additional_features['Total_Orders'] / (additional_features['Customer_Lifespan_Days'] + 1)
    )
    
    return additional_features