"""
K-Means Clustering Module for Customer Segmentation
Author: Pratik
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


class CustomerClustering:
    """
    K-Means clustering pipeline for customer segmentation.
    Handles scaling, optimal-k search, fitting, and evaluation.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.optimal_k = None
        self.feature_cols = None
        self.X_scaled = None

    def prepare_features(self, df: pd.DataFrame,
                         feature_cols: List[str]) -> np.ndarray:
        """Select and scale features for clustering."""
        self.feature_cols = feature_cols
        X = df[feature_cols].copy()

        # Guard against NaN / inf introduced by ratios (e.g. lifespan = 0)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        self.X_scaled = self.scaler.fit_transform(X)
        return self.X_scaled

    def find_optimal_k(self, X: np.ndarray,
                       k_range: range = range(2, 11)) -> pd.DataFrame:
        """
        Evaluate multiple k values using several metrics.
        Returns a DataFrame with one row per k.
        """
        results = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)

            results.append({
                'k': k,
                'inertia': km.inertia_,
                'silhouette': silhouette_score(X, labels),
                'calinski_harabasz': calinski_harabasz_score(X, labels),
                'davies_bouldin': davies_bouldin_score(X, labels),
            })

        return pd.DataFrame(results)

    def fit(self, X: np.ndarray, n_clusters: int) -> KMeans:
        """Fit the final K-Means model."""
        self.optimal_k = n_clusters
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.model.fit(X)
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict(X)

    def get_cluster_centers(self) -> pd.DataFrame:
        """Return cluster centers in the ORIGINAL feature scale."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")

        centers_scaled = self.model.cluster_centers_
        centers_original = self.scaler.inverse_transform(centers_scaled)

        centers_df = pd.DataFrame(centers_original, columns=self.feature_cols)
        centers_df.index.name = 'Cluster'
        return centers_df

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Return evaluation metrics for a given labelling."""
        return {
            'silhouette': silhouette_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'inertia': self.model.inertia_ if self.model else None,
        }


def add_cluster_labels(df: pd.DataFrame,
                       labels: np.ndarray,
                       col_name: str = 'Cluster') -> pd.DataFrame:
    """Attach cluster labels to a copy of the dataframe."""
    df_out = df.copy()
    df_out[col_name] = labels
    return df_out