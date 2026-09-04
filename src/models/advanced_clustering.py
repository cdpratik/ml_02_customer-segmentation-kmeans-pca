"""
Advanced Clustering and Segment Profiling Module
Author: Pratik
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


class AdvancedClustering:


    def __init__(self, random_state: int = 42, scaler_type: str = "robust"):
        self.random_state = random_state
        self.scaler_type = scaler_type

        if scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError("scaler_type must be either 'robust' or 'standard'")

        self.model = None
        self.algorithm = None
        self.n_clusters = None
        self.feature_cols = None
        self.log_transform_cols = []
        self.feature_medians = None
        self.X_scaled = None

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        log_transform_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Prepare features for clustering:
        - select columns
        - replace inf with NaN
        - fill missing values with medians
        - log1p transform selected skewed columns
        - scale features
        """

        self.feature_cols = feature_cols
        self.log_transform_cols = log_transform_cols or []

        X = df[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        self.feature_medians = X.median(numeric_only=True)
        X = X.fillna(self.feature_medians)

        for col in self.log_transform_cols:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0))

        self.X_scaled = self.scaler.fit_transform(X)

        return self.X_scaled

    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessing pipeline.
        """

        if self.feature_cols is None:
            raise ValueError("Features have not been prepared yet.")

        X = df[self.feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(self.feature_medians)

        for col in self.log_transform_cols:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0))

        return self.scaler.transform(X)

    def evaluate_labels(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering labels.
        Handles invalid cases safely.
        """

        labels_series = pd.Series(labels)

        # Ignore DBSCAN noise label if present
        valid_mask = labels_series != -1
        valid_labels = labels_series[valid_mask]

        n_clusters = valid_labels.nunique()

        if n_clusters < 2:
            return {
                "n_clusters": int(n_clusters),
                "n_noise": int((labels_series == -1).sum()),
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "min_cluster_size": np.nan,
                "min_cluster_pct": np.nan,
            }

        X_valid = X[valid_mask.values]
        labels_valid = labels_series[valid_mask].values

        cluster_counts = pd.Series(labels_valid).value_counts()
        min_cluster_size = int(cluster_counts.min())
        min_cluster_pct = float(min_cluster_size / len(labels_series) * 100)

        return {
            "n_clusters": int(n_clusters),
            "n_noise": int((labels_series == -1).sum()),
            "silhouette": float(silhouette_score(X_valid, labels_valid)),
            "calinski_harabasz": float(calinski_harabasz_score(X_valid, labels_valid)),
            "davies_bouldin": float(davies_bouldin_score(X_valid, labels_valid)),
            "min_cluster_size": min_cluster_size,
            "min_cluster_pct": min_cluster_pct,
        }

    def compare_algorithms(
        self,
        X: np.ndarray,
        k_range: range = range(3, 9),
        include_dbscan: bool = True,
    ) -> pd.DataFrame:
        """
        Compare clustering algorithms across k values.

        Algorithms:
        - K-Means
        - Gaussian Mixture Model
        - Agglomerative Clustering
        - optional DBSCAN
        """

        results = []

        for k in k_range:
            # K-Means
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
            )
            kmeans_labels = kmeans.fit_predict(X)
            metrics = self.evaluate_labels(X, kmeans_labels)
            metrics.update({
                "algorithm": "kmeans",
                "k": k,
                "inertia": float(kmeans.inertia_),
                "bic": np.nan,
                "aic": np.nan,
            })
            results.append(metrics)

            # Gaussian Mixture Model
            gmm = GaussianMixture(
                n_components=k,
                random_state=self.random_state,
                covariance_type="full",
            )
            gmm_labels = gmm.fit_predict(X)
            metrics = self.evaluate_labels(X, gmm_labels)
            metrics.update({
                "algorithm": "gmm",
                "k": k,
                "inertia": np.nan,
                "bic": float(gmm.bic(X)),
                "aic": float(gmm.aic(X)),
            })
            results.append(metrics)

            # Agglomerative
            agg = AgglomerativeClustering(n_clusters=k)
            agg_labels = agg.fit_predict(X)
            metrics = self.evaluate_labels(X, agg_labels)
            metrics.update({
                "algorithm": "agglomerative",
                "k": k,
                "inertia": np.nan,
                "bic": np.nan,
                "aic": np.nan,
            })
            results.append(metrics)

        if include_dbscan:
            eps_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            min_samples_values = [10, 20]

            for eps in eps_values:
                for min_samples in min_samples_values:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    dbscan_labels = dbscan.fit_predict(X)

                    metrics = self.evaluate_labels(X, dbscan_labels)
                    metrics.update({
                        "algorithm": "dbscan",
                        "k": np.nan,
                        "eps": eps,
                        "min_samples": min_samples,
                        "inertia": np.nan,
                        "bic": np.nan,
                        "aic": np.nan,
                    })
                    results.append(metrics)

        comparison_df = pd.DataFrame(results)

        if "eps" not in comparison_df.columns:
            comparison_df["eps"] = np.nan

        if "min_samples" not in comparison_df.columns:
            comparison_df["min_samples"] = np.nan

        return comparison_df

    def fit_predict(
        self,
        X: np.ndarray,
        algorithm: str = "kmeans",
        n_clusters: int = 5,
        **kwargs,
    ) -> Tuple[object, np.ndarray]:
        """
        Fit selected final clustering algorithm.
        """

        self.algorithm = algorithm
        self.n_clusters = n_clusters

        if algorithm == "kmeans":
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            labels = self.model.fit_predict(X)

        elif algorithm == "gmm":
            self.model = GaussianMixture(
                n_components=n_clusters,
                random_state=self.random_state,
                covariance_type=kwargs.get("covariance_type", "full"),
            )
            labels = self.model.fit_predict(X)

        elif algorithm == "agglomerative":
            self.model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = self.model.fit_predict(X)

        elif algorithm == "dbscan":
            eps = kwargs.get("eps", 1.0)
            min_samples = kwargs.get("min_samples", 10)

            self.model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = self.model.fit_predict(X)

        else:
            raise ValueError(
                "algorithm must be one of: 'kmeans', 'gmm', 'agglomerative', 'dbscan'"
            )

        return self.model, labels

    def get_pca_projection(self, X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
        """
        Create PCA projection for visualization.
        """

        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        return X_pca, pca

    def inverse_transform_centers(self) -> pd.DataFrame:
        """
        Return cluster centers in original feature scale.

        Works for:
        - KMeans cluster centers
        - GMM component means

        Agglomerative and DBSCAN do not have model centers.
        """

        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        if hasattr(self.model, "cluster_centers_"):
            centers_scaled = self.model.cluster_centers_
        elif hasattr(self.model, "means_"):
            centers_scaled = self.model.means_
        else:
            raise ValueError(
                "This model does not expose cluster centers. "
                "Use group-level profiles instead."
            )

        centers_transformed = self.scaler.inverse_transform(centers_scaled)
        centers_df = pd.DataFrame(centers_transformed, columns=self.feature_cols)

        for col in self.log_transform_cols:
            if col in centers_df.columns:
                centers_df[col] = np.expm1(centers_df[col]).clip(lower=0)

        centers_df.index.name = "Cluster"

        return centers_df


def profile_segments(
    df: pd.DataFrame,
    segment_col: str,
    customer_col: str = "Customer ID",
) -> pd.DataFrame:
    """
    Create business-friendly profile table for segments/clusters.
    """

    agg_dict = {
        customer_col: "count",
    }

    if "Recency" in df.columns:
        agg_dict["Recency"] = ["mean", "median"]

    if "Frequency" in df.columns:
        agg_dict["Frequency"] = ["mean", "median"]

    if "Monetary" in df.columns:
        agg_dict["Monetary"] = ["mean", "median", "sum"]

    if "Avg_Order_Value" in df.columns:
        agg_dict["Avg_Order_Value"] = "mean"

    if "Unique_Products" in df.columns:
        agg_dict["Unique_Products"] = "mean"

    if "Avg_Items_Per_Order" in df.columns:
        agg_dict["Avg_Items_Per_Order"] = "mean"

    if "Customer_Lifespan_Days" in df.columns:
        agg_dict["Customer_Lifespan_Days"] = "mean"

    if "Purchase_Frequency" in df.columns:
        agg_dict["Purchase_Frequency"] = "mean"

    profile = df.groupby(segment_col).agg(agg_dict).round(2)

    profile.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in profile.columns
    ]

    rename_map = {
        f"{customer_col}_count": "Count",
        "Recency_mean": "Avg_Recency",
        "Recency_median": "Median_Recency",
        "Frequency_mean": "Avg_Frequency",
        "Frequency_median": "Median_Frequency",
        "Monetary_mean": "Avg_Monetary",
        "Monetary_median": "Median_Monetary",
        "Monetary_sum": "Total_Revenue",
        "Avg_Order_Value_mean": "Avg_Order_Value",
        "Unique_Products_mean": "Avg_Unique_Products",
        "Avg_Items_Per_Order_mean": "Avg_Items_Per_Order",
        "Customer_Lifespan_Days_mean": "Avg_Customer_Lifespan_Days",
        "Purchase_Frequency_mean": "Avg_Purchase_Frequency",
    }

    profile = profile.rename(columns=rename_map)

    profile["Customer_Share_%"] = (
        profile["Count"] / profile["Count"].sum() * 100
    ).round(1)

    if "Total_Revenue" in profile.columns:
        profile["Revenue_Share_%"] = (
            profile["Total_Revenue"] / profile["Total_Revenue"].sum() * 100
        ).round(1)

    return profile


def assign_business_labels(profile: pd.DataFrame) -> Dict[int, str]:
    """
    Assign business labels based on segment strength.

    Lower Recency is better.
    Higher Frequency and Monetary are better.
    """

    scoring = pd.DataFrame(index=profile.index)

    if "Avg_Recency" in profile.columns:
        # lower recency should score higher
        scoring["recency_score"] = profile["Avg_Recency"].rank(ascending=False)

    if "Avg_Frequency" in profile.columns:
        scoring["frequency_score"] = profile["Avg_Frequency"].rank(ascending=True)

    if "Avg_Monetary" in profile.columns:
        scoring["monetary_score"] = profile["Avg_Monetary"].rank(ascending=True)

    if "Avg_Unique_Products" in profile.columns:
        scoring["product_score"] = profile["Avg_Unique_Products"].rank(ascending=True)

    scoring["total_score"] = scoring.sum(axis=1)

    sorted_clusters = scoring["total_score"].sort_values(ascending=False).index.tolist()

    labels_pool = [
        "Champions",
        "Loyal High Value",
        "Potential Loyalists",
        "At Risk",
        "Hibernating / Low Value",
        "Occasional Buyers",
        "Needs Attention",
        "Low Value",
    ]

    labels = {}

    for i, cluster in enumerate(sorted_clusters):
        labels[cluster] = labels_pool[i] if i < len(labels_pool) else f"Segment {cluster}"

    return labels