from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import RANDOM_SEED


DEFAULT_CLUSTER_FEATURES = [
    "mean_speed",
    "mean_accel_z",
    "std_accel_z",
    "mean_gyro_y",
    "std_gyro_y",
    "slip_ratio",
    "mean_motor_current",
]


def cluster_telemetry(
    df_features: pd.DataFrame,
    n_clusters: int = 4,
    feature_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Cluster telemetry feature windows with KMeans and add 'cluster_label'."""
    if df_features.empty:
        raise ValueError("df_features is empty.")

    feature_cols = list(feature_cols) if feature_cols is not None else DEFAULT_CLUSTER_FEATURES
    feature_cols = [c for c in feature_cols if c in df_features.columns]
    if not feature_cols:
        raise ValueError("No valid feature columns found for clustering.")

    df = df_features.copy()
    X = df[feature_cols].values

    # Handle NaNs by simple imputation (mean per column)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init="auto")
    df["cluster_label"] = kmeans.fit_predict(X)
    return df

