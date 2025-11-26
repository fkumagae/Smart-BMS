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

    # Remove features sem variancia para evitar clusters degenerados.
    feature_stds = df[feature_cols].std()
    zero_var_cols = [c for c, std in feature_stds.items() if np.isnan(std) or std < 1e-9]
    if zero_var_cols:
        feature_cols = [c for c in feature_cols if c not in zero_var_cols]
    if not feature_cols:
        raise ValueError(
            "Todas as features selecionadas tem variancia zero/NaN. "
            "Verifique se a telemetria esta constante ou se o janelamento retornou valores nulos."
        )

    X = df[feature_cols].values

    # Handle NaNs by simple imputation (mean per column).
    # Se a coluna inteira for NaN, substituímos por 0.
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    # Se existirem poucos pontos distintos, o KMeans pode colapsar em 1 cluster.
    distinct_points = np.unique(np.round(X, decimals=6), axis=0).shape[0]
    if distinct_points < n_clusters:
        raise ValueError(
            f"Apenas {distinct_points} pontos distintos para {n_clusters} clusters. "
            "Isso gera todos os rotulos iguais. "
            "Reduza N_CLUSTERS ou verifique se as features nao estao constantes/nulas."
        )

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init="auto")
    df["cluster_label"] = kmeans.fit_predict(X)
    return df


def compute_elbow_inertia(
    df_features: pd.DataFrame,
    k_range: Iterable[int],
    feature_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Calcula a inércia (soma das distâncias ao centróide) para vários K.

    Útil para aplicar a "regra do cotovelo" e escolher o número de clusters.
    """
    if df_features.empty:
        raise ValueError("df_features is empty.")

    feature_cols = list(feature_cols) if feature_cols is not None else DEFAULT_CLUSTER_FEATURES
    feature_cols = [c for c in feature_cols if c in df_features.columns]
    if not feature_cols:
        raise ValueError("No valid feature columns found for clustering.")

    df = df_features.copy()
    X = df[feature_cols].values

    # mesma imputação usada no cluster_telemetry
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init="auto")
        km.fit(X)
        results.append({"k": k, "inertia": km.inertia_})

    return pd.DataFrame(results)
