from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import PCA_COMPONENTS


def classify_usda(sand: float, clay: float, silt: float) -> str:
    """Simplified soil texture classification.

    Returns one of: 'sandy', 'clay', 'silty', 'loam'.
    Values are expected as percentages (0–100).
    """
    # Handle missing values robustly
    if any(pd.isna([sand, clay, silt])):
        return "loam"

    if sand >= 70 and clay < 20:
        return "sandy"
    if clay >= 40:
        return "clay"
    if silt >= 50 and clay < 35:
        return "silty"
    return "loam"


def add_soil_type_column(soil_df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'soil_type' column based on USDA-like texture rules."""
    for col in ["sand", "clay", "silt"]:
        if col not in soil_df.columns:
            soil_df[col] = np.nan

    soil_df = soil_df.copy()
    soil_df["soil_type"] = soil_df.apply(
        lambda row: classify_usda(row["sand"], row["clay"], row["silt"]), axis=1
    )
    return soil_df


def _numeric_feature_columns(soil_df: pd.DataFrame) -> List[str]:
    candidate_cols = ["sand", "clay", "silt", "ph", "organic_carbon"]
    return [c for c in candidate_cols if c in soil_df.columns]


def compute_soil_embeddings(
    soil_df: pd.DataFrame, n_components: int | None = None
) -> Tuple[pd.DataFrame, PCA, StandardScaler]:
    """Compute PCA embeddings for soil properties.

    Returns updated DataFrame, fitted PCA and scaler.
    """
    n_components = n_components or PCA_COMPONENTS
    feature_cols = _numeric_feature_columns(soil_df)
    if not feature_cols:
        raise ValueError("No numeric soil feature columns found for embeddings.")

    soil_df = soil_df.copy()
    X = soil_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_emb = pca.fit_transform(X_scaled)

    for i in range(n_components):
        soil_df[f"emb{i+1}"] = X_emb[:, i]

    return soil_df, pca, scaler

