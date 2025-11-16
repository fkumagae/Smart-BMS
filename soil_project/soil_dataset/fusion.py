from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def get_representative_embedding_for_soil_type(
    soil_df: pd.DataFrame, soil_type: str
) -> np.ndarray:
    """Compute average PCA embedding for subset of soil_df matching soil_type rules."""
    df = soil_df.copy()

    # Ensure texture columns exist
    for col in ["sand", "clay", "silt"]:
        if col not in df.columns:
            df[col] = np.nan

    if soil_type == "sandy":
        subset = df[df["sand"] > 70]
    elif soil_type == "clay":
        subset = df[df["clay"] > 40]
    elif soil_type == "silty":
        subset = df[df["silt"] > 50]
    else:  # 'loam' or others
        subset = df[
            (df["sand"].between(30, 70))
            & (df["clay"].between(10, 40))
            & (df["silt"].between(10, 50))
        ]

    emb_cols: List[str] = [c for c in df.columns if c.startswith("emb")]
    if not emb_cols:
        raise ValueError("No embedding columns (emb1..embN) found in soil_df.")

    if subset.empty:
        # Fallback to overall mean
        subset = df

    emb_mean = subset[emb_cols].mean().values
    return emb_mean


def attach_embeddings_to_telemetry(
    telemetry_windowed: pd.DataFrame, soil_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach soil embeddings to telemetry based on soil_type_cluster."""
    df = telemetry_windowed.copy()

    if "soil_type_cluster" not in df.columns:
        raise ValueError("telemetry_windowed must contain 'soil_type_cluster' column.")

    soil_types = df["soil_type_cluster"].dropna().unique()

    emb_cols = [c for c in soil_df.columns if c.startswith("emb")]
    if not emb_cols:
        raise ValueError("No embedding columns in soil_df.")

    n_emb = len(emb_cols)
    for i in range(n_emb):
        df[f"emb{i+1}"] = np.nan

    cache: Dict[str, np.ndarray] = {}
    for stype in soil_types:
        emb = get_representative_embedding_for_soil_type(soil_df, stype)
        cache[stype] = emb

    # Assign embeddings row-wise
    def _row_emb(row):
        stype = row["soil_type_cluster"]
        if stype not in cache:
            return [np.nan] * n_emb
        return cache[stype]

    emb_array = np.vstack(df.apply(_row_emb, axis=1).values)
    for i in range(n_emb):
        df[f"emb{i+1}"] = emb_array[:, i]

    # Final label column for training
    df = df.rename(columns={"soil_type_cluster": "soil_type"})
    return df

