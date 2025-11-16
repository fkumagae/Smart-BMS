from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_cluster_stats(telemetry_windowed: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per cluster."""
    if "cluster_label" not in telemetry_windowed.columns:
        raise ValueError("telemetry_windowed must contain 'cluster_label' column.")

    df = telemetry_windowed.copy()

    agg_dict = {}
    if "slip_ratio" in df.columns:
        agg_dict["slip_ratio"] = "mean"
    if "rms_accel_z" in df.columns:
        agg_dict["rms_accel_z"] = "mean"
    if "mean_motor_current" in df.columns:
        agg_dict["mean_motor_current"] = "mean"
    if "speed_drop" in df.columns:
        agg_dict["speed_drop"] = "mean"

    if not agg_dict:
        # Fallback using available numeric columns
        agg_dict = {
            "mean_speed": "mean",
            "mean_accel_z": "mean",
        }

    stats = df.groupby("cluster_label").agg(agg_dict)
    stats = stats.reset_index()
    return stats


def infer_soil_type_from_stats(stats_row: pd.Series) -> str:
    """Heuristic mapping from cluster stats to soil type label."""
    slip = stats_row.get("slip_ratio", np.nan)
    rms_vib = stats_row.get("rms_accel_z", np.nan)
    current = stats_row.get("mean_motor_current", np.nan)
    speed_drop = stats_row.get("speed_drop", np.nan)

    # Use NaN-safe defaults
    slip = 0.0 if np.isnan(slip) else slip
    rms_vib = 0.0 if np.isnan(rms_vib) else rms_vib
    current = 0.0 if np.isnan(current) else current
    speed_drop = 0.0 if np.isnan(speed_drop) else speed_drop

    # Example heuristic rules
    if slip > 0.25 and rms_vib < 0.5:
        return "sandy"
    if current > 0.7 and speed_drop > 0.2:
        return "clay"
    if rms_vib > 0.8 and slip < 0.15:
        return "gravel"
    return "loam"


def map_clusters_to_soil_types(cluster_stats_df: pd.DataFrame) -> Dict[int, str]:
    """Return mapping {cluster_label -> inferred_soil_type}."""
    mapping: Dict[int, str] = {}
    for _, row in cluster_stats_df.iterrows():
        cluster = int(row["cluster_label"])
        mapping[cluster] = infer_soil_type_from_stats(row)
    return mapping


def apply_cluster_soil_mapping(
    telemetry_windowed: pd.DataFrame, cluster_to_soil: Dict[int, str]
) -> pd.DataFrame:
    """Add 'soil_type_cluster' column to telemetry_windowed."""
    df = telemetry_windowed.copy()
    df["soil_type_cluster"] = df["cluster_label"].map(cluster_to_soil)
    return df

