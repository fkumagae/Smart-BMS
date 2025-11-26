from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_cluster_stats(telemetry_windowed: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per cluster."""
    if "cluster_label" not in telemetry_windowed.columns:
        raise ValueError("telemetry_windowed must contain 'cluster_label' column.")

    df = telemetry_windowed.copy()

    # Recolhe estatísticas relevantes das colunas disponíveis.
    agg_dict = {
        col: "mean"
        for col in [
            "slip_ratio",
            "rms_accel_z",
            "mean_motor_current",
            "speed_drop",
            "mean_speed",
            "std_speed",
            "mean_accel_z",
            "std_accel_z",
            "mean_gyro_y",
            "std_gyro_y",
        ]
        if col in df.columns
    }
    if not agg_dict:
        raise ValueError("Nenhuma coluna numérica conhecida encontrada para estatísticas de cluster.")

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


def infer_operation_mode_from_stats(stats_row: pd.Series) -> str:
    """Heurística para modo de operação com base no cluster."""
    speed = stats_row.get("mean_speed", np.nan)
    vib = stats_row.get("rms_accel_z", np.nan)
    speed_drop = stats_row.get("speed_drop", np.nan)

    # defaults NaN-safe
    speed = 0.0 if np.isnan(speed) else speed
    vib = 0.0 if np.isnan(vib) else vib
    speed_drop = 0.0 if np.isnan(speed_drop) else speed_drop

    # thresholds a partir dos clusters atuais (~2.5-4.2 m/s e rms 10-12.6)
    speed_level = (
        "slow" if speed < 3.0 else
        "cruise" if speed < 4.0 else
        "fast"
    )
    vib_level = (
        "smooth" if vib < 10.8 else
        "medium" if vib < 12.0 else
        "rough"
    )
    slope_hint = "descent_brake" if speed_drop < -0.02 else "climb_push" if speed_drop > 0.04 else "flat"

    return f"{speed_level}_{vib_level}_{slope_hint}"


def map_clusters_to_soil_types(cluster_stats_df: pd.DataFrame) -> Dict[int, str]:
    """Return mapping {cluster_label -> inferred_soil_type}."""
    mapping: Dict[int, str] = {}
    for _, row in cluster_stats_df.iterrows():
        cluster = int(row["cluster_label"])
        mapping[cluster] = infer_soil_type_from_stats(row)
    return mapping


def map_clusters_to_operation_modes(cluster_stats_df: pd.DataFrame) -> Dict[int, str]:
    """Return mapping {cluster_label -> inferred_operation_mode}."""
    mapping: Dict[int, str] = {}
    for _, row in cluster_stats_df.iterrows():
        cluster = int(row["cluster_label"])
        mapping[cluster] = infer_operation_mode_from_stats(row)
    return mapping


def apply_cluster_soil_mapping(
    telemetry_windowed: pd.DataFrame, cluster_to_soil: Dict[int, str]
) -> pd.DataFrame:
    """Add 'soil_type_cluster' column to telemetry_windowed."""
    df = telemetry_windowed.copy()
    df["soil_type_cluster"] = df["cluster_label"].map(cluster_to_soil)
    return df
