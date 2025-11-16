from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


TELEMETRY_REQUIRED_COLS = [
    "timestamp",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "speed",
]


def create_windows_by_index(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Create fixed-size sequential windows and aggregate features per window.

    The function assumes df is ordered by time already.
    """
    if df.empty:
        raise ValueError("Telemetry dataframe is empty.")

    # Ensure required columns exist; create NaN columns if missing
    for col in TELEMETRY_REQUIRED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    n_rows = len(df)
    n_windows = n_rows // window_size
    if n_windows == 0:
        raise ValueError(
            f"Not enough rows ({n_rows}) to create at least one window of size {window_size}."
        )

    records: List[Dict] = []
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        window = df.iloc[start:end]

        rec: Dict = {"sample_id": w}

        # Speed
        rec["mean_speed"] = window["speed"].mean()
        rec["std_speed"] = window["speed"].std()

        # Accelerometer stats
        for axis in ["x", "y", "z"]:
            col = f"accel_{axis}"
            rec[f"mean_{col}"] = window[col].mean()
            rec[f"std_{col}"] = window[col].std()

        # Gyro stats
        for axis in ["x", "y", "z"]:
            col = f"gyro_{axis}"
            rec[f"mean_{col}"] = window[col].mean()
            rec[f"std_{col}"] = window[col].std()

        # Optional features: wheel_rpm, motor_current
        if "wheel_rpm" in df.columns and "speed" in df.columns:
            # very rough slip proxy: normalized variance of wheel_rpm vs speed
            speed_mean = window["speed"].mean()
            rpm_mean = window["wheel_rpm"].mean()
            if speed_mean and not np.isnan(speed_mean):
                rec["slip_ratio"] = max(
                    0.0, (rpm_mean / (speed_mean + 1e-6)) - 1.0
                )
            else:
                rec["slip_ratio"] = np.nan

        if "motor_current" in df.columns:
            rec["mean_motor_current"] = window["motor_current"].mean()
            rec["std_motor_current"] = window["motor_current"].std()

        # Speed drop: first vs last speed in window
        rec["speed_drop"] = (
            window["speed"].iloc[0] - window["speed"].iloc[-1]
            if len(window["speed"]) > 1
            else 0.0
        )

        # RMS vibration from accel_z
        rec["rms_accel_z"] = float(
            np.sqrt(np.nanmean(window["accel_z"] ** 2))
        )

        records.append(rec)

    telemetry_windowed = pd.DataFrame.from_records(records)
    return telemetry_windowed

