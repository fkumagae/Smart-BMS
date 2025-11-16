from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


def load_telemetry(path: PathLike) -> pd.DataFrame:
    """Load telemetry CSV into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found at {path}")
    df = pd.read_csv(path)
    return df


def load_soil(path: PathLike) -> pd.DataFrame:
    """Load soil CSV into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Soil file not found at {path}")
    df = pd.read_csv(path)
    return df

