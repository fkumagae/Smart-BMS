import os
from pathlib import Path

import pandas as pd


# Raiz do projeto (pasta soil_project)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raiz dos dados brutos de telemetria (ajuste se precisar)
ROOT = PROJECT_ROOT / "data" / "raw" / "SensorData"


def _add_time_sec(df: pd.DataFrame) -> pd.DataFrame:
    """Cria coluna de tempo contínuo em segundos a partir de utc_s + utc_ms."""
    df = df.copy()
    if "utc_s (s)" in df.columns and "utc_ms (ms)" in df.columns:
        df["time_sec"] = df["utc_s (s)"] + df["utc_ms (ms)"] / 1000.0
    elif "utc_s (s)" in df.columns:
        df["time_sec"] = df["utc_s (s)"].astype(float)
    else:
        raise ValueError("Não encontrei colunas de tempo (utc_s/utc_ms) no CSV.")
    return df


def load_session(session_path: str | Path) -> pd.DataFrame:
    """Carrega e unifica os CSVs de uma sessão em um único dataframe.

    Estratégia para terramecânica:
    - Base de alta frequência: acelerômetro + giroscópio.
    - Variáveis “lentas”: velocidade, altitude e distância do record.csv.
    """
    session_path = Path(session_path)

    accel_path = session_path / "accelerometer_calibrated_split.csv"
    gyro_path = session_path / "gyroscope_calibrated_split.csv"
    record_path = session_path / "record.csv"

    if not accel_path.exists() or not gyro_path.exists() or not record_path.exists():
        raise FileNotFoundError(f"Arquivos esperados não encontrados em {session_path}")

    df_accel = pd.read_csv(accel_path)
    df_gyro = pd.read_csv(gyro_path)
    df_record = pd.read_csv(record_path)

    # 1) Unir acelerômetro + giroscópio por utc_s/utc_ms (mesma taxa)
    keys = ["utc_s (s)", "utc_ms (ms)"]
    for k in keys:
        if k not in df_accel.columns or k not in df_gyro.columns:
            raise ValueError(f"Coluna de tempo {k} ausente em accel ou gyro.")

    df_imu = pd.merge(df_accel, df_gyro, on=keys, suffixes=("_accel", "_gyro"))

    # 2) Criar eixo de tempo contínuo e unir com record (1 Hz) por “nearest”
    df_imu = _add_time_sec(df_imu).sort_values("time_sec")
    df_record = _add_time_sec(df_record).sort_values("time_sec")

    record_cols = [
        "time_sec",
        "distance (m)",
        "enhanced_speed (m/s)",
        "enhanced_altitude (m)",
    ]
    record_cols = [c for c in record_cols if c in df_record.columns]

    df_all = pd.merge_asof(
        df_imu,
        df_record[record_cols],
        on="time_sec",
        direction="nearest",
    )

    # 3) Alias de velocidade padrão para a pipeline futura
    if "enhanced_speed (m/s)" in df_all.columns:
        df_all["speed"] = df_all["enhanced_speed (m/s)"]

    return df_all


def merge_all_sessions() -> pd.DataFrame:
    sessions = [f for f in os.listdir(ROOT) if (ROOT / f).is_dir()]

    dfs = []

    for sess in sessions:
        session_path = ROOT / sess
        print(f"Carregando sessão: {sess}")

        try:
            df_sess = load_session(session_path)
            df_sess["session"] = sess
            dfs.append(df_sess)
        except Exception as e:
            print(f"Erro ao carregar {sess}: {e}")

    if not dfs:
        raise RuntimeError(f"Nenhuma sessão válida encontrada em {ROOT}")

    df_all = pd.concat(dfs, ignore_index=True)

    # Salvar dataset unificado de telemetria em data/processed
    out = PROJECT_ROOT / "data" / "processed" / "telemetry.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out, index=False)

    print(f"Dataset final salvo em {out}")
    print("Total de linhas:", len(df_all))

    return df_all


if __name__ == "__main__":
    merge_all_sessions()
