from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

# garantir acesso ao pacote quando rodar como script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soil_dataset import config
from soil_dataset.data_loading import load_telemetry
from soil_dataset.telemetry_processing import create_windows_by_index
from soil_dataset.clustering import compute_elbow_inertia, DEFAULT_CLUSTER_FEATURES


def main() -> None:
    print("Carregando telemetry.csv processado...")
    telemetry_df = load_telemetry(config.TELEMETRY_PATH)

    print("Gerando janelas e features para análise de elbow...")
    telemetry_windowed = create_windows_by_index(
        telemetry_df, window_size=config.WINDOW_SIZE
    )

    feature_cols = [c for c in DEFAULT_CLUSTER_FEATURES if c in telemetry_windowed.columns]
    print("Features usadas para clustering:", feature_cols)

    k_range = range(2, 11)
    print(f"Calculando inércia para K em {list(k_range)} ...")
    elbow_df = compute_elbow_inertia(telemetry_windowed, k_range, feature_cols)

    out_path = config.PROCESSED_DATA_DIR / "cluster_elbow.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(elbow_df["k"], elbow_df["inertia"], marker="o")
    plt.xlabel("Número de clusters (K)")
    plt.ylabel("Inércia (soma das distâncias ao centróide)")
    plt.title("Gráfico de Elbow para definição de K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    print("Resultados de inércia por K:")
    print(elbow_df)
    print(f"Gráfico de elbow salvo em: {out_path}")
    print("Escolha um K (olhando o 'cotovelo') e ajuste N_CLUSTERS em soil_dataset/config.py.")


if __name__ == "__main__":
    main()
