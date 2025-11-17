from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

# garantir acesso ao pacote quando rodar como script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soil_dataset import config


def main() -> None:
    path = config.CLUSTERED_TELEMETRY_PATH
    print(f"Carregando janelas de telemetria com clusters de {path} ...")
    df = pd.read_csv(path)

    if "cluster_label" not in df.columns:
        raise ValueError("cluster_label não encontrado em telemetry_windowed_with_clusters.csv.")

    # escolher features para plotar
    feat_x, feat_y = "mean_speed", "mean_accel_z"
    if feat_x not in df.columns or feat_y not in df.columns:
        numeric_cols = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and c not in {"cluster_label", "sample_id"}
        ]
        if len(numeric_cols) < 2:
            raise ValueError("Não há features numéricas suficientes para plotar.")
        feat_x, feat_y = numeric_cols[0], numeric_cols[1]

    out_path = config.PROCESSED_DATA_DIR / "clusters_scatter.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        df[feat_x],
        df[feat_y],
        c=df["cluster_label"],
        cmap="tab10",
        s=8,
        alpha=0.5,
    )
    plt.xlabel(feat_x)
    plt.ylabel(feat_y)
    plt.title(f"Clusters K={config.N_CLUSTERS} em espaço de features")
    cbar = plt.colorbar(scatter)
    cbar.set_label("cluster_label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    print(f"Gráfico de clusters salvo em: {out_path}")


if __name__ == "__main__":
    main()

