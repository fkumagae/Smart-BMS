from __future__ import annotations

from pathlib import Path
import sys

# garante que o pacote soil_dataset seja encontrado
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soil_dataset import config
from soil_dataset.data_loading import load_telemetry
from soil_dataset.telemetry_processing import create_windows_by_index
from soil_dataset.clustering import cluster_telemetry
from soil_dataset.cluster_to_soil_mapping import (
    apply_cluster_soil_mapping,
    compute_cluster_stats,
    map_clusters_to_soil_types,
    map_clusters_to_operation_modes,
)


def main() -> None:
    print("Loading telemetry dataset...")
    telemetry_df = load_telemetry(config.TELEMETRY_PATH)

    print("\nTelemetry head():")
    print(telemetry_df.head())
    print("\nTelemetry info():")
    print(telemetry_df.info())

    # Telemetry processing
    print("\nCreating telemetry windows and features...")
    telemetry_windowed = create_windows_by_index(
        telemetry_df, window_size=config.WINDOW_SIZE
    )

    # Clustering
    print(f"\nClustering telemetry windows with K={config.N_CLUSTERS} ...")
    telemetry_windowed = cluster_telemetry(
        telemetry_windowed, n_clusters=config.N_CLUSTERS
    )

    # Compute cluster stats and map to soil types
    print("\nComputing cluster statistics and mapping to soil types...")
    cluster_stats = compute_cluster_stats(telemetry_windowed)
    cluster_mapping = map_clusters_to_soil_types(cluster_stats)
    mode_mapping = map_clusters_to_operation_modes(cluster_stats)
    print("Cluster -> soil_type mapping:", cluster_mapping)
    print("Cluster -> operation_mode mapping:", mode_mapping)

    telemetry_with_soil = apply_cluster_soil_mapping(
        telemetry_windowed, cluster_mapping
    )
    telemetry_with_soil["operation_mode"] = telemetry_with_soil["cluster_label"].map(mode_mapping)

    # Neste estágio inicial, usamos apenas telemetria + heurísticas
    # terramecânicas: o rótulo final é soil_type derivado de clusters.
    df_final = telemetry_with_soil.rename(
        columns={"soil_type_cluster": "soil_type"}
    )

    cols_order = ["sample_id", "soil_type", "operation_mode"] + [
        c
        for c in df_final.columns
        if c not in {"sample_id", "soil_type", "operation_mode"}
    ]
    df_final = df_final[cols_order]

    # Save intermediate and final datasets (sem embeddings por enquanto)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    telemetry_with_soil.to_csv(config.CLUSTERED_TELEMETRY_PATH, index=False)
    df_final.to_csv(config.OUTPUT_DATASET_PATH, index=False)

    print(f"\nSaved clustered telemetry to {config.CLUSTERED_TELEMETRY_PATH}")
    print(f"Saved final dataset to {config.OUTPUT_DATASET_PATH}")


if __name__ == "__main__":
    main()
