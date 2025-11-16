from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Input paths
TELEMETRY_PATH = PROCESSED_DATA_DIR / "telemetry.csv"

# Output paths (sempre em processed/)
OUTPUT_DATASET_PATH = PROCESSED_DATA_DIR / "final_soil_telemetry_dataset.csv"
CLUSTERED_TELEMETRY_PATH = PROCESSED_DATA_DIR / "telemetry_windowed_with_clusters.csv"
CLASSIFIER_PATH = MODELS_DIR / "soil_classifier.pkl"

# General configuration
RANDOM_SEED = 42

# Telemetry processing
WINDOW_SIZE = 128  # number of raw samples per window

# Dimensionality reduction
PCA_COMPONENTS = 4
