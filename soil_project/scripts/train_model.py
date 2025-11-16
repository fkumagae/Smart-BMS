from __future__ import annotations

import pandas as pd

from soil_dataset import config
from soil_dataset.training import train_soil_classifier


def main() -> None:
    print(f"Loading final dataset from {config.OUTPUT_DATASET_PATH} ...")
    df_final = pd.read_csv(config.OUTPUT_DATASET_PATH)
    print("Dataset head():")
    print(df_final.head())

    print("\nTraining soil classifier...")
    train_soil_classifier(df_final, save_model=True)


if __name__ == "__main__":
    main()

