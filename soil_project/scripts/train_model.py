#
# 
# 
# AINDA NAO TESTADO!!! 
# 
# 



from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# garante que o pacote soil_dataset seja encontrado
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
