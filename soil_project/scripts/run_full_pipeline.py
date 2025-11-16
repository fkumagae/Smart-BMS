from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = project_root / "scripts"

    print("Running build_dataset.py ...")
    result = subprocess.run(
        [sys.executable, str(scripts_dir / "build_dataset.py")],
        cwd=project_root,
        check=True,
    )
    print(f"build_dataset.py finished with return code {result.returncode}")

    print("\nRunning train_model.py ...")
    result = subprocess.run(
        [sys.executable, str(scripts_dir / "train_model.py")],
        cwd=project_root,
        check=True,
    )
    print(f"train_model.py finished with return code {result.returncode}")


if __name__ == "__main__":
    main()

