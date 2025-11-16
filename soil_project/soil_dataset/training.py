from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from joblib import dump

from .config import CLASSIFIER_PATH, RANDOM_SEED


def _feature_target_split(df_final: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df_final.copy()
    if "soil_type" not in df.columns:
        raise ValueError("df_final must contain 'soil_type' column.")

    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            "soil_type",
            "sample_id",
            "cluster_label",
        }
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols]
    y = df["soil_type"].astype(str)
    return X, y


def train_soil_classifier(df_final: pd.DataFrame, save_model: bool = True) -> RandomForestClassifier:
    """Train a RandomForest classifier to predict soil_type from features."""
    X, y = _feature_target_split(df_final)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y if len(y.unique()) > 1 else None
    )

    clf = RandomForestClassifier(random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(report)

    if save_model:
        CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
        dump(clf, CLASSIFIER_PATH)
        print(f"Saved trained model to {CLASSIFIER_PATH}")

    return clf

