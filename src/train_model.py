"""
train_model.py

Trains a RandomForestClassifier on the symptom -> disease dataset,
evaluates it, and persists the model + metrics + feature importances.

Run from the project root:
    python -m src.train_model
"""

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src import config


def train():
    df = pd.read_csv(config.DATASET_PATH)

    with open(config.SYMPTOMS_PATH) as f:
        symptoms = json.load(f)

    X = df[symptoms]
    y = df["disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": y.nunique(),
        "classification_report": report,
    }

    joblib.dump(
        {"model": model, "symptoms": symptoms, "classes": list(model.classes_)},
        config.MODEL_PATH,
        compress=3,
    )

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    importances = sorted(
        zip(symptoms, model.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    with open(config.FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(importances, f, indent=2)

    print(f"Accuracy: {accuracy:.4f}  |  F1 (macro): {f1_macro:.4f}")
    print(f"Model saved to {config.MODEL_PATH}")
    print(f"Metrics saved to {config.METRICS_PATH}")


if __name__ == "__main__":
    train()
