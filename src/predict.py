"""
predict.py

Loads the trained model and exposes a simple predict_top_k() function
that turns a list of selected symptoms into ranked disease probabilities.
"""

from functools import lru_cache
from typing import List, Tuple

import joblib
import pandas as pd

from src import config


@lru_cache(maxsize=1)
def load_bundle():
    bundle = joblib.load(config.MODEL_PATH)
    return bundle["model"], bundle["symptoms"], bundle["classes"]


def predict_top_k(selected_symptoms: List[str], k: int = 5) -> List[Tuple[str, float]]:
    """
    selected_symptoms: list of symptom names (must match data/symptoms.json)
    returns: list of (disease, probability) tuples, sorted descending, length k
    """
    model, symptoms, _classes = load_bundle()

    row = {s: (1 if s in selected_symptoms else 0) for s in symptoms}
    X = pd.DataFrame([row], columns=symptoms)

    proba = model.predict_proba(X)[0]
    ranked = sorted(zip(model.classes_, proba), key=lambda x: x[1], reverse=True)
    return ranked[:k]


if __name__ == "__main__":
    demo_symptoms = ["high_fever", "chills", "body_ache", "fatigue"]
    for disease, prob in predict_top_k(demo_symptoms):
        print(f"{disease:30s} {prob:.3f}")
