"""
generate_dataset.py

Builds a synthetic-but-medically-informed symptom -> disease dataset.

Why synthetic: there's no reliably licensed, always-available public dataset
we can pull at build time. Instead we encode real symptom-disease associations
(each disease has a set of "core" symptoms with a high probability of
appearing, and a small chance of unrelated/noise symptoms), then sample many
patient records from that distribution. This mirrors how the well-known
Kaggle "Disease-Symptom-Prediction" style datasets are structured, and is
transparent + reproducible (see DISCLAIMER in README).

Output: data/dataset.csv  (one row per synthetic patient, binary symptom
columns + a `disease` label column)
"""

import json
import random
from pathlib import Path

import pandas as pd

random.seed(42)

# ---------------------------------------------------------------------------
# 1. Master symptom list
# ---------------------------------------------------------------------------
SYMPTOMS = [
    "fever", "high_fever", "chills", "fatigue", "weakness", "body_ache",
    "headache", "dizziness", "sweating",
    "cough", "persistent_cough", "sore_throat", "runny_nose",
    "nasal_congestion", "sneezing", "shortness_of_breath", "wheezing",
    "chest_pain", "difficulty_breathing", "loss_of_taste_smell",
    "nausea", "vomiting", "diarrhea", "abdominal_pain", "abdominal_cramps",
    "loss_of_appetite", "bloating", "constipation", "heartburn",
    "acid_reflux", "jaundice",
    "burning_urination", "frequent_urination", "lower_abdominal_pain",
    "excessive_thirst", "excessive_hunger", "blurred_vision",
    "unexplained_weight_loss", "weight_gain",
    "palpitations", "rapid_heartbeat", "high_blood_pressure_reading",
    "cold_intolerance", "heat_intolerance", "hair_loss", "tremors",
    "skin_rash", "itching", "blisters", "rose_spots",
    "joint_pain", "joint_stiffness", "joint_swelling", "muscle_pain",
    "pale_skin", "easy_bruising", "watery_eyes", "itchy_eyes", "facial_pain",
]

# ---------------------------------------------------------------------------
# 2. Disease -> {symptom: probability of occurring} core profile.
#    Any symptom not listed gets a small "noise" chance (NOISE_PROB) so the
#    classifier has to learn robust patterns rather than memorize exact sets.
# ---------------------------------------------------------------------------
DISEASE_PROFILES = {
    "Common Cold": {
        "runny_nose": 0.9, "nasal_congestion": 0.85, "sneezing": 0.8,
        "sore_throat": 0.7, "cough": 0.6, "fatigue": 0.4, "headache": 0.3,
    },
    "Influenza (Flu)": {
        "high_fever": 0.85, "chills": 0.75, "body_ache": 0.85,
        "fatigue": 0.8, "headache": 0.6, "cough": 0.6, "sore_throat": 0.4,
    },
    "COVID-19": {
        "fever": 0.7, "cough": 0.75, "fatigue": 0.7,
        "loss_of_taste_smell": 0.6, "shortness_of_breath": 0.4,
        "sore_throat": 0.35, "body_ache": 0.4,
    },
    "Migraine": {
        "headache": 0.95, "nausea": 0.6, "blurred_vision": 0.4,
        "dizziness": 0.3, "fatigue": 0.3,
    },
    "Tension Headache": {
        "headache": 0.9, "fatigue": 0.3, "muscle_pain": 0.3,
    },
    "Gastroenteritis": {
        "diarrhea": 0.9, "vomiting": 0.7, "abdominal_cramps": 0.7,
        "nausea": 0.65, "fever": 0.3, "loss_of_appetite": 0.4,
    },
    "Food Poisoning": {
        "vomiting": 0.85, "diarrhea": 0.8, "abdominal_pain": 0.7,
        "nausea": 0.7, "fever": 0.3,
    },
    "Urinary Tract Infection": {
        "burning_urination": 0.9, "frequent_urination": 0.85,
        "lower_abdominal_pain": 0.6, "fever": 0.25,
    },
    "Type 2 Diabetes": {
        "excessive_thirst": 0.85, "excessive_hunger": 0.6,
        "frequent_urination": 0.75, "unexplained_weight_loss": 0.4,
        "blurred_vision": 0.35, "fatigue": 0.4,
    },
    "Hypertension": {
        "headache": 0.4, "dizziness": 0.45, "high_blood_pressure_reading": 0.9,
        "palpitations": 0.3, "blurred_vision": 0.2,
    },
    "Asthma": {
        "wheezing": 0.85, "shortness_of_breath": 0.8, "cough": 0.5,
        "chest_pain": 0.3, "difficulty_breathing": 0.6,
    },
    "Bronchitis": {
        "persistent_cough": 0.85, "chest_pain": 0.3, "fatigue": 0.4,
        "shortness_of_breath": 0.35, "fever": 0.3,
    },
    "Pneumonia": {
        "high_fever": 0.7, "cough": 0.75, "difficulty_breathing": 0.65,
        "chest_pain": 0.55, "fatigue": 0.5, "chills": 0.4,
    },
    "Allergic Rhinitis": {
        "sneezing": 0.85, "itchy_eyes": 0.7, "watery_eyes": 0.65,
        "runny_nose": 0.75, "nasal_congestion": 0.6,
    },
    "Sinusitis": {
        "facial_pain": 0.8, "nasal_congestion": 0.75, "headache": 0.5,
        "runny_nose": 0.4,
    },
    "GERD": {
        "heartburn": 0.9, "acid_reflux": 0.85, "chest_pain": 0.3,
        "bloating": 0.3,
    },
    "Irritable Bowel Syndrome": {
        "abdominal_pain": 0.75, "bloating": 0.65, "constipation": 0.4,
        "diarrhea": 0.4,
    },
    "Anemia": {
        "fatigue": 0.85, "pale_skin": 0.7, "weakness": 0.6,
        "dizziness": 0.4, "shortness_of_breath": 0.3,
    },
    "Hypothyroidism": {
        "cold_intolerance": 0.75, "weight_gain": 0.6, "fatigue": 0.7,
        "hair_loss": 0.4, "constipation": 0.3,
    },
    "Hyperthyroidism": {
        "heat_intolerance": 0.7, "unexplained_weight_loss": 0.6,
        "rapid_heartbeat": 0.6, "tremors": 0.5, "sweating": 0.4,
    },
    "Chickenpox": {
        "skin_rash": 0.9, "itching": 0.8, "blisters": 0.75, "fever": 0.55,
    },
    "Dengue Fever": {
        "high_fever": 0.85, "body_ache": 0.75, "headache": 0.6,
        "skin_rash": 0.35, "easy_bruising": 0.3, "joint_pain": 0.4,
    },
    "Malaria": {
        "high_fever": 0.85, "chills": 0.8, "sweating": 0.65,
        "headache": 0.5, "muscle_pain": 0.4,
    },
    "Typhoid Fever": {
        "high_fever": 0.8, "abdominal_pain": 0.5, "rose_spots": 0.3,
        "loss_of_appetite": 0.5, "weakness": 0.4,
    },
    "Osteoarthritis": {
        "joint_pain": 0.9, "joint_stiffness": 0.8, "joint_swelling": 0.55,
        "muscle_pain": 0.3,
    },
}

NOISE_PROB = 0.02        # chance any unrelated symptom is also present
SAMPLES_PER_DISEASE = 220


def make_row(disease: str, profile: dict) -> dict:
    row = {s: 0 for s in SYMPTOMS}
    for symptom, prob in profile.items():
        row[symptom] = 1 if random.random() < prob else 0
    for symptom in SYMPTOMS:
        if symptom not in profile and random.random() < NOISE_PROB:
            row[symptom] = 1
    row["disease"] = disease
    return row


def main():
    rows = []
    for disease, profile in DISEASE_PROFILES.items():
        for _ in range(SAMPLES_PER_DISEASE):
            rows.append(make_row(disease, profile))

    df = pd.DataFrame(rows)
    # guard against all-zero rows (extremely rare, but re-roll them)
    zero_mask = df[SYMPTOMS].sum(axis=1) == 0
    for idx in df[zero_mask].index:
        core_symptom = random.choice(list(DISEASE_PROFILES[df.loc[idx, "disease"]].keys()))
        df.loc[idx, core_symptom] = 1

    out_dir = Path(__file__).parent
    df.to_csv(out_dir / "dataset.csv", index=False)

    with open(out_dir / "symptoms.json", "w") as f:
        json.dump(SYMPTOMS, f, indent=2)

    with open(out_dir / "diseases.json", "w") as f:
        json.dump(sorted(DISEASE_PROFILES.keys()), f, indent=2)

    print(f"Generated {len(df)} rows across {len(DISEASE_PROFILES)} diseases "
          f"and {len(SYMPTOMS)} symptoms -> {out_dir / 'dataset.csv'}")


if __name__ == "__main__":
    main()
