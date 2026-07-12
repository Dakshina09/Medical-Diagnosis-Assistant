# 🩺 Medical Diagnosis Assistant

A hybrid **ML + LLM** symptom checker built as an AI Engineer portfolio project.

A trained classifier makes the actual prediction from selected symptoms; a
Groq-hosted LLM (Llama 3.3) then explains that prediction in plain,
patient-friendly language. The two are kept separate on purpose — the
model's probabilities are always visible and auditable, and the LLM's job
is only to add clear explanation on top, not to invent a diagnosis.

> ⚠️ **Disclaimer:** This is an educational demo, not a medical device. It
> does not replace professional medical advice, diagnosis, or treatment.

## Architecture

```
Symptoms (Streamlit checklist)
        │
        ▼
RandomForestClassifier  ──► ranked disease probabilities
        │
        ▼
Groq LLM (Llama 3.3)  ──►  plain-language explanation, self-care tips,
                            red-flag warnings, disclaimer
        │
        ▼
Streamlit UI (charts + explanation panel)
```

## Features

- **Trained ML classifier** (RandomForest, scikit-learn) predicting across
  25 common conditions from 59 symptoms, ~88% test accuracy / 0.88 macro F1.
- **Synthetic but medically-informed dataset generator** (`data/generate_dataset.py`)
  that encodes real symptom-disease association patterns with realistic
  noise, rather than depending on an external download.
- **LLM explanation layer** (`src/llm_explainer.py`) using Groq's fast Llama
  3.3 API, prompted to return structured JSON: summary, condition
  description, why-this-prediction reasoning, self-care tips, and red-flag
  "see a doctor if" warnings — kept separate from the diagnosis itself for
  auditability.
- **Interactive Streamlit UI**: symptoms grouped into categories
  (General, Respiratory, Digestive, Cardiovascular, etc.), a probability bar
  chart of the top candidate conditions (Plotly), and a persistent medical
  disclaimer banner.
- **Graceful degradation**: the ML prediction still works with no API key;
  only the LLM explanation panel requires `GROQ_API_KEY`.
- **Model introspection**: saved accuracy/F1/per-class metrics and feature
  importances (`models/metrics.json`, `models/feature_importance.json`),
  surfaced in the sidebar.
- **Unit tests** (pytest) covering the prediction pipeline.
- Clean, modular structure (`data/`, `src/`, `models/`, `tests/`) with a
  config module and `.env`-based secrets — no hardcoded keys.

## Project structure

```
medical-diagnosis-assistant/
├── app.py                     # Streamlit UI
├── data/
│   ├── generate_dataset.py    # builds dataset.csv, symptoms.json, diseases.json
│   ├── dataset.csv            # generated training data (5,500 rows)
│   ├── symptoms.json
│   └── diseases.json
├── src/
│   ├── config.py              # paths + env vars
│   ├── train_model.py         # trains + evaluates the RandomForest
│   ├── predict.py             # loads model, exposes predict_top_k()
│   └── llm_explainer.py       # Groq API call + structured prompt
├── models/                    # generated: model.joblib, metrics.json, feature_importance.json
├── tests/
│   └── test_predict.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup (VS Code)

1. **Open the folder in VS Code**: `File > Open Folder...` and select
   `medical-diagnosis-assistant/`.

2. **Create a virtual environment** (VS Code will prompt you to select it
   as the interpreter — do so):
   ```bash
   python -m venv .venv
   # Windows (Git Bash):
   source .venv/Scripts/activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Groq API key** (free at https://console.groq.com/keys):
   ```bash
   cp .env.example .env
   # then edit .env and paste your key
   ```

5. **Generate the dataset and train the model** (already included in this
   handoff, but `data/dataset.csv` and `models/*.joblib` are git-ignored on
   purpose -- generated artifacts shouldn't live in version control. Anyone
   cloning the repo regenerates them locally):
   ```bash
   python data/generate_dataset.py
   python -m src.train_model
   ```

6. **Run the app**:
   ```bash
   streamlit run app.py
   ```
   It opens at `http://localhost:8501`.

7. **(Optional) Run tests**:
   ```bash
   pytest tests/ -v
   ```

## Talking points for interviews

- Why a hybrid architecture instead of "just ask an LLM": determinism,
  auditability (you can inspect exactly which symptoms drove the
  prediction via feature importances), and cost/latency (classifier
  inference is instant and free; the LLM call is only for framing).
- Why synthetic data generation was chosen over scraping/downloading a
  dataset, and how the noise-injection design avoids the model simply
  memorizing exact symptom sets.
- The separation of concerns between `src/predict.py` (deterministic ML)
  and `src/llm_explainer.py` (LLM framing), and how that boundary makes
  the system safer and easier to test.
- Possible extensions: swap RandomForest for gradient boosting (XGBoost/
  LightGBM) and compare metrics; add a real-world dataset with proper
  licensing; add multi-turn follow-up questions before the final
  prediction; add user auth + history with a database.
