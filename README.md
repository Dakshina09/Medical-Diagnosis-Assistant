#  Medical Diagnosis Assistant



** Live demo: [medical-diagnosis-assistant-rkemrzy3nweuy9fal6oxzt.streamlit.app](https://medical-diagnosis-assistant-rkemrzy3nweuy9fal6oxzt.streamlit.app/)**

A hybrid **ML + LLM** symptom checker built as an AI Engineer portfolio project.

## The problem

Most "AI symptom checker" demos just pipe user input straight into an LLM and
call it a diagnosis — which is fast to build but impossible to audit or
trust: you can't tell *why* the model said what it said, and the LLM is free
to hallucinate a condition that doesn't fit the symptoms at all.

## The approach

This project separates the two concerns instead of collapsing them into one
LLM call:

- A **trained classifier** (scikit-learn RandomForest) makes the actual
  prediction from selected symptoms. Its output is deterministic and
  inspectable — you can always see exactly which symptoms drove which
  probability.
- A **Groq-hosted LLM** (Llama 3.3) takes that prediction and turns it into
  plain-language explanation, self-care tips, and red-flag warnings. It
  never gets to invent the diagnosis itself — only to explain one that
  already came from the model.

> ⚠️ **Disclaimer:** This is an educational demo, not a medical device. It
> does not replace professional medical advice, diagnosis, or treatment.

## Tech stack

| Layer               | Technology                                   |
|---------------------|-----------------------------------------------|
| ML model            | scikit-learn (RandomForestClassifier)          |
| LLM explanation      | Groq API — Llama 3.3 70B                       |
| Frontend             | Streamlit + Plotly                             |
| Data                 | Custom-generated symptom-disease dataset       |
| Testing              | pytest                                         |
| Deployment           | Streamlit Community Cloud                      |

## How it works

1. User checks off symptoms in the UI, grouped by category (General,
   Respiratory, Digestive, Cardiovascular, etc.).
2. The RandomForest classifier scores all 25 conditions and returns the
   top 5 ranked by probability.
3. The top prediction + differentials + selected symptoms are passed to
   Groq's Llama 3.3, prompted to return **structured JSON only** (summary,
   condition description, reasoning, self-care tips, red flags,
   disclaimer) — never free-form prose that could drift off-topic.
4. Streamlit renders a probability bar chart alongside the LLM's
   explanation panel, with a persistent medical disclaimer banner.

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

## Results

| Metric                  | Value |
|--------------------------|-------|
| Test accuracy             | ~87.5% |
| F1 score (macro)          | ~0.87 |
| Conditions covered         | 25 |
| Symptoms tracked           | 59 |
| Training samples           | 5,500 |

## Features

- **Trained ML classifier** (RandomForest, scikit-learn) — see Results above.
- **Synthetic but medically-informed dataset generator** (`data/generate_dataset.py`)
  that encodes real symptom-disease association patterns with realistic
  noise, rather than depending on an external download.
- **LLM explanation layer** (`src/llm_explainer.py`) prompted to return
  structured JSON only (no free-form prose): summary, condition
  description, why-this-prediction reasoning, self-care tips, and red-flag
  "see a doctor if" warnings.
- **Interactive Streamlit UI**: symptoms grouped into categories, a
  probability bar chart of top candidates (Plotly), and a persistent
  medical disclaimer banner.
- **Graceful degradation**: the ML prediction still works with no API key;
  only the LLM explanation panel requires `GROQ_API_KEY`.
- **Model introspection**: saved accuracy/F1/per-class metrics and feature
  importances, surfaced live in the sidebar.
- **Unit tests** (pytest) covering the prediction pipeline.
- Clean, modular structure (`data/`, `src/`, `models/`, `tests/`) with a
  config module and environment-based secrets (`.env` locally,
  `st.secrets` on Streamlit Cloud) — no hardcoded keys.

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

> No setup needed to just try it — see the live demo link at the top. The
> steps below are for running/editing it locally.

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

