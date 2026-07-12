"""
llm_explainer.py

Takes the ML classifier's output (top predicted diseases + probabilities +
the symptoms the user selected) and asks a Groq-hosted LLM to turn that into
a clear, structured, patient-friendly explanation.

The LLM is deliberately NOT used to make the diagnosis itself -- that's the
ML model's job, and keeping the two separated makes the system auditable:
you can always see exactly which features drove the prediction (via
feature_importance.json / the classifier's own probabilities), and the LLM's
role is strictly explanatory framing on top of that.
"""

import json
from typing import List, Tuple

from groq import Groq

from src import config

SYSTEM_PROMPT = """You are a medical information assistant helping explain the \
output of a symptom-checker machine learning model to a general audience.

Rules:
- You are NOT diagnosing the patient. A statistical model has already produced \
the ranked predictions; your job is only to explain them clearly.
- Keep tone calm, plain-language, non-alarming.
- Always include a short list of "red flag" symptoms that would warrant \
urgent/emergency care, if relevant to the top condition.
- Always end by recommending the person consult a licensed medical \
professional for an actual diagnosis.
- Respond ONLY with valid JSON matching this schema, no markdown fences, no \
extra commentary:
{
  "summary": "1-2 sentence plain-language summary of the top prediction",
  "about_condition": "2-3 sentences describing what the top predicted condition is",
  "why_this_prediction": "1-2 sentences connecting the user's selected symptoms to the prediction",
  "self_care_tips": ["short tip 1", "short tip 2", "short tip 3"],
  "see_a_doctor_if": ["red flag symptom or situation 1", "red flag symptom or situation 2"],
  "disclaimer": "a clear one-sentence disclaimer that this is not a medical diagnosis"
}
"""


def _client() -> Groq:
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Groq(api_key=config.GROQ_API_KEY)


def explain_prediction(
    selected_symptoms: List[str],
    ranked_predictions: List[Tuple[str, float]],
) -> dict:
    top_disease, top_prob = ranked_predictions[0]
    differentials = ranked_predictions[1:4]

    user_prompt = f"""
Selected symptoms: {", ".join(selected_symptoms) if selected_symptoms else "none"}

Model's top prediction: {top_disease} (confidence: {top_prob:.1%})

Other differential possibilities considered by the model:
{chr(10).join(f"- {d} ({p:.1%})" for d, p in differentials) if differentials else "none"}

Explain this to the patient following the JSON schema in your instructions.
"""

    client = _client()
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=700,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "summary": "The assistant's explanation could not be parsed.",
            "about_condition": raw,
            "why_this_prediction": "",
            "self_care_tips": [],
            "see_a_doctor_if": ["Please consult a doctor for a proper evaluation."],
            "disclaimer": "This tool does not provide medical diagnoses.",
        }
