"""
app.py - Streamlit frontend for the Medical Diagnosis Assistant.

Flow:
  1. User checks off symptoms grouped by category.
  2. RandomForest classifier (src/predict.py) ranks candidate diseases.
  3. Groq LLM (src/llm_explainer.py) turns the top prediction into a
     plain-language explanation, self-care tips, and red-flag warnings.
  4. Everything is rendered with a persistent medical disclaimer.
"""

import json

import plotly.express as px
import streamlit as st

from src import config
from src.llm_explainer import explain_prediction
from src.predict import predict_top_k

st.set_page_config(
    page_title="Medical Diagnosis Assistant",
    page_icon="🩺",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load symptom list and group into categories for a friendlier UI
# ---------------------------------------------------------------------------
with open(config.SYMPTOMS_PATH) as f:
    ALL_SYMPTOMS = json.load(f)

CATEGORIES = {
    "General": ["fever", "high_fever", "chills", "fatigue", "weakness",
                "body_ache", "headache", "dizziness", "sweating"],
    "Respiratory": ["cough", "persistent_cough", "sore_throat", "runny_nose",
                    "nasal_congestion", "sneezing", "shortness_of_breath",
                    "wheezing", "chest_pain", "difficulty_breathing",
                    "loss_of_taste_smell"],
    "Digestive": ["nausea", "vomiting", "diarrhea", "abdominal_pain",
                  "abdominal_cramps", "loss_of_appetite", "bloating",
                  "constipation", "heartburn", "acid_reflux", "jaundice"],
    "Urinary / Metabolic": ["burning_urination", "frequent_urination",
                             "lower_abdominal_pain", "excessive_thirst",
                             "excessive_hunger", "blurred_vision",
                             "unexplained_weight_loss", "weight_gain"],
    "Cardiovascular": ["palpitations", "rapid_heartbeat",
                        "high_blood_pressure_reading"],
    "Hormonal": ["cold_intolerance", "heat_intolerance", "hair_loss",
                 "tremors"],
    "Skin": ["skin_rash", "itching", "blisters", "rose_spots",
             "pale_skin", "easy_bruising"],
    "Musculoskeletal": ["joint_pain", "joint_stiffness", "joint_swelling",
                         "muscle_pain"],
    "Eyes / Face": ["watery_eyes", "itchy_eyes", "facial_pain"],
}


def label(symptom: str) -> str:
    return symptom.replace("_", " ").capitalize()


# ---------------------------------------------------------------------------
# Sidebar: about + model info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.write(
        "A hybrid AI system: a trained **RandomForest classifier** predicts "
        "likely conditions from symptoms, and a **Groq-hosted LLM** "
        "(Llama 3.3) explains the result in plain language."
    )
    try:
        with open(config.METRICS_PATH) as f:
            metrics = json.load(f)
        st.metric("Model test accuracy", f"{metrics['accuracy']*100:.1f}%")
        st.metric("F1 score (macro)", f"{metrics['f1_macro']*100:.1f}%")
        st.caption(
            f"Trained on {metrics['n_train']} samples across "
            f"{metrics['n_classes']} conditions."
        )
    except FileNotFoundError:
        st.warning("Model not trained yet. Run `python -m src.train_model`.")

    st.divider()
    st.caption(
        "Built as a portfolio project demonstrating an end-to-end ML + LLM "
        "pipeline: data generation → model training → inference → LLM "
        "explanation layer → Streamlit UI."
    )

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🩺 Medical Diagnosis Assistant")
st.warning(
    "⚠️ **This is an educational demo, not a medical device.** It does not "
    "replace professional medical advice, diagnosis, or treatment. If you "
    "are experiencing a medical emergency, call your local emergency "
    "number immediately.",
    icon="⚠️",
)

st.subheader("Select your symptoms")
selected = []
cols_per_row = 3
for category, symptoms in CATEGORIES.items():
    st.markdown(f"**{category}**")
    cols = st.columns(cols_per_row)
    for i, symptom in enumerate(symptoms):
        with cols[i % cols_per_row]:
            if st.checkbox(label(symptom), key=symptom):
                selected.append(symptom)

st.divider()

analyze_clicked = st.button("🔍 Analyze Symptoms", type="primary", disabled=len(selected) == 0)
if len(selected) == 0:
    st.caption("Select at least one symptom to enable analysis.")

if analyze_clicked:
    with st.spinner("Running classifier..."):
        ranked = predict_top_k(selected, k=5)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Model prediction")
        top_disease, top_prob = ranked[0]
        st.success(f"**Most likely: {top_disease}** ({top_prob:.1%} confidence)")

        fig = px.bar(
            x=[p for _, p in ranked],
            y=[d for d, _ in ranked],
            orientation="h",
            labels={"x": "Probability", "y": "Condition"},
            title="Top candidate conditions",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("AI-generated explanation")
        try:
            with st.spinner("Asking the LLM to explain the result..."):
                explanation = explain_prediction(selected, ranked)

            st.markdown(f"**{explanation.get('summary', '')}**")
            st.write(explanation.get("about_condition", ""))

            if explanation.get("why_this_prediction"):
                st.markdown("**Why this prediction:**")
                st.write(explanation["why_this_prediction"])

            if explanation.get("self_care_tips"):
                st.markdown("**Self-care tips:**")
                for tip in explanation["self_care_tips"]:
                    st.markdown(f"- {tip}")

            if explanation.get("see_a_doctor_if"):
                st.markdown("**🚩 See a doctor if:**")
                for flag in explanation["see_a_doctor_if"]:
                    st.markdown(f"- {flag}")

            st.info(explanation.get("disclaimer", ""), icon="ℹ️")

        except RuntimeError as e:
            st.error(str(e))
            st.caption(
                "The ML prediction above still works without an API key -- "
                "only this explanation layer requires GROQ_API_KEY."
            )

    with st.expander("Show all selected symptoms"):
        st.write(", ".join(label(s) for s in selected))
