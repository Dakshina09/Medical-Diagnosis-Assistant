"""
Basic unit tests for the prediction module.
Run with: pytest
(Requires models/model.joblib to exist -- run `python -m src.train_model` first.)
"""

import pytest

from src.predict import predict_top_k


def test_predict_returns_k_results():
    results = predict_top_k(["fever", "cough"], k=5)
    assert len(results) == 5


def test_predict_probabilities_sum_reasonably():
    results = predict_top_k(["fever", "cough"], k=25)
    total = sum(p for _, p in results)
    assert 0.99 <= total <= 1.01


def test_predict_flu_like_symptoms():
    results = predict_top_k(
        ["high_fever", "chills", "body_ache", "fatigue"], k=1
    )
    top_disease, prob = results[0]
    assert top_disease == "Influenza (Flu)"
    assert prob > 0.5


def test_predict_handles_empty_symptoms():
    results = predict_top_k([], k=3)
    assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
