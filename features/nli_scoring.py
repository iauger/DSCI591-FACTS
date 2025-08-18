# nli_scoring.py

from transformers.pipelines import pipeline
from typing import Any


nli_pipeline = pipeline("text-classification", model="roberta-large-mnli")

def score_nli(question: str, answer: str) -> dict[str, float]:
    outputs = nli_pipeline({"text": question, "text_pair": answer}, top_k=None)

    # Hugging Face returns a list of dicts like:
    # [{'label': 'ENTAILMENT', 'score': 0.9}, {'label': 'NEUTRAL', 'score': 0.05}, {'label': 'CONTRADICTION', 'score': 0.05}]
    if not isinstance(outputs, list):
        return {"ENTAILMENT": 0.0, "NEUTRAL": 0.0, "CONTRADICTION": 0.0}

    scores = {item["label"]: float(item["score"]) for item in outputs if isinstance(item, dict)}
    return scores
