# nli_scoring.py

from transformers.pipelines import pipeline
from typing import Any


_nli = None
def _get_nli():
    global _nli
    if _nli is None:
        _nli = pipeline("text-classification", model="roberta-large-mnli")
    return _nli

def score_nli(question: str, answer: str) -> dict[str, float]:
    nli_pipeline = _get_nli()
    outputs = nli_pipeline({"text": question, "text_pair": answer}, top_k=None)

    # Hugging Face returns a list of dicts like:
    # [{'label': 'ENTAILMENT', 'score': 0.9}, {'label': 'NEUTRAL', 'score': 0.05}, {'label': 'CONTRADICTION', 'score': 0.05}]
    if not isinstance(outputs, list):
        return {"ENTAILMENT": 0.0, "NEUTRAL": 0.0, "CONTRADICTION": 0.0}

    scores = {item["label"]: float(item["score"]) for item in outputs if isinstance(item, dict)}
    return scores

def score_answer_vs_bta(bta_text: str, cand_answer: str) -> dict[str, float]:
    """
    Score candidate answer vs. best truthful answer (BTA) using NLI model.
    Returns dict with keys: "ENTAILMENT", "NEUTRAL", "CONTRADICTION"
    """
    return score_nli(bta_text, cand_answer)
