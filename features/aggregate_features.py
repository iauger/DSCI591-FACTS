# aggregate_features.py
from typing import Dict, Iterable, Optional, List, Any
from statistics import mean, stdev
from . import ALL_EXTRACTORS
from .nli_scoring import score_nli

def compute_features(
    text: str,
    use: Optional[Iterable[str]] = None 
) -> Dict[str, float]:
    """
    Compute all features for the given text.
    
    :param text: The input text to analyze.
    :param use: Optional list of feature extractor names to use. If None, all extractors are used.
    :return: A dictionary of computed features.
    """
    selected = ALL_EXTRACTORS if not use else [
        ext for ext in ALL_EXTRACTORS if ext.name in use
    ]
    
    out: Dict[str, float] = {}
    for ext in selected:
        features = ext.compute(text)
        out.update(features)
    return out


def aggregate_scores(scores: List[Dict[str, float]], labels=("ENTAILMENT", "NEUTRAL", "CONTRADICTION")) -> Dict[str, float]:
    """
    Aggregate a list of NLI score dicts into min/mean/max/std for each label.
    """
    agg = {}
    for label in labels:
        vals = [s[label] for s in scores if label in s]
        if vals:
            agg[f"nli_pair_{label.lower()}_min"] = min(vals)
            agg[f"nli_pair_{label.lower()}_max"] = max(vals)
            agg[f"nli_pair_{label.lower()}_mean"] = mean(vals)
            agg[f"nli_pair_{label.lower()}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        else:
            agg[f"nli_pair_{label.lower()}_min"] = 0.0
            agg[f"nli_pair_{label.lower()}_max"] = 0.0
            agg[f"nli_pair_{label.lower()}_mean"] = 0.0
            agg[f"nli_pair_{label.lower()}_std"] = 0.0
    return agg


def process_answer(qid: int,
    question: str,
    answer: str,
    all_answers: List[str],
    *,
    is_true: bool,
    is_best: bool,
) -> Dict[str, Any]:

    """
    Compute features for a single answer including:
      - text-based features
      - NLI with the question
      - Aggregated NLI scores vs. all other answers for this question
    """
    feats = compute_features(answer)
    
    row = {
        "qid": qid,
        "question": question,
        "answer": answer,
        "true_answer": bool(is_true),
        "best_true_answer": bool(is_true and is_best),
        "best_false_answer": bool((not is_true) and is_best),
        "group_answer_count": float(len(all_answers)),
    }
    
    row.update(feats)
    # flatten basic text features
    for k, v in feats.items():
        row[k] = v

    # NLI question:answer
    for label, score in score_nli(question, answer).items():
        row[f"nli_q_{label.lower()}"] = score

    # NLI answer:answer (all other answers)
    pair_scores = []
    for other in all_answers:
        if other.strip() == answer.strip():
            continue
        pair_scores.append(score_nli(answer, other))

    # aggregate pairwise results
    agg = aggregate_scores(pair_scores)
    row.update(agg)

    return row