from typing import List, Optional
import pandas as pd
from rouge_score import rouge_scorer

def compute_rouge(candidate: str, positive_refs: List[str], leave_one_out: bool = True) -> float:
    """
    Compute ROUGE-L F1 score for a candidate vs. positive references.
    Returns the best ROUGE-L F1 across references.
    """
    if not positive_refs:
        return 0.0

    candidate = str(candidate).strip()
    refs = [str(r).strip() for r in positive_refs if pd.notna(r)]

    if leave_one_out and candidate in refs:
        refs = [r for r in refs if r != candidate]
        if not refs:
            return 0.0

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(candidate, ref)['rougeL'].fmeasure for ref in refs]
    return max(scores) if scores else 0.0

def compute_rouge_anchor(candidate: str,
                         bta_text: Optional[str] = None,
                         bfa_text: Optional[str] = None) -> dict:
    """
    Compute ROUGE-L F1 against best-true and best-false anchors.
    Returns dict with rouge_bta, rouge_bfa, rouge_bta_minus_bfa.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def _score_or_zero(cand, ref):
        if not ref or not cand:
            return 0.0
        return scorer.score(cand, ref)['rougeL'].fmeasure

    rouge_bta = _score_or_zero(candidate, bta_text)
    rouge_bfa = _score_or_zero(candidate, bfa_text)
    return {
        "rouge_bta": rouge_bta,
        "rouge_bfa": rouge_bfa,
        "rouge_bta_minus_bfa": rouge_bta - rouge_bfa
    }


def compute_rouge_contrastive(candidate: str,
                              refs_pos: List[str],
                              refs_neg: Optional[List[str]] = None,
                              leave_one_out: bool = True) -> dict:
    """
    Compute ROUGE-L F1 against positive references and contrast with negatives.
    """
    rouge_pos = compute_rouge(candidate, refs_pos, leave_one_out)

    rouge_neg_max = 0.0
    if refs_neg:
        scores = [compute_rouge(candidate, [neg], leave_one_out=False) for neg in refs_neg]
        rouge_neg_max = max(scores) if scores else 0.0

    return {
        "rouge_pos": rouge_pos,
        "rouge_neg_max": rouge_neg_max,
        "rouge_contrast": rouge_pos - rouge_neg_max
    }


if __name__ == "__main__":
    candidate = "The watermelon seeds pass through your digestive system"
    refs = ["Nothing happens", "The watermelon seeds pass through your digestive system"]
    print(compute_rouge(candidate, refs))
    
    candidate = "The watermelon seeds pass through your digestive system"
    refs = ["The watermelon seeds pass through your digestive system"]
    print(compute_rouge(candidate, refs))  # should be 1.0
