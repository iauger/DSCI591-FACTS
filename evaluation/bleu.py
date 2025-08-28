# evaluation/bleu.py
from typing import List, Optional, Dict
import pandas as pd
import sacrebleu

def compute_bleu(candidate: str, positive_refs: List[str], leave_one_out: bool = True) -> float:
    if not positive_refs:
        return 0.0
    candidate = str(candidate).strip()
    refs = [str(r).strip() for r in positive_refs if pd.notna(r) and str(r).strip()]
    if leave_one_out and candidate in refs:
        refs = [r for r in refs if r != candidate]
        if not refs:
            return 0.0
    references = [[r] for r in refs]
    bleu = sacrebleu.corpus_bleu([candidate], references)
    return float(bleu.score)

def compute_bleu_contrastive(candidate: str,
                             refs_pos: List[str],
                             refs_neg: Optional[List[str]] = None,
                             leave_one_out: bool = True) -> Dict[str, float]:
    bleu_pos = compute_bleu(candidate, refs_pos, leave_one_out=leave_one_out)
    bleu_neg_max = 0.0
    if refs_neg:
        scores = [compute_bleu(candidate, [neg], leave_one_out=False) for neg in refs_neg if str(neg).strip()]
        bleu_neg_max = max(scores) if scores else 0.0
    return {"bleu_pos": bleu_pos, "bleu_neg_max": bleu_neg_max, "bleu_contrast": bleu_pos - bleu_neg_max}

# NEW: strictly pairwise “anchor” BLEU, congruent with per-row features
def compute_bleu_anchor(candidate: str,
                        bta_text: Optional[str] = None,
                        bfa_text: Optional[str] = None) -> Dict[str, float]:
    """
    BLEU vs. the best true anchor (BTA), and optionally vs. best false anchor (BFA).
    Returns 0.0 if an anchor is missing.
    """
    cand = str(candidate).strip()
    bleu_bta = compute_bleu(cand, [bta_text], leave_one_out=False) if bta_text and str(bta_text).strip() else 0.0
    bleu_bfa = compute_bleu(cand, [bfa_text], leave_one_out=False) if bfa_text and str(bfa_text).strip() else 0.0
    return {
        "bleu_bta": bleu_bta,
        "bleu_bfa": bleu_bfa,
        "bleu_bta_minus_bfa": bleu_bta - bleu_bfa
    }

# Example Usage
if __name__ == "__main__":
    candidate = "The cat sits on the mat."
    bta_text = "The cat is on the mat."
    bfa_text = "A dog is on the mat."
    bleu_scores = compute_bleu_anchor(candidate, bta_text, bfa_text)
    print(bleu_scores)