# -*- coding: utf-8 -*-
from typing import Dict, Any

def _safe_get(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    v = row.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


"""Relative Agreement: relies on NLI vs best_true if available, else QA NLI fallback."""
def compute_ra(row: Dict[str, Any]) -> float:
    ent_best_true = row.get("nli_entailment_vs_best_true", None)
    con_best_true = row.get("nli_contradiction_vs_best_true", None)
    if ent_best_true is not None and con_best_true is not None:
        ra_ent = 1.0 - _safe_get(row, "nli_entailment_vs_best_true", 0.0)
        ra_ent = max(0.0, ra_ent - 0.3)  # subtract threshold
        ra_con = _safe_get(row, "nli_contradiction_vs_best_true", 0.0)
        return max(0.0, min(1.0, 0.6*ra_ent + 0.4*ra_con))

    # fallback to question-answer NLI
    ent_q = _safe_get(row, "nli_q_entailment", 0.0)
    con_q = _safe_get(row, "nli_q_contradiction", 0.0)
    ra_ent = max(0.0, (1.0 - ent_q) - 0.3)
    return max(0.0, min(1.0, 0.6*ra_ent + 0.4*con_q))

"""Contradiction Consistency: max of best_true contradiction and pairwise contradiction."""
def compute_cc(row: Dict[str, Any]) -> float:
    contra_best = _safe_get(row, "nli_contradiction_vs_best_true", 0.0)
    contra_pair_max = _safe_get(row, "nli_pair_contradiction_max", 0.0)
    return max(0.0, min(1.0, max(0.8*contra_best, 0.6*contra_pair_max)))

"""Low-Hallucination Confidence: entity features + readability penalty."""
def compute_lhc(row: Dict[str, Any]) -> float:
    years = _safe_get(row, "entity_year_count", 0.0)
    geos  = _safe_get(row, "entity_geo_count", 0.0)
    nums  = _safe_get(row, "entity_number_count", 0.0)
    curr  = _safe_get(row, "entity_currency_count", 0.0)
    ent_signal = min(1.0, (years + geos + nums + curr) / 5.0)
    reading_ease = row.get("reading_ease", None)
    try:
        re_val = float(reading_ease) if reading_ease is not None else None
    except:
        re_val = None
    if re_val is not None:
        read_signal = max(0.0, min(1.0, (60.0 - re_val) / 60.0))
    else:
        read_signal = 0.0
    return max(0.0, min(1.0, 0.7*ent_signal + 0.3*read_signal))

"""Global Hallucination Index (weighted composite)."""
def compute_ghi(row: Dict[str, Any],
                alpha: float = 0.3,  # RA weight reduced
                gamma: float = 0.5,  # CC weight increased
                delta: float = 0.2   # LHC weight added
               ) -> float:

    #without reference bias

    ra = compute_ra(row)
    cc = compute_cc(row)
    lhc = compute_lhc(row)
    ghi = alpha*ra + gamma*cc + delta*lhc
    return max(0.0, min(1.0, ghi))

"""Tuned Global Hallucination Index (weighted composite)."""
def compute_tuned_ghi(row: Dict[str, Any],
                      w_ra: float = 0.70,  # stronger weight on RA
                      w_cc: float = 0.15,  # smaller weight on CC
                      w_lhc: float = 0.15  # smaller weight on LHC
                     ) -> float:
    """
    Computes an alternative GHI score using tuned weights based on feature importance review.
    """
    ra = compute_ra(row)
    cc = compute_cc(row)
    lhc = compute_lhc(row)

    ghi = w_ra * ra + w_cc * cc + w_lhc * lhc
    return max(0.0, min(1.0, ghi))
