# scripts/metrics_script.py
import os, sys, json, argparse
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metric_calculation import compute_ra, compute_cc, compute_lhc, compute_ghi, compute_tuned_ghi
from evaluation.bleu import compute_bleu_contrastive, compute_bleu_anchor
from evaluation.rouge import compute_rouge_contrastive, compute_rouge_anchor

def _load_rows(in_path: str, preview_n: Optional[int] = None) -> List[Dict]:
    rows: List[Dict] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if preview_n is not None and i >= preview_n:
                break
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows

def _build_refs_by_qid(rows: List[Dict]) -> Dict[str, Dict[str, List[str]]]:
    refs: Dict[str, Dict[str, List[str]]] = {}
    for r in rows:
        qid = str(r.get("qid"))
        ans = str(r.get("answer", "")).strip()
        if not ans:
            continue
        is_true = bool(r.get("true_answer", False))
        bucket = "pos" if is_true else "neg"
        if qid not in refs:
            refs[qid] = {"pos": [], "neg": []}
        if ans not in refs[qid][bucket]:
            refs[qid][bucket].append(ans)
    return refs

# Best anchors per qid (from row flags)
def _build_best_anchors(rows: List[Dict]) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Returns anchors_by_qid[qid] = {"bta": <best true>, "bfa": <best false>} if present.
    """
    anchors: Dict[str, Dict[str, Optional[str]]] = {}
    for r in rows:
        qid = str(r.get("qid"))
        ans = str(r.get("answer", "")).strip()
        if not ans:
            continue
        if qid not in anchors:
            anchors[qid] = {"bta": None, "bfa": None}
        if r.get("best_true_answer", False):
            anchors[qid]["bta"] = ans
        if r.get("best_false_answer", False):
            anchors[qid]["bfa"] = ans
    return anchors

def main(in_path: str, out_path: str, preview_n: Optional[int] = None, bleu_mode: str = "anchor"):
    rows = _load_rows(in_path, preview_n=preview_n)

    refs_by_qid = _build_refs_by_qid(rows)
    anchors_by_qid = _build_best_anchors(rows)  # for pairwise BLEU

    out_rows = []
    for r in tqdm(rows, desc="Scoring rows"):
        qid = str(r.get("qid"))
        candidate = str(r.get("answer", "")).strip()

        # Core metrics (per-row)
        r["RA"]  = compute_ra(r)
        r["CC"]  = compute_cc(r)
        r["LHC"] = compute_lhc(r)
        r["GHI"] = compute_ghi(r)
        r["alt_GHI"] = compute_tuned_ghi(r)

        # BLEU: compute both corpus-level and anchor-level
        pos_refs = refs_by_qid.get(qid, {}).get("pos", [])
        neg_refs = refs_by_qid.get(qid, {}).get("neg", [])

        # Corpus BLEU (contrastive)
        bleu_corpus = compute_bleu_contrastive(
            candidate=candidate,
            refs_pos=pos_refs,
            refs_neg=neg_refs,
            leave_one_out=True
        )

        # Anchor BLEU (local BTA/BFA comparison)
        bta = anchors_by_qid.get(qid, {}).get("bta")  # may be None
        bfa = anchors_by_qid.get(qid, {}).get("bfa")  # may be None
        bleu_anchor = compute_bleu_anchor(
            candidate=candidate,
            bta_text=bta,
            bfa_text=bfa
        )

        # Add all 6 metrics to the row
        r.update({
            # anchors
            "bta": bta if bta else "",
            "bfa": bfa if bfa else "",
            # corpus-level
            "bleu_pos": bleu_corpus["bleu_pos"],
            "bleu_neg_max": bleu_corpus["bleu_neg_max"],
            "bleu_contrast": bleu_corpus["bleu_contrast"],
            # anchor-level
            "bleu_bta": bleu_anchor["bleu_bta"],
            "bleu_bfa": bleu_anchor["bleu_bfa"],
            "bleu_bta_minus_bfa": bleu_anchor["bleu_bta_minus_bfa"],
        })
        
        # ROUGE: corpus-level (contrastive)
        rouge_corpus = compute_rouge_contrastive(
            candidate=candidate,
            refs_pos=pos_refs,
            refs_neg=neg_refs,
            leave_one_out=True
        )

        # ROUGE: anchor-level
        rouge_anchor = compute_rouge_anchor(
            candidate=candidate,
            bta_text=bta,
            bfa_text=bfa
        )

        r.update({
            # corpus-level
            "rouge_pos": rouge_corpus["rouge_pos"],
            "rouge_neg_max": rouge_corpus["rouge_neg_max"],
            "rouge_contrast": rouge_corpus["rouge_contrast"],
            # anchor-level
            "rouge_bta": rouge_anchor["rouge_bta"],
            "rouge_bfa": rouge_anchor["rouge_bfa"],
            "rouge_bta_minus_bfa": rouge_anchor["rouge_bta_minus_bfa"],
        })

        out_rows.append(r)

    df = pd.DataFrame(out_rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows with metrics (+BLEU:{bleu_mode}) to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="Input JSONL features file")
    p.add_argument("output", type=str, help="Output CSV file with metrics")
    p.add_argument("--preview", type=int, default=None, help="Only process first N rows")
    p.add_argument("--bleu-mode", choices=["anchor","corpus"], default="anchor",
                   help="anchor = candidate vs. BTA/BFA only; corpus = vs. all refs for the QID")
    args = p.parse_args()
    main(args.input, args.output, args.preview, args.bleu_mode)
