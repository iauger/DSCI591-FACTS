# scripts/features_script.py
import ast
import sys
import os
import json
import argparse
from typing import List, Any, Dict  
import pandas as pd
from tqdm import tqdm

# add project root to sys.path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.nli_scoring import score_nli
from features.aggregate_features import compute_features, aggregate_scores, process_answer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def process_row(qid: int,
    question: str,
    answer: str,
    is_true: bool,
    is_best: bool,
    best_true: str,
    best_false: str
):
    feats = compute_features(str(answer))

    row = {
        "qid": qid,
        "question": question,
        "answer": answer,
        "true_answer": bool(is_true),
        "false_answer": not bool(is_true),
        "best_true_answer": bool(is_true and is_best),
        "best_false_answer": bool((not is_true) and is_best),
    }

    # flatten features with prefix
    for k, v in feats.items():
        row[f"{k}"] = v
    
    # flatten NLI scores with a prefix
    for label, score in score_nli(question, answer).items():
        row[f"nli_{label.lower()}"] = score
    
    # NLI: answer → opposite best anchor
    # TODO: decide on best approach. Everything against Everything or Everything against Best True
    if is_true and best_false:
        for label, score in score_nli(answer, best_false).items():
            row[f"nli_{label.lower()}_vs_best_false"] = score
    elif (not is_true) and best_true:
        for label, score in score_nli(answer, best_true).items():
            row[f"nli_{label.lower()}_vs_best_true"] = score

    return row

def main(in_path, out_path, preview_n=None):
    df = pd.read_csv(in_path)
    if preview_n:
        df = df.head(preview_n)

    out_rows: List[dict] = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Questions"):
        qid = row.get("Question ID", row.get("qid", idx))
        q = row.get("Question", "")
        true_list = list(ast.literal_eval(row.get("Correct Answers", "[]").strip()))
        false_list = list(ast.literal_eval(row.get("Incorrect Answers", "[]").strip()))
        best_true = str(row.get("Best Answer", "")).strip()
        best_false = str(row.get("Best Incorrect Answer", "")).strip()
        
        all_answers = true_list + false_list
        if not all_answers:
            print(f"Skipping empty question {qid}: {q}")
            continue
        
        for ans in tqdm(all_answers, desc=f"Answers for Q{qid}", leave=False):
            is_true = ans in true_list
            is_best = (ans == best_true) if is_true else (ans == best_false)

            out_rows.append(
                process_answer(
                    qid=qid,
                    question=q,
                    answer=ans,
                    all_answers=all_answers,
                    is_true=is_true,
                    is_best=is_best
                )
            )


    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input CSV file")
    parser.add_argument("output", type=str, help="Output JSONL file")
    parser.add_argument(
        "--preview", type=int, default=None, help="Only process the first N rows"
    )
    args = parser.parse_args()

    # call main with args
    main(args.input, args.output, args.preview)

