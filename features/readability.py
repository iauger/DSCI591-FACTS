# readability.py

from __future__ import annotations
from typing import Any, Dict
import math
import re
import textstat

# Keep contractions/hyphenated words
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")

TERMINATORS = [".", "!", "?"]
LINE_BREAKS = ["\n", "\r"]

def clean_and_tokenize(sentence: str, make_lower: bool = True) -> list[str]:
    WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")

    # Remove special characters and tokenize
    sentence = sentence.strip()
    if make_lower:
        sentence = sentence.lower()
    words = WORD_RE.findall(sentence)
    return words

def split_sentences_as_dict(text: str, make_lower: bool = True) -> Dict[int, Dict[str, Any]]:
    """
    Split text into sentences and return a dict with raw text and cleaned tokens.
    Example:
    {
      0: {"raw": "This is the first.", "tokens": ["this", "is", "the", "first"]},
      1: {"raw": "Second one!", "tokens": ["second", "one"]}
    }
    """
    for br in LINE_BREAKS:
        text = text.replace(br, " ")
    text = text.strip()
    if not text:
        return {}

    if not any(p in text for p in TERMINATORS):
        return {0: {"raw": text, "tokens": clean_and_tokenize(text, make_lower=make_lower)}}

    pattern = "[" + re.escape("".join(TERMINATORS)) + "]"
    parts = re.split(pattern, text)
    parts = [p.strip() for p in parts if p.strip()]

    return {
        i: {"raw": s, "tokens": clean_and_tokenize(s, make_lower=make_lower)}
        for i, s in enumerate(parts)
    }

def compute_readability(answer_text: str) -> Dict[str, float]:
    """
    Compute readability features for a single ANSWER string,
    using textstat for FE/FK and our richer sentence dict for length stats.
    """
    text = (answer_text or "").strip()

    fe = float(textstat.flesch_reading_ease(text)) if text else 0.0 # type: ignore
    fk = float(textstat.flesch_kincaid_grade(text)) if text else 0.0 # type: ignore

    sent_dict = split_sentences_as_dict(text) 

    if not sent_dict:
        return {
            "reading_ease": fe,
            "fk_grade": fk,
            "sentence_count": 0.0,
            "token_count": 0.0,
            "avg_sentence_len": 0.0,
            "sentence_len_std": 0.0,
        }

    lengths = [len(s["tokens"]) for s in sent_dict.values()]
    sentence_count = len(lengths)
    token_count = sum(lengths)

    mean_len = sum(lengths) / sentence_count
    var = sum((l - mean_len) ** 2 for l in lengths) / sentence_count
    std = math.sqrt(var)

    return {
        "reading_ease": fe,
        "fk_grade": fk,
        "sentence_count": float(sentence_count),
        "token_count": float(token_count),
        "avg_sentence_len": float(mean_len),
        "sentence_len_std": float(std),
    }
