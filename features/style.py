#style.py

import math
from typing import Dict
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')   # optional but helps with coverage

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from .readability import split_sentences_as_dict

NEGATORS = {
    "no", "not", "never", "none", "nothing", "nowhere", "neither",
    "cannot", "can't", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
    "without", "nor"
}

HEDGES = {
    "may", "might", "could", "possibly", "perhaps", "seems",
    "suggests", "likely", "unlikely", "appears", "approximate",
    "generally", "apparently", "probably", "often"
}

BOOSTERS = {
    "always", "definitely", "certainly", "absolutely",
    "undoubtedly", "must", "guaranteed",
    "completely", "entirely", "totally", "forever", "perfectly"
}

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
    
def normalize_token(tok: str, method="lemma") -> str:
    tok = tok.lower()
    
    
    if method == "stem":
        return stemmer.stem(tok)
    elif method == "lemma":
        return lemmatizer.lemmatize(tok)
    return tok

def normalize_lexicon(words, method="exact"):
    return {normalize_token(w, method) for w in words}

def compute_style_features(answer_text: str, norm: str = "exact") -> Dict[str, float]:
    
    text = (answer_text or "").strip()
    sent_dict = split_sentences_as_dict(text)  # {0: {"raw": str, "tokens": [..]}, ...}

    tokens = [tok for s in sent_dict.values() for tok in s["tokens"]]
    token_count = len(tokens)
    
    if not tokens:
        return {
            "negation_count": 0.0,
            "negation_ratio": 0.0,
            "hedge_ratio": 0.0,
            "booster_ratio": 0.0,
            "modality_balance": 0.0,
        }
    
    # Normalize tokens and lexicons according to chosen method
    tokens_lower = [t.lower() for t in tokens]
    tokens_norm = [normalize_token(t, method=norm) for t in tokens_lower]
    negators_norm = normalize_lexicon(NEGATORS, method=norm)
    hedges_norm = normalize_lexicon(HEDGES, method=norm)
    boosters_norm = normalize_lexicon(BOOSTERS, method=norm)

    neg_count = sum(t in negators_norm for t in tokens_norm)
    hedge_count = sum(t in hedges_norm for t in tokens_norm)
    booster_count = sum(t in boosters_norm for t in tokens_norm)

    neg_ratio = float(neg_count) / token_count if token_count > 0 else 0.0
    hedge_ratio = float(hedge_count) / token_count if token_count > 0 else 0.0
    booster_ratio = float(booster_count) / token_count if token_count > 0 else 0.0
    # Ensure no zero division error
    modality_balance_simple = float(booster_count) / (hedge_count + 1) 
    # Apply log transformation for relative scaling
    eps = 1e-6
    modality_balance_log = math.log1p(booster_count) - math.log1p(hedge_count + eps)

    return {
        "negation_count": float(neg_count),
        "negation_ratio": float(neg_ratio),
        "hedge_ratio": float(hedge_ratio),
        "booster_ratio": float(booster_ratio),
        "modality_balance_simple": float(modality_balance_simple),
        "modality_balance_log": float(modality_balance_log),
    }
