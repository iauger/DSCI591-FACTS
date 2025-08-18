# lexical.py

from typing import Dict, List
from collections import Counter
from .readability import split_sentences_as_dict

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "on", "in", "to",
    "with", "for", "at", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "that", "this", "these", "those", "he", "she", "it", "they",
    "them", "his", "her", "their", "we", "you", "i", "me", "my", "your", "our"
}

def compute_lexical_features(answer_text: str) -> Dict[str, float]:
    
    text = (answer_text or "").strip().lower()
    sent_dict = split_sentences_as_dict(text)  # {0: {"raw": str, "tokens": [..]}, ...}
    
    tokens = [tok for s in sent_dict.values() for tok in s["tokens"]]
    if not tokens: 
        return {
            "unique_token_count": 0.0,
            "type_token_ratio": 0.0,
            "lexical_density": 0.0,
            "repetition_ratio": 0.0,
            "unique_bigram_ratio": 0.0
        }
        
    unique_tokens = set(tokens)
    unique_count = len(unique_tokens)
    ttr = unique_count / len(tokens)
    
    content_tokens = [t for t in tokens if t not in STOPWORDS]
    lexical_density = len(content_tokens) / len(tokens) if tokens else 0.0
    
    counts = Counter(tokens)
    most_common_count = counts.most_common(1)[0][1] if counts else 0.0
    repetition_ratio = most_common_count / len(tokens) if tokens else 0.0
    
    bigrams = list(zip(tokens, tokens[1:]))
    unique_bigrams = set(bigrams)
    unique_bigram_count = len(unique_bigrams)
    unique_bigram_ratio = unique_bigram_count / (len(tokens) - 1) if len(tokens) > 1 else 0.0

    return {
        "unique_token_count": float(unique_count),
        "type_token_ratio": float(ttr),
        "lexical_density": float(lexical_density),
        "repetition_ratio": float(repetition_ratio),
        "unique_bigram_ratio": float(unique_bigram_ratio)
    }