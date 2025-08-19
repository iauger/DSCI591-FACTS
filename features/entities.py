import re
import pycountry
from typing import Dict
from .readability import split_sentences_as_dict, clean_and_tokenize


# --- Regex patterns ---
NUMBER_PATTERN = re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b')
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
CURRENCY_PATTERN = re.compile(r'[$€£¥]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?')

# --- Geographic terms ---
regions = [
    "europe", "asia", "africa", "north america", "south america",
    "central america", "latin america", "middle east", "eastern europe",
    "western europe", "southeast asia", "east asia", "west africa",
    "central asia", "scandinavia", "balkans", "caucasus",
    "british isles", "gulf region"
]

country_names = [country.name.lower() for country in pycountry.countries] # type: ignore
geo_terms = set(regions + country_names)

def compute_entities_features(answer_text: str) -> Dict[str, float]:
    text = (answer_text or "").strip()
    sent_dict = split_sentences_as_dict(text)  # {0: {"raw": str, "tokens": [..]}, ...}
    tokens = [tok.lower() for s in sent_dict.values() for tok in s["tokens"]]
    token_count = len(tokens)
    
    if not tokens:
        return {
            "entity_number_count": 0.0,
            "entity_year_count": 0.0,
            "entity_currency_count": 0.0,
            "entity_geo_count": 0.0,
            "entity_capitalized_count": 0.0,
            "entity_ratio": 0.0,
        }
    
    # Regex matches
    num_count = len(NUMBER_PATTERN.findall(text))
    year_count = len(YEAR_PATTERN.findall(text))
    currency_count = len(CURRENCY_PATTERN.findall(text))    
    
    # Geographic matches
    geo_count = sum(1 for tok in tokens if tok in geo_terms)    
    
    # Capitalized words
    # Skips first word in sentence, just trying to ID proper nouns
    cap_count = 0
    for s in sent_dict.values():
        raw_tokens = clean_and_tokenize(s["raw"])
        for i, tok in enumerate(raw_tokens):
            if i == 0:  # skip first token in sentence
                continue
            if tok[:1].isupper():
                cap_count += 1

    entity_total = num_count + year_count + currency_count + geo_count + cap_count
    entity_ratio = entity_total / token_count if token_count > 0 else 0.0

    return {
        "entity_number_count": float(num_count),
        "entity_year_count": float(year_count),
        "entity_currency_count": float(currency_count),
        "entity_geo_count": float(geo_count),
        "entity_capitalized_count": float(cap_count),
        "entity_ratio": float(entity_ratio),
    }