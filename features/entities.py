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

us_states = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming"
]

country_names = [country.name.lower() for country in pycountry.countries] # type: ignore
geo_terms = set(regions + country_names + us_states)

def count_geo_terms(tokens: list[str], geo_terms: set[str]) -> int:
    used = set()
    count = 0
    for n in (3, 2, 1):  # check trigrams, bigrams, unigrams in that order
        i = 0
        while i <= len(tokens) - n:
            span_idx = tuple(range(i, i+n))
            if any(j in used for j in span_idx):
                i += 1
                continue
            ngram = " ".join(tokens[i:i+n])
            if ngram in geo_terms:
                count += 1
                used.update(span_idx)
                i += n
            else:
                i += 1
    return count

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
    geo_count = count_geo_terms(tokens, geo_terms)   
    
    # Capitalized words
    # Skips first word in sentence, just trying to ID proper nouns
    cap_count = 0
    for s in split_sentences_as_dict(text, make_lower=False).values():
        raw_tokens = clean_and_tokenize(s["raw"], make_lower=False)
        for i, tok in enumerate(raw_tokens):
            if i == 0:
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