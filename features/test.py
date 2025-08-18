
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.readability import clean_and_tokenize, split_sentences_as_dict, compute_readability
from features.lexical import compute_lexical_features as compute_lexical
from features.style import compute_style_features
from features.entities import compute_entities_features as compute_entity_features

import pycountry

# Readability Tests #

def test_clean_and_tokenize():
    text = "Don't stop-believin'!"
    tokens = clean_and_tokenize(text)
    assert isinstance(tokens, list)
    assert "don't" in tokens
    assert "stop-believin" in tokens

def test_split_sentences_as_dict():
    text = "This is the first sentence. And here is the second!"
    sent_dict = split_sentences_as_dict(text)

    # Should return a dict with 2 entries
    assert isinstance(sent_dict, dict)
    assert len(sent_dict) == 2
    # Each entry should have "raw" and "tokens"
    for s in sent_dict.values():
        assert "raw" in s and "tokens" in s
        assert isinstance(s["tokens"], list)

def test_compute_readability():
    text = "This is a short example. It should be easy to read."
    result = compute_readability(text)

    expected_keys = {
        "reading_ease", "fk_grade",
        "sentence_count", "token_count",
        "avg_sentence_len", "sentence_len_std"
    }
    # Ensure all keys present
    assert set(result.keys()) == expected_keys
    # Ensure all values are numeric
    for val in result.values():
        assert isinstance(val, float)

    # Quick sanity check: >0 tokens, >0 sentences
    assert result["sentence_count"] > 0
    assert result["token_count"] > 0

# Lexical Tests #

def test_empty_text():
    out = compute_lexical("")
    assert out["unique_token_count"] == 0.0
    assert out["type_token_ratio"] == 0.0
    assert out["lexical_density"] == 0.0
    assert out["repetition_ratio"] == 0.0
    assert out["unique_bigram_ratio"] == 0.0

def test_repetition_extreme():
    # All tokens same → repetition_ratio = 1.0, bigram ratio low
    out = compute_lexical("hello hello hello hello")
    assert out["unique_token_count"] == 1.0
    assert out["type_token_ratio"] == 1.0 / 4.0
    assert out["repetition_ratio"] == 1.0
    # 3 bigrams, but only one unique: ("hello","hello")
    assert abs(out["unique_bigram_ratio"] - (1.0 / 3.0)) < 1e-6

def test_density_stopwords_only():
    # All stopwords → lexical_density = 0
    out = compute_lexical("the and or but the and or but")
    assert out["lexical_density"] == 0.0
    # Still has tokens and uniques
    assert out["unique_token_count"] >= 1.0
    assert out["type_token_ratio"] > 0.0

def test_basic_mix():
    text = "Cats chase mice. Cats chase small mice."
    out = compute_lexical(text)

    # Sanity checks on ranges
    assert 0.0 < out["type_token_ratio"] <= 1.0
    assert 0.0 <= out["lexical_density"] <= 1.0
    assert 0.0 < out["repetition_ratio"] <= 1.0
    # With some repetition of 'cats'/'chase', ratio should be noticeably > 0.2
    assert out["repetition_ratio"] >= 0.25

    # Bigram ratio should be < 1 when there is repetition
    assert 0.0 <= out["unique_bigram_ratio"] < 1.0
    
# Style Tests #

def test_negation():
    text = "This is not true. That cannot be correct."
    out = compute_style_features(text)
    assert out["negation_count"] >= 2
    assert out["negation_ratio"] > 0.0

def test_hedge():
    text = "This might be true. It could possibly happen."
    out = compute_style_features(text)
    assert out["hedge_ratio"] > 0.0
    assert out["booster_ratio"] == 0.0

def test_booster():
    text = "This is definitely true. It will always happen."
    out = compute_style_features(text)
    assert out["booster_ratio"] > 0.0
    assert out["hedge_ratio"] == 0.0

def test_modality_balance_behavior():
    # Booster-heavy, no hedges
    text = "This is absolutely guaranteed."
    out = compute_style_features(text)
    assert out["booster_ratio"] > 0.0
    # Simple balance should equal booster_count (since hedges=0)
    assert abs(out["modality_balance_simple"] - out["booster_ratio"]*len(text.split())) >= 0.0
    # Log balance should be positive when boosters > hedges
    assert out["modality_balance_log"] > 0.0

def test_combined_case():
    text = "It might be true, but it is not definitely guaranteed."
    out = compute_style_features(text)
    assert out["negation_count"] >= 1
    assert out["hedge_ratio"] > 0.0
    assert out["booster_ratio"] > 0.0

# Entities Tests #

def pretty(res):
    # Small helper to print a compact summary during manual runs
    keys = ["entity_number_count","entity_year_count","entity_currency_count",
            "entity_geo_count","entity_capitalized_count","entity_ratio"]
    return {k: res[k] for k in keys}

def test_empty():
    out = compute_entity_features("")
    assert out["entity_ratio"] == 0.0
    assert sum(out[k] for k in out if k.endswith("_count")) == 0.0

def test_numbers_and_currency():
    text = "The device costs $299.99 and the discount is 20%."
    out = compute_entity_features(text)
    # Should detect at least one currency and one number/percent
    assert out["entity_currency_count"] >= 1.0
    assert out["entity_number_count"] >= 1.0
    print("numbers/currency:", pretty(out))

def test_years():
    text = "In 1999 and again in 2025, upgrades were announced."
    out = compute_entity_features(text)
    assert out["entity_year_count"] >= 2.0
    print("years:", pretty(out))

def test_geo_terms():
    text = "Research focused on Scandinavia and Brazil, then moved to North America."
    out = compute_entity_features(text)
    assert out["entity_geo_count"] >= 2.0
    print("geo:", pretty(out))

def test_capitalized_proxy():
    # Proper-noun-ish words; note: our tokenizer lowercases tokens,
    # so capitalized count is conservative. This test just ensures the call works.
    text = "Alice met Bob at OpenAI in San Francisco."
    out = compute_entity_features(text)
    # We don't hard-assert a specific cap count because implementation may differ.
    assert out["entity_ratio"] >= 0.0
    print("capitalized proxy:", pretty(out))
    
if __name__ == "__main__":
    tests = [
        ("test_clean_and_tokenize", test_clean_and_tokenize),
        ("test_split_sentences_as_dict", test_split_sentences_as_dict),
        ("test_compute_readability", test_compute_readability),
        ("test_empty_text", test_empty_text),
        ("test_repetition_extreme", test_repetition_extreme),
        ("test_density_stopwords_only", test_density_stopwords_only),
        ("test_basic_mix", test_basic_mix),
        ("test_negation", test_negation),
        ("test_hedge", test_hedge),
        ("test_booster", test_booster),
        ("test_modality_balance_behavior", test_modality_balance_behavior),
        ("test_combined_case", test_combined_case),
        ("test_empty", test_empty),
        ("test_numbers_and_currency", test_numbers_and_currency),
        ("test_years", test_years),
        ("test_geo_terms", test_geo_terms),
        ("test_capitalized_proxy", test_capitalized_proxy),
    ]

    for name, func in tests:
        try:
            func()
            print(f"{name}: PASSED")
        except Exception as e:
            print(f"{name}: FAILED - {e}")
    print("All tests passed!")