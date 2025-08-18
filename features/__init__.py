# __init__.py
from .readability import compute_readability
from .lexical import compute_lexical_features
from .style import compute_style_features
from .entities import compute_entities_features

class _Readability:
    name = "readability"
    def compute(self, text): return compute_readability(text)

class _Lexical:
    name = "lexical"
    def compute(self, text): return compute_lexical_features(text)

class _Style:
    name = "style"
    def compute(self, text): 
        # default to lemma for robust matching; override per-call if needed
        return compute_style_features(text, norm="lemma")

class _Entities:
    name = "entities"
    def compute(self, text): return compute_entities_features(text)

ALL_EXTRACTORS = [_Readability(), _Lexical(), _Style(), _Entities()]