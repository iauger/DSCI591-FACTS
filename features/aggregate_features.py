from typing import Dict, Iterable, Optional
from . import ALL_EXTRACTORS

def compute_features(
    text: str,
    use: Optional[Iterable[str]] = None 
) -> Dict[str, float]:
    """
    Compute all features for the given text.
    
    :param text: The input text to analyze.
    :param use: Optional list of feature extractor names to use. If None, all extractors are used.
    :return: A dictionary of computed features.
    """
    selected = ALL_EXTRACTORS if not use else [
        ext for ext in ALL_EXTRACTORS if ext.name in use
    ]
    
    out: Dict[str, float] = {}
    for ext in selected:
        features = ext.compute(text)
        out.update(features)
    return out