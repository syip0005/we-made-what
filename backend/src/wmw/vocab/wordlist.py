import re

import nltk
from wordfreq import zipf_frequency


def ensure_nltk_data() -> None:
    """Download required NLTK data if not present."""
    for corpus in ("words",):
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            nltk.download(corpus, quiet=True)


def load_english_words(min_zipf: float = 2.0) -> list[dict]:
    """Load common English words from NLTK, filtered by word frequency.

    Returns list of {"word": str, "category": "english"} dicts.
    Filters out:
    - Words shorter than 2 characters
    - Words with non-alpha characters
    - Proper nouns (capitalized in the corpus)
    - Words below the zipf frequency threshold
    """
    ensure_nltk_data()

    from nltk.corpus import words as nltk_words

    raw_words = set(nltk_words.words())
    result = []
    seen = set()

    for word in raw_words:
        lower = word.lower()

        if lower in seen:
            continue
        if len(lower) < 2:
            continue
        if not re.match(r"^[a-z]+$", lower):
            continue
        # Filter by frequency — removes obscure words
        if zipf_frequency(lower, "en") < min_zipf:
            continue

        seen.add(lower)
        result.append({"word": lower, "category": "english"})

    result.sort(key=lambda x: x["word"])
    return result
