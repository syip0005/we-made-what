import re

import nltk
from wordfreq import zipf_frequency


def ensure_nltk_data() -> None:
    """Download required NLTK data if not present."""
    for corpus in ("words", "wordnet"):
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            nltk.download(corpus, quiet=True)


def _is_noun(word: str) -> bool:
    """Check if a word can be used as a noun via WordNet.

    We want nouns, places, events, and things — not adjectives/verbs/adverbs.
    WordNet synsets with pos='n' indicate the word has a noun sense.
    """
    from nltk.corpus import wordnet

    return len(wordnet.synsets(word, pos=wordnet.NOUN)) > 0


def load_english_words(min_zipf: float = 2.0) -> list[dict]:
    """Load common English nouns from NLTK, filtered by word frequency.

    Returns list of {"word": str, "category": "english"} dicts.
    Filters to nouns only (via WordNet) and removes:
    - Words shorter than 2 characters
    - Words with non-alpha characters
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
        if zipf_frequency(lower, "en") < min_zipf:
            continue
        # Only keep words that have a noun sense
        if not _is_noun(lower):
            continue

        seen.add(lower)
        result.append({"word": lower, "category": "english"})

    result.sort(key=lambda x: x["word"])
    return result
