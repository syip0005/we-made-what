"""Fetch pop culture entities from Wikidata SPARQL.

All categories (films, TV, actors, musicians, video games, fictional characters,
internet memes, subcultures) are sourced from Wikidata — single source, no API keys.

Results are filtered to English-only by requiring an English Wikipedia article.

Each entry is {"word": str, "category": str}.
"""

import logging
import re
import time

import httpx

logger = logging.getLogger(__name__)

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

WIKIDATA_USER_AGENT = "WeMadeWhat/0.1 (https://github.com/syip0005/we-made-what) httpx"

# All queries require an English Wikipedia article to ensure English labels.
# This filters out non-English entries that would be noise in the game.
WIKIDATA_QUERIES: dict[str, tuple[str, str]] = {
    "film_recent": (
        "film",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q11424.
          ?item wdt:P577 ?date.
          FILTER(YEAR(?date) >= 2000)
          ?article schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 10000
        """,
    ),
    "film_classic": (
        "film",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q11424.
          ?item wdt:P577 ?date.
          FILTER(YEAR(?date) >= 1970 && YEAR(?date) < 2000)
          ?article schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 10000
        """,
    ),
    "tv_recent": (
        "tv",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q5398426.
          ?item wdt:P580|wdt:P577 ?date.
          FILTER(YEAR(?date) >= 2000)
          ?article schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 10000
        """,
    ),
    "tv_classic": (
        "tv",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q5398426.
          ?item wdt:P580|wdt:P577 ?date.
          FILTER(YEAR(?date) < 2000)
          ?article schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 10000
        """,
    ),
    "actor": (
        "actor",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P106 wd:Q33999.
          ?article schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 10000
        """,
    ),
    "musician": (
        "music",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          {?item wdt:P106 wd:Q177220.} UNION {?item wdt:P106 wd:Q639669.}
          ?article schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 10000
        """,
    ),
    "video_game": (
        "game",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q7889.
          ?item wdt:P577 ?date.
          FILTER(YEAR(?date) >= 1985)
          ?article schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 10000
        """,
    ),
    "fictional_character": (
        "character",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q95074.
          ?article schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 15000
        """,
    ),
    "internet_meme": (
        "meme",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q2927074.
          ?article schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 5000
        """,
    ),
    "subculture": (
        "culture",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          {?item wdt:P31 wd:Q1752346.}
          UNION {?item wdt:P31 wd:Q264965.}
          ?article schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 5000
        """,
    ),
}

# Common Unicode → ASCII replacements
_UNICODE_REPLACEMENTS = {
    "\u2013": "-",  # en-dash → hyphen
    "\u2014": "-",  # em-dash → hyphen
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",  # non-breaking space
}

# Regex to detect non-English entries after normalization
_NON_ENGLISH_RE = re.compile(r"[^\x00-\x7F]")


def _normalize_entry(text: str) -> str | None:
    """Normalize a pop culture entry. Returns None if it should be skipped."""
    text = text.strip()
    if not text:
        return None
    # Skip Wikidata Q-IDs that weren't resolved to labels
    if text.startswith("Q") and text[1:].isdigit():
        return None
    # Skip entries that are too long (likely descriptions, not names)
    if len(text) > 60:
        return None
    # Normalize common Unicode punctuation to ASCII
    for unicode_char, ascii_char in _UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, ascii_char)
    # Skip entries with remaining non-ASCII characters (non-English)
    if _NON_ENGLISH_RE.search(text):
        return None
    # Skip single-character entries
    if len(text) < 2:
        return None
    # Lowercase for consistency
    return text.lower()


def fetch_all_popculture() -> list[dict]:
    """Fetch all pop culture entities from Wikidata SPARQL."""
    results = []
    seen: set[str] = set()

    with httpx.Client() as client:
        for query_name, (category, sparql) in WIKIDATA_QUERIES.items():
            logger.info(f"Fetching {query_name} from Wikidata...")
            try:
                resp = client.get(
                    WIKIDATA_ENDPOINT,
                    params={"query": sparql, "format": "json"},
                    headers={
                        "User-Agent": WIKIDATA_USER_AGENT,
                        "Accept": "application/sparql-results+json",
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                # Wikidata sometimes returns malformed JSON (control chars,
                # unescaped quotes in labels). Clean aggressively.
                import json

                cleaned = re.sub(r"[\x00-\x1f\x7f]", "", resp.text)
                try:
                    data = json.loads(cleaned)
                except json.JSONDecodeError:
                    # Last resort: try parsing with strict=False
                    data = json.loads(cleaned, strict=False)  # type: ignore[call-overload]

                count = 0
                for binding in data.get("results", {}).get("bindings", []):
                    label = binding.get("itemLabel", {}).get("value", "")
                    normalized = _normalize_entry(label)
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        results.append({"word": normalized, "category": category})
                        count += 1

                logger.info(f"  Got {count} {query_name} entries")
            except Exception as e:
                logger.warning(f"  Failed to fetch {query_name}: {e}")

            # Be polite to Wikidata
            time.sleep(2)

    return results
