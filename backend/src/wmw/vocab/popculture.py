"""Fetch pop culture entities from Wikidata SPARQL.

All categories (films, TV, actors, musicians, video games, anime, manga,
fictional characters) are sourced from Wikidata — single source, no API keys.

Each entry is {"word": str, "category": str}.
"""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# Wikidata SPARQL queries for different pop culture categories.
# Q-codes: Q11424=film, Q5398426=TV series, Q33999=actor, Q177220=singer,
# Q639669=musician, Q7889=video game, Q95074=fictional character,
# Q63952888=anime series, Q21198342=manga series, Q1569167=anime character,
# Q2927074=internet meme, Q1068038=internet phenomenon, Q184130=neologism,
# Q1580752=catchphrase, Q1752346=youth subculture
WIKIDATA_QUERIES: dict[str, tuple[str, str]] = {
    "film": (
        "film",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q11424.
          ?item wdt:P577 ?date.
          FILTER(YEAR(?date) >= 1970)
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 15000
        """,
    ),
    "tv": (
        "tv",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q5398426.
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
          ?item wdt:P27 ?country.
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
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 10000
        """,
    ),
    "anime": (
        "anime",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          {?item wdt:P31 wd:Q63952888.}
          UNION {?item wdt:P31 wd:Q1107.}
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 10000
        """,
    ),
    "manga": (
        "manga",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q21198342.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 5000
        """,
    ),
    "fictional_character": (
        "character",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q95074.
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
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 5000
        """,
    ),
    "internet_phenomenon": (
        "meme",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q1068038.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 5000
        """,
    ),
    "neologism": (
        "slang",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q184130.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 5000
        """,
    ),
    "catchphrase": (
        "meme",
        """
        SELECT DISTINCT ?itemLabel WHERE {
          ?item wdt:P31 wd:Q1580752.
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
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 5000
        """,
    ),
}


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
                    headers={"User-Agent": "WeMadeWhat/0.1 (word-mixing-game)"},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

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
