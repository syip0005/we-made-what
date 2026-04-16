"""Fetch Gen-Z slang terms from HuggingFace dataset.

Source: https://huggingface.co/datasets/MLBtrio/genz-slang-dataset
~1,779 slang terms with descriptions, filtered to noun-like concepts.
"""

import csv
import io
import logging

import httpx

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://huggingface.co/datasets/MLBtrio/genz-slang-dataset/resolve/main/all_slangs.csv"
)


def _is_useful_slang(slang: str, description: str) -> bool:
    """Filter slang entries to noun-like concepts useful for word arithmetic.

    Rejects:
    - Verb-action entries (description starts with "To ")
    - Short all-caps acronyms (<=4 chars, e.g. "TBH", "NGL", "IMO")
    - Single character entries
    """
    if len(slang) < 2:
        return False
    # Skip verb-action slang
    if description.lstrip().startswith(("To ", "to ")):
        return False
    # Skip short all-uppercase acronyms — they're abbreviations, not concepts
    return not (slang.isupper() and len(slang) <= 4)


def _normalize_slang(text: str) -> str | None:
    """Normalize a slang entry. Returns None if it should be skipped."""
    text = text.strip().lower()
    if not text:
        return None
    if len(text) > 60:
        return None
    return text


def fetch_slang() -> list[dict]:
    """Download Gen-Z slang dataset from HuggingFace and return filtered entries."""
    logger.info("Fetching Gen-Z slang dataset from HuggingFace...")

    with httpx.Client() as client:
        resp = client.get(DATASET_URL, timeout=60, follow_redirects=True)
        resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    results = []
    seen: set[str] = set()
    skipped = 0

    for row in reader:
        slang = row.get("Slang", "")
        description = row.get("Description", "")

        if not _is_useful_slang(slang, description):
            skipped += 1
            continue

        normalized = _normalize_slang(slang)
        if normalized and normalized not in seen:
            seen.add(normalized)
            results.append({"word": normalized, "category": "slang"})

    logger.info(f"  Got {len(results)} slang entries (skipped {skipped})")
    return results
