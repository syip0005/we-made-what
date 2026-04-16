"""Fetch Gen-Z slang terms from HuggingFace dataset.

Source: https://huggingface.co/datasets/MLBtrio/genz-slang-dataset
~1,779 slang terms with descriptions.
"""

import io
import logging

import httpx

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://huggingface.co/datasets/MLBtrio/genz-slang-dataset"
    "/resolve/main/data/train-00000-of-00001.parquet"
)


def _normalize_slang(text: str) -> str | None:
    """Normalize a slang entry. Returns None if it should be skipped."""
    text = text.strip().lower()
    if not text:
        return None
    # Skip very short entries (single letters)
    if len(text) < 2:
        return None
    # Skip entries that are too long
    if len(text) > 60:
        return None
    return text


def fetch_slang() -> list[dict]:
    """Download Gen-Z slang dataset from HuggingFace and return as word entries."""
    import pyarrow.parquet as pq

    logger.info("Fetching Gen-Z slang dataset from HuggingFace...")

    with httpx.Client() as client:
        resp = client.get(DATASET_URL, timeout=60, follow_redirects=True)
        resp.raise_for_status()

    table = pq.read_table(io.BytesIO(resp.content))
    df_slang = table.column("Slang")

    results = []
    seen: set[str] = set()

    for value in df_slang:
        text = value.as_py()
        if text is None:
            continue
        normalized = _normalize_slang(text)
        if normalized and normalized not in seen:
            seen.add(normalized)
            results.append({"word": normalized, "category": "slang"})

    logger.info(f"  Got {len(results)} slang entries")
    return results
