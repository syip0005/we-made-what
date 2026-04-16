#!/usr/bin/env python3
"""Fetch pop culture + slang vocabulary and cache as JSON.

Sources:
- Wikidata SPARQL: films, TV, actors, musicians, games, anime, manga, characters
- HuggingFace MLBtrio/genz-slang-dataset: Gen-Z slang terms

Run this before build-vocab.py to include pop culture in the vocabulary.
This is a separate step because fetching is slow and rate-limited,
while vocab building (embedding) can be re-run quickly.

Usage:
    uv run python scripts/fetch-popculture.py
"""

import json
import logging
import sys
from pathlib import Path

# Add backend src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from wmw.vocab.popculture import fetch_all_popculture
from wmw.vocab.slang import fetch_slang

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    output_dir = Path(__file__).parent.parent / "backend" / "src" / "wmw" / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch pop culture from Wikidata
    logger.info("=== Fetching pop culture entities from Wikidata ===")
    logger.info("This may take several minutes due to rate limiting.\n")
    popculture_entries = fetch_all_popculture()

    popculture_path = output_dir / "popculture.json"
    popculture_path.write_text(json.dumps(popculture_entries, ensure_ascii=False, indent=2))
    logger.info(f"\nSaved {len(popculture_entries)} pop culture entries to {popculture_path}")

    # Fetch Gen-Z slang from HuggingFace
    logger.info("\n=== Fetching Gen-Z slang from HuggingFace ===\n")
    slang_entries = fetch_slang()

    slang_path = output_dir / "slang.json"
    slang_path.write_text(json.dumps(slang_entries, ensure_ascii=False, indent=2))
    logger.info(f"Saved {len(slang_entries)} slang entries to {slang_path}")

    # Summary
    all_entries = popculture_entries + slang_entries
    logger.info(f"\n=== Total: {len(all_entries)} entries ===")
    categories: dict[str, int] = {}
    for entry in all_entries:
        cat = entry["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
