#!/usr/bin/env python3
"""Fetch pop culture entities from Wikidata SPARQL and cache as JSON.

Run this before build-vocab.py to include pop culture in the vocabulary.
This is a separate step because API fetching is slow and rate-limited,
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    output_dir = Path(__file__).parent.parent / "backend" / "src" / "wmw" / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "popculture.json"

    logger.info("Fetching pop culture entities from Wikidata...")
    logger.info("This may take several minutes due to rate limiting.\n")

    entries = fetch_all_popculture()

    logger.info(f"\nTotal pop culture entries: {len(entries)}")

    # Show breakdown by category
    categories: dict[str, int] = {}
    for entry in entries:
        cat = entry["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat}: {count}")

    # Save
    output_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    logger.info(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
