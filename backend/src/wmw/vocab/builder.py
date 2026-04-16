"""Build the FAISS vocabulary index from word lists."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from wmw.embedding.provider import EmbeddingProvider
from wmw.vocab.index import VocabIndex

logger = logging.getLogger(__name__)


def build_index(
    entries: list[dict],
    provider: EmbeddingProvider,
    batch_size: int = 256,
) -> VocabIndex:
    """Build a VocabIndex from a list of {"word": str, "category": str} entries.

    Embeds all words in batches, L2-normalizes, and creates a FAISS IndexFlatIP.
    """
    words = [e["word"] for e in entries]
    categories = [e["category"] for e in entries]

    logger.info(f"Embedding {len(words)} words with {provider.__class__.__name__}...")

    # Embed in batches
    all_embeddings = provider.embed(words, batch_size=batch_size)

    # Ensure L2-normalized (sentence-transformers should already do this,
    # but be safe)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    all_embeddings = all_embeddings / norms

    logger.info(
        f"Building FAISS index: {len(words)} vectors, {provider.dimension}d..."
    )
    index = faiss.IndexFlatIP(provider.dimension)
    index.add(all_embeddings)

    return VocabIndex(words=words, categories=categories, index=index)


def load_all_entries(data_dir: Path, min_zipf: float = 2.0) -> list[dict]:
    """Load all word entries from English words + any cached pop culture data."""
    from wmw.vocab.wordlist import load_english_words

    entries = load_english_words(min_zipf=min_zipf)
    logger.info(f"Loaded {len(entries)} English words")

    # Load cached pop culture data if available
    raw_dir = data_dir / "raw"
    popculture_path = raw_dir / "popculture.json"
    if popculture_path.exists():
        pop_entries = json.loads(popculture_path.read_text())
        logger.info(f"Loaded {len(pop_entries)} pop culture entries from cache")

        # Deduplicate against English words
        seen = {e["word"] for e in entries}
        added = 0
        for entry in pop_entries:
            if entry["word"] not in seen:
                seen.add(entry["word"])
                entries.append(entry)
                added += 1
        logger.info(f"Added {added} unique pop culture entries")
    else:
        logger.info(
            "No pop culture cache found. Run scripts/fetch-popculture.py first."
        )

    logger.info(f"Total vocabulary: {len(entries)} entries")
    return entries
