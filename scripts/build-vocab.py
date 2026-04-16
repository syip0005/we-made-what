#!/usr/bin/env python3
"""Build the FAISS vocabulary index.

Loads English words + cached pop culture data, embeds everything
with the configured model, and saves the FAISS index to disk.

Usage:
    uv run python scripts/build-vocab.py

Environment variables:
    WMW_EMBEDDING_MODEL  - Model name (default: Qwen/Qwen3-Embedding-8B)
    WMW_EMBEDDING_DEVICE - Device (default: cuda)
    WMW_EMBEDDING_DIM    - Matryoshka dimension (default: 512)
    WMW_EMBEDDING_QUANTIZATION - "4bit", "8bit", or unset for full precision (default: 4bit)
    WMW_MIN_WORD_ZIPF    - Min zipf frequency for English words (default: 2.0)
"""

import logging
import sys
import time
from pathlib import Path

# Add backend src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from wmw.config import Settings
from wmw.embedding.registry import create_provider
from wmw.vocab.builder import build_index, load_all_entries

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    settings = Settings()

    logger.info(f"Model: {settings.embedding_model}")
    logger.info(f"Device: {settings.embedding_device}")
    logger.info(f"Dimensions: {settings.embedding_dim}")
    logger.info(f"Quantization: {settings.embedding_quantization or 'none (full precision)'}")
    logger.info(f"Min zipf: {settings.min_word_zipf}\n")

    # Load all word entries
    entries = load_all_entries(settings.data_dir, min_zipf=settings.min_word_zipf)

    # Create embedding provider
    logger.info(f"\nLoading model {settings.embedding_model}...")
    provider = create_provider(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        truncate_dim=settings.embedding_dim,
        quantization=settings.embedding_quantization,
    )
    logger.info(f"Model loaded. Dimension: {provider.dimension}\n")

    # Build index
    start = time.time()
    vocab_index = build_index(entries, provider)
    elapsed = time.time() - start
    logger.info(f"Index built in {elapsed:.1f}s")

    # Save
    vocab_index.save(settings.vocab_index_path, settings.vocab_meta_path)
    logger.info(f"Saved index to {settings.vocab_index_path}")
    logger.info(f"Saved metadata to {settings.vocab_meta_path}")
    logger.info(f"Vocabulary size: {vocab_index.size}")


if __name__ == "__main__":
    main()
