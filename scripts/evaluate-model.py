#!/usr/bin/env python3
"""Evaluate embedding model quality for word arithmetic.

Tests a set of word combinations and shows results to help
compare different embedding models.

Usage:
    uv run python scripts/evaluate-model.py
"""

import sys
from pathlib import Path

# Add backend src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from wmw.config import Settings
from wmw.engine.mixer import WordMixer
from wmw.vocab.index import VocabIndex

TEST_CASES = [
    # (word_a, word_b, operation, expected-ish)
    ("fire", "water", "add", "steam"),
    ("king", "woman", "add", "queen"),
    ("japan", "food", "add", "sushi"),
    ("ice", "fire", "add", "steam"),
    ("dog", "ocean", "add", "seal"),
    ("king", "man", "subtract", "queen"),
    ("planet", "small", "add", "moon"),
    ("music", "japan", "add", "anime"),
    ("sword", "japan", "add", "samurai"),
    ("cat", "water", "add", "fish"),
]


def main():
    settings = Settings()

    print(f"Loading index from {settings.vocab_index_path}...")
    index = VocabIndex.load(settings.vocab_index_path, settings.vocab_meta_path)
    print(f"Loaded {index.size} words, {index.dimension}d\n")

    mixer = WordMixer(index)

    print(f"{'Operation':<30} {'Expected':<15} {'Got':<20} {'Score':<8} {'Alts'}")
    print("-" * 100)

    for word_a, word_b, op, expected in TEST_CASES:
        if op == "add":
            result = mixer.combine(word_a, word_b)
            op_str = f"{word_a} + {word_b}"
        else:
            result = mixer.subtract(word_a, word_b)
            op_str = f"{word_a} - {word_b}"

        if result is None:
            print(f"{op_str:<30} {expected:<15} {'ERROR':<20}")
            continue

        match = "✓" if result.result == expected else ""
        alts = ", ".join(f"{w}({s:.2f})" for w, _, s in result.alternatives[:3])
        print(
            f"{op_str:<30} {expected:<15} {result.result:<15}{match:<5} "
            f"{result.score:<8.3f} {alts}"
        )


if __name__ == "__main__":
    main()
