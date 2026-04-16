"""Tests for vocab builder."""

import json
import tempfile
from pathlib import Path

from wmw.vocab.builder import build_index, load_all_entries


def test_build_index(fake_provider):
    entries = [
        {"word": "fire", "category": "english"},
        {"word": "water", "category": "english"},
        {"word": "goku", "category": "anime"},
    ]
    index = build_index(entries, fake_provider)
    assert index.size == 3
    assert index.dimension == fake_provider.dimension
    assert index.has_word("fire")
    assert index.has_word("goku")
    assert index.get_category("goku") == "anime"


def test_load_all_entries_english_only():
    """With no raw data files, should return only English words."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        (data_dir / "raw").mkdir()

        entries = load_all_entries(data_dir, min_zipf=5.0)  # high threshold for speed
        assert len(entries) > 50
        assert all(e["category"] == "english" for e in entries)


def test_load_all_entries_with_extra_data():
    """Should merge in data from raw/ JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        raw_dir = data_dir / "raw"
        raw_dir.mkdir()

        # Write a fake pop culture file
        pop_data = [
            {"word": "goku", "category": "anime"},
            {"word": "naruto", "category": "anime"},
        ]
        (raw_dir / "popculture.json").write_text(json.dumps(pop_data))

        entries = load_all_entries(data_dir, min_zipf=5.0)
        words = {e["word"] for e in entries}
        assert "goku" in words
        assert "naruto" in words


def test_load_all_entries_deduplicates():
    """Words that exist in English AND pop culture should not be duplicated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        raw_dir = data_dir / "raw"
        raw_dir.mkdir()

        # "water" is an English word that also appears in pop culture
        pop_data = [{"word": "water", "category": "film"}]
        (raw_dir / "popculture.json").write_text(json.dumps(pop_data))

        entries = load_all_entries(data_dir, min_zipf=5.0)
        water_entries = [e for e in entries if e["word"] == "water"]
        assert len(water_entries) == 1  # not duplicated
