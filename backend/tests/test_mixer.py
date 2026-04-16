"""Tests for WordMixer."""

from wmw.engine.mixer import WordMixer, _clean_result


def test_combine_returns_result(fake_llm):
    mixer = WordMixer(fake_llm)
    result = mixer.combine("fire", "water")
    assert result.operation == "add"
    assert result.inputs == ("fire", "water")
    assert result.result == "steam"


def test_subtract_returns_result(fake_llm):
    mixer = WordMixer(fake_llm)
    result = mixer.subtract("king", "man")
    assert result.operation == "subtract"
    assert result.inputs == ("king", "man")
    assert result.result  # some non-empty result


def test_combine_caches_results(fake_llm):
    mixer = WordMixer(fake_llm)
    r1 = mixer.combine("fire", "water")
    r2 = mixer.combine("fire", "water")
    assert r1.result == r2.result
    assert mixer.cache_size == 1


def test_cache_is_case_insensitive(fake_llm):
    mixer = WordMixer(fake_llm)
    r1 = mixer.combine("Fire", "Water")
    r2 = mixer.combine("fire", "water")
    assert r1.result == r2.result
    assert mixer.cache_size == 1


def test_combine_and_subtract_cache_separately(fake_llm):
    mixer = WordMixer(fake_llm)
    mixer.combine("fire", "water")
    mixer.subtract("fire", "water")
    assert mixer.cache_size == 2


def test_clean_result_basic():
    assert _clean_result("Steam") == "steam"
    assert _clean_result("  Steam  ") == "steam"


def test_clean_result_strips_quotes():
    assert _clean_result('"steam"') == "steam"
    assert _clean_result("**steam**") == "steam"


def test_clean_result_takes_first_line():
    assert _clean_result("steam\nsome explanation") == "steam"


def test_clean_result_truncates_long():
    assert _clean_result("one two three four five") == "one two three"


def test_clean_result_fallback():
    assert _clean_result("") == "nothing"
    assert _clean_result("   ") == "nothing"
