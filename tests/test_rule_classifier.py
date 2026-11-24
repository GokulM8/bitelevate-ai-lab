import pytest

from app.services.rule_classifier import (
    CATEGORIES, CONFIDENCE_THRESHOLD, KEYWORD_MAPS, NEEDS_AI_INFERENCE,
    classify_ticket, predict,
)


@pytest.mark.parametrize("text,expected", [
    ("I was charged twice, please refund my invoice", "BILLING"),
    ("Please send me a receipt for my subscription", "BILLING"),
    ("I forgot my password and I am locked out", "AUTH"),
    ("Two-factor code never arrives", "AUTH"),
    ("I cannot log in", "AUTH"),
    ("The API returns a 500 error and times out", "TECHNICAL"),
    ("Webhook integration returns an exception", "TECHNICAL"),
    ("My laptop battery is overheating and the screen flickers", "HARDWARE"),
    ("My bluetooth keyboard stopped working", "HARDWARE"),
])
def test_routes_clear_tickets(text, expected):
    result = classify_ticket(text)
    assert result.category == expected
    assert result.confidence >= CONFIDENCE_THRESHOLD
    assert not result.needs_ai


@pytest.mark.parametrize("text", ["Thanks for the great service!", "", "   ", None])
def test_no_keywords_falls_back_to_ai(text):
    result = classify_ticket(text)
    assert result.category == NEEDS_AI_INFERENCE
    assert result.confidence == 0.0
    assert result.best_guess is None
    assert result.needs_ai


def test_mixed_signals_fall_back_but_keep_best_guess():
    result = classify_ticket("payment failed after login")
    assert result.category == NEEDS_AI_INFERENCE
    assert result.best_guess == "AUTH"
    assert result.confidence < CONFIDENCE_THRESHOLD


def test_case_insensitive_and_confidence_value():
    result = classify_ticket("INVOICE")
    assert result.category == "BILLING"
    assert result.confidence == pytest.approx(2 / 2.25, abs=1e-3)


def test_whole_word_matching_charger_is_not_charge():
    result = classify_ticket("I need a new charger")
    assert result.category == "HARDWARE"
    assert "BILLING" not in result.matched


def test_matched_keywords_are_reported():
    assert classify_ticket("Please refund me").matched == {"BILLING": ["refund"]}


def test_custom_threshold_changes_routing():
    assert classify_ticket("The app is slow").category == "TECHNICAL"
    assert classify_ticket("The app is slow", threshold=0.9).category == NEEDS_AI_INFERENCE


def test_confidence_is_between_zero_and_one():
    for text in ["refund invoice billing", "login", "slow", "random words"]:
        assert 0.0 <= classify_ticket(text).confidence <= 1.0


def test_predict_returns_label_string():
    assert predict("refund my invoice") == "BILLING"


def test_keyword_maps_cover_all_categories_without_overlap():
    assert set(KEYWORD_MAPS) == set(CATEGORIES)
    seen = {}
    for category, keywords in KEYWORD_MAPS.items():
        for keyword in keywords:
            assert keyword not in seen, f"'{keyword}' is in {seen[keyword]} and {category}"
            seen[keyword] = category
