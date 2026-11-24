"""Rule-based baseline classifier for support tickets.

Fast, zero-cost routing using weighted keyword maps. If the rules are not
confident enough (< CONFIDENCE_THRESHOLD), the ticket is marked
NEEDS_AI_INFERENCE so it can be sent to an ML/LLM model.

confidence = top_category_score / (total_score_of_all_categories + SMOOTHING)
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern

NEEDS_AI_INFERENCE = "NEEDS_AI_INFERENCE"
CONFIDENCE_THRESHOLD = 0.60
SMOOTHING = 0.25

CATEGORIES = ("BILLING", "AUTH", "TECHNICAL", "HARDWARE")

KEYWORD_MAPS: Dict[str, Dict[str, float]] = {
    "BILLING": {
        "billing": 2, "invoice": 2, "refund": 2, "charged": 2, "overcharged": 2,
        "charge": 1, "payment": 1.5, "receipt": 1.5, "credit card": 1.5,
        "subscription": 1, "price": 1, "pricing": 1, "renewal": 1, "tax": 1, "coupon": 1,
    },
    "AUTH": {
        "password": 2, "login": 2, "log in": 2, "sign in": 2, "signin": 2,
        "locked out": 2, "two-factor": 2, "2fa": 2, "verification code": 2,
        "sso": 2, "authentication": 2, "otp": 1.5, "token expired": 1.5,
        "session expired": 1.5, "unauthorized": 1.5, "reset": 1, "401": 1,
    },
    "TECHNICAL": {
        "crash": 2, "bug": 2, "timeout": 2, "exception": 2, "stack trace": 2,
        "api": 1.5, "not loading": 1.5, "latency": 1.5, "webhook": 1.5,
        "slow": 1, "500": 1, "integration": 1, "deploy": 1,
        "error": 0.5, "fails": 0.5, "failed": 0.5, "broken": 0.5,
    },
    "HARDWARE": {
        "printer": 2, "scanner": 2, "battery": 2, "keyboard": 2, "router": 2,
        "usb": 2, "charger": 2, "sensor": 2, "hardware": 2, "overheating": 2,
        "device": 1.5, "screen": 1.5, "bluetooth": 1.5, "monitor": 1.5,
        "laptop": 1.5, "cable": 1.5,
    },
}


def _compile(keyword: str) -> Pattern[str]:
    """Whole-word match (optional plural s/es) so 'charge' never matches 'charger'."""
    return re.compile(r"(?<!\w)" + re.escape(keyword) + r"(?:s|es)?(?!\w)")


_PATTERNS: Dict[str, Dict[str, Pattern[str]]] = {
    category: {kw: _compile(kw) for kw in keywords}
    for category, keywords in KEYWORD_MAPS.items()
}


@dataclass(frozen=True)
class ClassificationResult:
    category: str
    confidence: float
    best_guess: Optional[str]
    scores: Dict[str, float]
    matched: Dict[str, List[str]]

    @property
    def needs_ai(self) -> bool:
        return self.category == NEEDS_AI_INFERENCE


def classify_ticket(text: Optional[str], threshold: float = CONFIDENCE_THRESHOLD) -> ClassificationResult:
    """Classify a ticket with keyword rules; fall back below `threshold`."""
    normalized = " ".join((text or "").lower().split())

    scores = {category: 0.0 for category in CATEGORIES}
    matched: Dict[str, List[str]] = {}
    for category in CATEGORIES:
        hits = [kw for kw, pattern in _PATTERNS[category].items() if pattern.search(normalized)]
        if hits:
            matched[category] = hits
            scores[category] = sum(KEYWORD_MAPS[category][kw] for kw in hits)

    total = sum(scores.values())
    if total == 0:
        return ClassificationResult(NEEDS_AI_INFERENCE, 0.0, None, scores, matched)

    best = max(CATEGORIES, key=lambda c: (scores[c], -CATEGORIES.index(c)))
    confidence = round(scores[best] / (total + SMOOTHING), 4)
    category = best if confidence >= threshold else NEEDS_AI_INFERENCE
    return ClassificationResult(category, confidence, best, scores, matched)


def predict(text: Optional[str]) -> str:
    """Convenience wrapper that returns only the label."""
    return classify_ticket(text).category
