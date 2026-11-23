"""Text preprocessing for raw customer support tickets.

`clean_ticket_text` turns a messy ticket (HTML, email headers, signatures,
timestamps, PII) into clean, normalized text that is safe to send to an AI
model. Code snippets and error identifiers (e.g. ERR_404) keep their
original case and spacing.
"""
import html
import re
from typing import Callable, List

# --- Code & identifiers (preserved as-is) ------------------------------------
_HTML_CODE = re.compile(r"<code>(.*?)</code>", re.IGNORECASE | re.DOTALL)
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
# ERR_404, HTTP_500, E1001, NullPointerException
_ERROR_ID = re.compile(
    r"\b(?:[A-Z][A-Z0-9]*_[A-Z0-9_]+"
    r"|[A-Z]+\d+[A-Z0-9]*"
    r"|[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b"
)

# --- HTML ---------------------------------------------------------------------
_HTML_BREAK = re.compile(r"<\s*(?:br|/p|/div|/li)\s*/?\s*>", re.IGNORECASE)
_HTML_TAG = re.compile(r"<.*?>", re.DOTALL)

# --- Email noise ----------------------------------------------------------------
_HEADER_LINE = re.compile(
    r"^\s*(?:from|to|cc|bcc|subject|sent|date|reply-to)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)
_QUOTED_REPLY = re.compile(r"^\s*(?:>.*|on .+ wrote:)\s*$", re.IGNORECASE | re.MULTILINE)
_SIGNATURE = re.compile(
    r"^\s*(?:--\s*"
    r"|(?:best |kind |warm )?regards,?"
    r"|thanks(?: (?:and|&) regards)?,?"
    r"|thank you,?|cheers,?|sincerely,?"
    r"|sent from my .*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# --- Timestamps -----------------------------------------------------------------
_TIMESTAMP = re.compile(
    r"\[?\b\d{4}-\d{2}-\d{2}"                                   # 2026-11-20
    r"(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b\]?"
    r"|\[?\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b\]?"                 # 20/11/2026
    r"|\[?\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s?[ap]m)?\b\]?",  # 2:14:02.108, 9:30 am
    re.IGNORECASE,
)

# --- PII ------------------------------------------------------------------------
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE_CANDIDATE = re.compile(r"(?<!\w)\+?\(?\d[\d\s().-]{6,}\d(?!\w)")
_PHONE_DIGITS = (10, 15)  # E.164 numbers have at most 15 digits

# --- Punctuation & whitespace ------------------------------------------------------
_REPEATED_PUNCT = re.compile(r"([!?.,;:])\1+")
_MIXED_PUNCT = re.compile(r"[!?]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
_WHITESPACE = re.compile(r"\s+")

_PLACEHOLDER = re.compile("(\\d+)")


def _protect(text: str, pattern: re.Pattern, store: List[str],
             transform: Callable[[re.Match], str] = lambda m: m.group(0)) -> str:
    """Swap each match for a placeholder so later steps can't modify it."""
    def repl(match: re.Match) -> str:
        replacement = transform(match)
        if replacement is None:  # transform declined: keep the original text
            return match.group(0)
        store.append(replacement)
        return f"{len(store) - 1}"
    return pattern.sub(repl, text)


def _mask_phone(match: re.Match):
    digits = sum(ch.isdigit() for ch in match.group(0))
    low, high = _PHONE_DIGITS
    return "[PHONE]" if low <= digits <= high else None


def clean_ticket_text(raw_text: str) -> str:
    """Clean and normalize a raw customer support ticket for AI processing.

    Steps: keep code snippets intact, strip HTML, drop email headers, quoted
    replies and signatures, remove timestamps, mask emails and phone numbers,
    lowercase everything except code and error identifiers, collapse repeated
    punctuation, and normalize whitespace.
    """
    if not raw_text:
        return ""

    protected: List[str] = []
    text = _HTML_CODE.sub(lambda m: f"`{m.group(1)}`", raw_text)

    # 1. Keep code snippets exactly as written
    text = _protect(text, _CODE_BLOCK, protected, lambda m: html.unescape(m.group(0)))
    text = _protect(text, _INLINE_CODE, protected, lambda m: html.unescape(m.group(0)))

    # 2. Strip HTML tags and entities (twice catches escaped tags like &lt;b&gt;)
    text = _HTML_BREAK.sub("\n", text)
    text = _HTML_TAG.sub(" ", text)
    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)

    # 3. Drop email headers and quoted replies
    text = _HEADER_LINE.sub("", text)
    text = _QUOTED_REPLY.sub("", text)

    # 4. Cut the signature (only if real content comes before it)
    for match in _SIGNATURE.finditer(text):
        if text[:match.start()].strip():
            text = text[:match.start()]
            break

    # 5. Remove timestamps
    text = _TIMESTAMP.sub(" ", text)

    # 6. Anonymize PII
    text = _protect(text, _EMAIL, protected, lambda m: "[EMAIL]")
    text = _protect(text, _PHONE_CANDIDATE, protected, _mask_phone)

    # 7. Lowercase everything except error identifiers
    text = _protect(text, _ERROR_ID, protected)
    text = text.lower()

    # 8. Collapse repeated punctuation ("!!!" -> "!", "?!?" -> "?")
    text = _REPEATED_PUNCT.sub(r"\1", text)
    text = _MIXED_PUNCT.sub(lambda m: m.group(0)[0], text)

    # 9. Normalize whitespace
    text = _WHITESPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text).strip()

    # 10. Put protected pieces back
    return _PLACEHOLDER.sub(lambda m: protected[int(m.group(1))], text)
