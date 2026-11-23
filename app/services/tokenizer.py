"""Lightweight, zero-dependency tokenizer for SupportPulse.

Mimics what a real transformer tokenizer (e.g. BERT/MiniLM) returns:
- input_ids:      integers the model reads, starting with [CLS] and ending with [SEP]
- attention_mask: 1 for real tokens, 0 for padding

Words are split into subword pieces (word start + "##" continuation chunks)
and mapped to IDs with a stable hash, so the same text always gives the same IDs.
"""
import hashlib
import re
from dataclasses import dataclass
from typing import List

PAD_ID, UNK_ID, CLS_ID, SEP_ID = 0, 100, 101, 102
_RESERVED_IDS = 1000          # IDs below this are kept for special tokens
VOCAB_SIZE = 30522            # same size as the BERT/MiniLM vocabulary
MAX_LENGTH = 128
SUBWORD_LEN = 4               # max characters per subword piece

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


@dataclass
class TokenizedText:
    tokens: List[str]
    input_ids: List[int]
    attention_mask: List[int]


def split_subwords(word: str) -> List[str]:
    """'authentication' -> ['auth', '##enti', '##cati', '##on']"""
    pieces = [word[i:i + SUBWORD_LEN] for i in range(0, len(word), SUBWORD_LEN)]
    return [pieces[0]] + [f"##{p}" for p in pieces[1:]]


def token_to_id(token: str) -> int:
    """Map a token to a stable ID in [_RESERVED_IDS, VOCAB_SIZE)."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return _RESERVED_IDS + int.from_bytes(digest[:4], "big") % (VOCAB_SIZE - _RESERVED_IDS)


def tokenize(text: str, max_length: int = MAX_LENGTH, pad: bool = True) -> TokenizedText:
    """Turn text into [CLS] + subword tokens + [SEP], truncated/padded to max_length."""
    if max_length < 2:
        raise ValueError("max_length must be at least 2 to fit [CLS] and [SEP]")

    tokens: List[str] = []
    for word in _WORD_RE.findall((text or "").lower()):
        tokens.extend(split_subwords(word))
    tokens = tokens[: max_length - 2]  # leave room for [CLS] and [SEP]

    input_ids = [CLS_ID] + [token_to_id(t) for t in tokens] + [SEP_ID]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    attention_mask = [1] * len(input_ids)

    if pad:
        padding = max_length - len(input_ids)
        input_ids += [PAD_ID] * padding
        attention_mask += [0] * padding
        tokens += ["[PAD]"] * padding

    return TokenizedText(tokens=tokens, input_ids=input_ids, attention_mask=attention_mask)