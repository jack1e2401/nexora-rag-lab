"""Helpers that work only with model tokens."""

from typing import Any


def count_tokens(text: str, tokenizer: Any) -> int:
    """Count content tokens without adding model special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_special_tokens(tokenizer: Any) -> int:
    """Return how many special tokens the model adds to one text input."""
    count_special_tokens = getattr(tokenizer, "num_special_tokens_to_add", None)

    if count_special_tokens is None:
        return 0

    return count_special_tokens(pair=False)
