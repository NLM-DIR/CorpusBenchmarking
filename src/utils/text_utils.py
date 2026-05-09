import re
from typing import Any

from nltk.tokenize import sent_tokenize


def extract_sentences_from_texts(texts: list[str]) -> list[str]:
    sentences: list[str] = list()
    for text in texts:
        sentences.extend(sent_tokenize(text))
    return sentences


def parse_tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.replace("_", " ").lower())


def extract_tokens_from_texts(texts: list[str]) -> list[str]:
    tokens: list[str] = list()
    for text in texts:
        tokens.extend(parse_tokens(text))
    return tokens


def str_to_bool(val):
    """Convert a string representation of truth to True or False."""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        logger.warning("Invalid boolean value encountered: %s", val)
        raise ValueError(f"Invalid boolean value: {val}")


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value or None


def dedupe_strings(values: list[Any]) -> list[str]:
    seen = set()
    deduped: list[str] = []
    for value in values:
        value = clean_text(value)
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped
