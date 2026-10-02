"""Small text/timing utilities used across the platform."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, List, Optional

from utils.timing import Stopwatch, timed  # re-exported for convenience


def humanise(identifier: str) -> str:
    """``annual_income`` -> ``Annual income``."""
    if not identifier:
        return ""
    text = str(identifier).replace("_", " ").replace("-", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:1].upper() + text[1:]


def title_case_snake(name: str) -> str:
    return humanise(name)


def truncate(text: str, max_length: int = 160, suffix: str = "...") -> str:
    text = (text or "").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)].rstrip() + suffix


def pct(value: Optional[float], digits: int = 1) -> str:
    """Format a 0-1 ratio as a percentage string."""
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def number(value: Optional[float], digits: int = 3) -> str:
    """Compact numeric formatting for narratives."""
    if value is None:
        return "n/a"
    try:
        magnitude = abs(float(value))
    except (TypeError, ValueError):
        return str(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if magnitude >= 10_000:
        return f"{value / 1000:.1f}k"
    if magnitude >= 100:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:.{digits}f}"
    if magnitude == 0:
        return "0"
    return f"{value:.{digits + 1}f}"


def pluralise(count: int, singular: str, plural: Optional[str] = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count:,} {word}"


def bullet_list(items: List[str], limit: int = 8) -> str:
    """Render a markdown bullet list, truncating long lists."""
    visible = items[:limit]
    text = "\n".join(f"- {item}" for item in visible)
    if len(items) > limit:
        text += f"\n- ... and {len(items) - limit} more"
    return text


def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and value != value:  # NaN
        return default
    return str(value)


def strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()


__all__ = [
    "Stopwatch",
    "bullet_list",
    "humanise",
    "normalise_whitespace",
    "number",
    "pct",
    "pluralise",
    "safe_str",
    "strip_accents",
    "timed",
    "title_case_snake",
    "truncate",
]
