"""Timing helpers used for observability (elapsed time per node / tool)."""

from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, Iterator


class Stopwatch:
    """Context manager that records wall-clock elapsed time in milliseconds."""

    def __init__(self) -> None:
        self.start: float = time.perf_counter()
        self.elapsed_ms: float = 0.0

    def stop(self) -> float:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0
        return self.elapsed_ms

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start

    def as_dict(self) -> Dict[str, float]:
        self.stop()
        return {"elapsed_ms": round(self.elapsed_ms, 3)}

    def __enter__(self) -> "Stopwatch":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


@contextlib.contextmanager
def timed() -> Iterator[Stopwatch]:
    """``with timed() as watch: ...`` -- always stops, even on exception."""
    watch = Stopwatch()
    try:
        yield watch
    finally:
        watch.stop()


def format_duration(seconds: float) -> str:
    """Human readable duration (``1m 04s``)."""
    seconds = max(float(seconds), 0.0)
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


__all__ = ["Stopwatch", "format_duration", "timed"]
