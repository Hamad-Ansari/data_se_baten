"""Unit tests for the cross-cutting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.errors import (
    DataSenseError,
    EmptyDatasetError,
    SQLAccessError,
    SchemaError,
    UnsupportedFormatError,
    describe_exception,
    user_message_for,
)
from utils.files import (
    append_jsonl,
    ensure_directory,
    read_json,
    read_jsonl,
    sanitize_filename,
    slugify,
    timestamp_slug,
    write_json,
)
from utils.serialization import to_jsonable
from utils.timing import Stopwatch, format_duration, timed
from utils.validation import validate_sql_query, validate_upload_filename


def test_errors_carry_user_messages() -> None:
    error = SchemaError("internal detail", user_message="Please pick a target column.")
    payload = error.to_dict()
    assert payload["error"] == "SchemaError"
    assert payload["message"] == "Please pick a target column."
    assert user_message_for(error) == "Please pick a target column."
    described = describe_exception(error)
    assert described["error"] == "SchemaError"
    assert described["message"] == "Please pick a target column."

    plain = describe_exception(ValueError("boom"))
    assert plain["error"] == "ValueError"
    assert "The technical details were logged" in plain["message"]
    assert plain["detail"] == "boom"


def test_plain_exception_gets_a_friendly_message() -> None:
    message = user_message_for(ValueError("boom"))
    assert isinstance(message, str) and message


def test_empty_dataset_error_is_a_data_sense_error() -> None:
    assert issubclass(EmptyDatasetError, DataSenseError)


def test_slugify_and_sanitize() -> None:
    assert slugify("Customer Churn Rates!") == "customer-churn-rates"
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("my data (final).csv") == "my_data_final.csv"
    assert len(timestamp_slug()) == 15 and timestamp_slug().count("-") == 1


def test_json_and_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    write_json(path, {"a": 1, "b": [1, 2]})
    assert read_json(path) == {"a": 1, "b": [1, 2]}
    assert read_json(tmp_path / "missing.json", default={"x": 1}) == {"x": 1}

    log = tmp_path / "events.jsonl"
    append_jsonl(log, {"step": 1})
    append_jsonl(log, {"step": 2})
    assert [row["step"] for row in read_jsonl(log)] == [1, 2]
    assert ensure_directory(tmp_path / "nested" / "dir").exists()


def test_to_jsonable_handles_numpy_and_dataclasses() -> None:
    import numpy as np
    from dataclasses import dataclass

    @dataclass
    class Item:
        name: str
        value: float

    payload = to_jsonable(
        {"array": np.array([1, 2]), "scalar": np.float32(1.5), "item": Item("x", 2.0),
         "nan": float("nan")}
    )
    assert payload["array"] == [1, 2]
    assert payload["scalar"] == pytest.approx(1.5)
    assert payload["item"] == {"name": "x", "value": 2.0}
    assert payload["nan"] is None


def test_timing_helpers() -> None:
    with Stopwatch() as watch:
        pass
    assert watch.elapsed_ms >= 0
    with timed() as timed_watch:
        pass
    assert timed_watch.as_dict()["elapsed_ms"] >= 0
    assert format_duration(0.25).endswith("ms")
    assert format_duration(64) == "1m 04s"


def test_upload_filename_validation(settings) -> None:
    assert validate_upload_filename("my data.csv") == "my_data.csv"
    with pytest.raises(UnsupportedFormatError):
        validate_upload_filename("payload.exe")


def test_sql_guard_blocks_writes() -> None:
    assert validate_sql_query("SELECT * FROM customers") == "SELECT * FROM customers"
    with pytest.raises(SQLAccessError):
        validate_sql_query("DROP TABLE customers")
