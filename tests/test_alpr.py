"""tests/test_alpr.py — ALPR pipeline unit tests."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from alpr.alpr_pipeline import ALPRPipeline, normalize_plate, fallback_plate_id, _FALLBACK_COUNTER


def test_normalize_standard_plate():
    assert normalize_plate("abc 1234") == "ABC 1234"


def test_normalize_strips_noise():
    assert normalize_plate("!ABC-1234.") == "ABC 1234"


def test_normalize_numeric_only():
    result = normalize_plate("12345")
    assert result == "12345"


def test_fallback_id_format():
    # Reset counter for this test run_id
    _FALLBACK_COUNTER.pop("test_run", None)
    fid = fallback_plate_id("test_run", prefix="VEHICLE-ID")
    assert fid == "VEHICLE-ID-0001"
    fid2 = fallback_plate_id("test_run", prefix="VEHICLE-ID")
    assert fid2 == "VEHICLE-ID-0002"


def test_fallback_id_zero_padded():
    _FALLBACK_COUNTER["pad_run"] = 0
    fid = fallback_plate_id("pad_run")
    assert len(fid.split("-")[-1]) == 4


def test_finalize_returns_fallback_when_no_ocr():
    """If no OCR readings, finalize must return a fallback ID."""
    alpr = ALPRPipeline(run_id="fallback_test")
    result = alpr.finalize(track_id=999)
    assert result.is_fallback
    assert "VEHICLE-ID" in result.plate_text


def test_finalize_returns_best_reading():
    """Finalize picks highest-confidence reading."""
    alpr = ALPRPipeline(run_id="best_test")
    alpr._readings[1] = [("ABC 1234", 0.7), ("XYZ 999", 0.9), ("BAD", 0.3)]
    result = alpr.finalize(track_id=1)
    assert result.plate_text == "XYZ 999"
    assert result.confidence == pytest.approx(0.9)
    assert not result.is_fallback


def test_get_best_current_none_when_empty():
    alpr = ALPRPipeline(run_id="empty_test")
    assert alpr.get_best_current(track_id=42) is None


def test_finalized_result_cached():
    """Second call to finalize should return same result without re-running."""
    alpr = ALPRPipeline(run_id="cache_test")
    alpr._readings[55] = [("LEA 5678", 0.88)]
    r1 = alpr.finalize(55)
    r2 = alpr.finalize(55)
    assert r1 is r2
