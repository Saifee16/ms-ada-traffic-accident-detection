from __future__ import annotations

from alpr.alpr_pipeline import ALPRPipeline, normalize_plate


def test_low_confidence_rejected():
    alpr = ALPRPipeline(run_id="low_conf", autoload_reader=False, min_confidence=0.5)
    assert alpr.submit_reading(1, "ABC 1234", 0.2) is None
    assert alpr.get_best_current(1) is None


def test_high_confidence_plate_not_overwritten_by_lower_result():
    alpr = ALPRPipeline(run_id="no_downgrade", autoload_reader=False, min_confidence=0.3)
    assert alpr.submit_reading(1, "ABC 1234", 0.85)[0] == "ABC 1234"
    assert alpr.submit_reading(1, "XYZ 999", 0.4)[0] == "ABC 1234"


def test_fallback_assignment_on_exit():
    alpr = ALPRPipeline(run_id="fallback_exit", autoload_reader=False)
    result = alpr.finalize(42)
    assert result.is_fallback is True
    assert result.plate_text.startswith("VEHICLE-ID-")
    assert len(result.plate_text) == len("VEHICLE-ID-0001")


def test_normalization_fixes_common_confusions():
    assert normalize_plate("abc-1234") == "ABC 1234"
    assert normalize_plate("ABO-I23S") == "ABO 1235"

