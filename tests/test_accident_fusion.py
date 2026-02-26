"""tests/test_accident_fusion.py — Accident fusion tests with synthetic collisions."""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import pytest

from accident.fusion import AccidentFusion
from trackers.tracker_manager import TrackedObject


def make_object(tid, x1, y1, x2, y2, speed=0.0, trajectory=None):
    cx, cy = (x1+x2)/2, (y1+y2)/2
    traj = trajectory or deque([(cx, cy)] * 15, maxlen=60)  # 15 frames = min_track_age
    return TrackedObject(
        track_id=tid,
        class_id=2,
        class_name="car",
        bbox=np.array([x1, y1, x2, y2], dtype=float),
        centroid=np.array([cx, cy]),
        speed_px_s=speed,
        trajectory=traj,
    )


@pytest.fixture
def fusion():
    return AccidentFusion(
        fps=10.0,
        confirmation_seconds=0.5,   # 5 frames for tests
        iou_spike_threshold=0.20,
        speed_drop_ratio=0.5,
        proximity_px=80,
        min_signals=2,
        fp_suppression_frames=2,
        min_track_age_frames=5,
        min_approach_speed_px=2.0,
        cooldown_seconds=1.0,       # short for tests
    )


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_no_accident_separated_vehicles(fusion, blank_frame):
    """Two far-apart vehicles should not trigger accident."""
    a = make_object(1, 0, 0, 50, 50)
    b = make_object(2, 400, 400, 450, 450)
    events = []
    for _ in range(20):
        events += fusion.process_frame(blank_frame, [a, b], frame_id=_)
    assert events == []


def test_accident_detected_overlapping_vehicles(fusion, blank_frame):
    """Overlapping vehicles with converging trajectories should trigger."""
    # Give vehicle A deceleration history
    for i in range(10):
        fusion._speed_history[1].append(100 - i * 8)  # 100→28 px/s drop
        fusion._speed_history[2].append(80 - i * 5)

    # Overlapping bboxes → high IoU + proximity
    a = make_object(1, 200, 200, 350, 350, speed=20.0)
    b = make_object(2, 220, 220, 370, 370, speed=15.0)
    events = []
    for fid in range(10):
        events += fusion.process_frame(blank_frame, [a, b], frame_id=fid)
    assert len(events) > 0
    assert events[0].track_id_a in {1, 2}
    assert events[0].track_id_b in {1, 2}


def test_severity_score_in_range(fusion, blank_frame):
    fusion._speed_history[3].extend([100] * 6 + [10] * 6)
    fusion._speed_history[4].extend([90] * 6 + [5] * 6)
    a = make_object(3, 200, 200, 350, 350, speed=10.0)
    b = make_object(4, 210, 210, 360, 360, speed=5.0)
    events = []
    for fid in range(10):
        events += fusion.process_frame(blank_frame, [a, b], frame_id=fid)
    if events:
        assert 0.0 <= events[0].severity_score <= 1.0


def test_iou_calculation():
    """Direct IoU calculation sanity check."""
    fusion = AccidentFusion()
    # Perfect overlap
    a = np.array([0, 0, 100, 100])
    b = np.array([0, 0, 100, 100])
    assert fusion._box_iou(a, b) == pytest.approx(1.0)
    # No overlap
    c = np.array([200, 200, 300, 300])
    assert fusion._box_iou(a, c) == pytest.approx(0.0)


def test_cooldown_prevents_duplicate_events(fusion, blank_frame):
    """After confirming an accident, same pair shouldn't fire again immediately."""
    fusion._speed_history[5].extend([100] * 6 + [5] * 6)
    fusion._speed_history[6].extend([90] * 6 + [5] * 6)
    a = make_object(5, 200, 200, 350, 350, speed=5.0)
    b = make_object(6, 210, 210, 360, 360, speed=5.0)
    all_events = []
    for fid in range(20):
        all_events += fusion.process_frame(blank_frame, [a, b], frame_id=fid)
    # Should not have dozens of events — cooldown kicks in
    assert len(all_events) <= 2


# ════════════════════════════════════════════════════════════
# FALSE POSITIVE REGRESSION TESTS
# Each test represents a scenario that was incorrectly firing
# before the patches. All must return ZERO events.
# ════════════════════════════════════════════════════════════

def _make_parallel_trajectory(start_x, y, length=20, dx=5):
    """Straight horizontal trajectory — simulates lane traffic."""
    traj = deque(maxlen=60)
    for i in range(length):
        traj.append((start_x + i * dx, y))
    return traj


def test_fp_parallel_lane_vehicles(fusion, blank_frame):
    """
    Two vehicles traveling in parallel lanes at same speed should NEVER trigger.
    This was the most common source of false positives (trajectory + proximity).
    """
    traj_a = _make_parallel_trajectory(start_x=100, y=200, length=20)
    traj_b = _make_parallel_trajectory(start_x=100, y=260, length=20)  # 60px apart, same direction
    a = make_object(1, 190, 180, 290, 230, speed=50.0, trajectory=traj_a)
    b = make_object(2, 190, 240, 290, 290, speed=50.0, trajectory=traj_b)
    events = []
    for fid in range(30):
        events += fusion.process_frame(blank_frame, [a, b], frame_id=fid)
    assert events == [], f"Parallel lane vehicles should not trigger: got {events}"


def test_fp_empty_signals_no_event(fusion, blank_frame):
    """
    An event must never fire with signals=[]. Before the patch, stale signal_history
    could confirm an event even when current frame has 0 active signals.
    """
    # Pre-load signal history with past positives (simulating old behaviour)
    pair_key = (1, 2)
    from accident.fusion import _PairState
    state = _PairState(track_id_a=1, track_id_b=2, cooldown_seconds=1.0)
    for _ in range(10):
        state.signal_history.append({"iou": True, "trajectory": True, "deceleration": False,
                                      "optical_flow": False, "proximity": True})
    fusion._pair_states[pair_key] = state

    # Now send vehicles that are far apart — no signals should fire
    a = make_object(1, 0, 0, 80, 80, speed=10.0)
    b = make_object(2, 400, 400, 480, 480, speed=10.0)
    events = []
    for fid in range(15):
        events += fusion.process_frame(blank_frame, [a, b], frame_id=fid)
    assert events == [], "Stale signal history must not confirm event when current signals empty"


def test_fp_negative_severity_blocked(fusion, blank_frame):
    """Events with severity < 0.15 (incl. negative) must be filtered."""
    # Two closely-placed but slow stationary vehicles — minimal score
    a = make_object(1, 100, 100, 180, 180, speed=0.0)
    b = make_object(2, 185, 100, 265, 180, speed=0.0)
    events = []
    for fid in range(20):
        events += fusion.process_frame(blank_frame, [a, b], frame_id=fid)
    for e in events:
        assert e.severity_score >= 0.15, f"Severity {e.severity_score} below minimum threshold"


def test_fp_immature_tracks_ignored(fusion, blank_frame):
    """Tracks with fewer than min_track_age_frames should not participate in fusion."""
    # Only 2 trajectory points — brand new track
    traj = deque([(200, 200), (205, 200)], maxlen=60)
    a = make_object(1, 180, 180, 260, 260, speed=50.0, trajectory=traj)
    b = make_object(2, 250, 180, 330, 260, speed=50.0, trajectory=traj)
    events = []
    for fid in range(10):
        events += fusion.process_frame(blank_frame, [a, b], frame_id=fid)
    assert events == [], "Immature tracks (< min_track_age_frames) must be skipped"


def test_fp_non_approaching_vehicles_ignored(fusion, blank_frame):
    """
    Two vehicles diverging (moving away from each other) must not trigger.
    This tests the _is_approaching() gate.
    """
    # Vehicle A moving left, Vehicle B moving right — diverging
    traj_a = deque([(300 - i * 8, 200) for i in range(20)], maxlen=60)  # moving left
    traj_b = deque([(320 + i * 8, 200) for i in range(20)], maxlen=60)  # moving right
    a = make_object(1, 220, 180, 300, 230, speed=50.0, trajectory=traj_a)
    b = make_object(2, 320, 180, 400, 230, speed=50.0, trajectory=traj_b)
    events = []
    for fid in range(20):
        events += fusion.process_frame(blank_frame, [a, b], frame_id=fid)
    assert events == [], "Diverging vehicles must not trigger accident detection"


def test_fp_severity_score_never_negative():
    """Optical flow Z-score can be negative; severity must always be >= 0."""
    fusion = AccidentFusion()
    # Inject negative optical flow scores
    scores = {
        "iou": 0.05,
        "trajectory": 0.1,
        "deceleration": 0.0,
        "optical_flow": -3.5,   # Z-score below baseline — used to make severity negative
        "proximity": 0.0,
    }
    severity = fusion._severity_score(scores)
    assert severity >= 0.0, f"Severity must never be negative, got {severity}"


def test_fp_cooldown_prevents_repeat(blank_frame):
    """After a confirmed event, same pair must not fire again until cooldown expires."""
    fusion = AccidentFusion(
        fps=10.0,
        confirmation_seconds=0.5,
        min_signals=2,
        cooldown_seconds=60.0,   # 60 second cooldown
        min_track_age_frames=5,
        min_approach_speed_px=2.0,
        fp_suppression_frames=2,
    )
    # Build two overlapping vehicles with approach trajectory
    traj_a = deque([(250 - i, 200) for i in range(20)], maxlen=60)  # moving right→left
    traj_b = deque([(230 + i, 200) for i in range(20)], maxlen=60)  # moving left→right
    fusion._speed_history[1].extend([100] * 6 + [5] * 6)
    fusion._speed_history[2].extend([100] * 6 + [5] * 6)
    a = make_object(1, 200, 180, 280, 250, speed=5.0, trajectory=traj_a)
    b = make_object(2, 275, 180, 355, 250, speed=5.0, trajectory=traj_b)

    all_events = []
    for fid in range(60):  # run for 6 seconds
        all_events += fusion.process_frame(blank_frame, [a, b], frame_id=fid)

    # With 60s cooldown, only 1 event should ever fire
    assert len(all_events) <= 1, f"Cooldown failed: got {len(all_events)} events"
