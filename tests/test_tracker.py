"""tests/test_tracker.py — Tracker ID persistence and re-ID gap tests."""
from __future__ import annotations

import numpy as np
import pytest

from trackers.tracker_manager import TrackerManager


@pytest.fixture
def tracker():
    return TrackerManager(reid_gap_seconds=1.0, fps=10.0)  # gap = 10 frames


def _make_update(tracker, tid, x1=100, y1=100, x2=200, y2=200):
    return tracker.update(
        track_ids=np.array([tid]),
        bboxes=np.array([[x1, y1, x2, y2]], dtype=float),
        class_ids=np.array([2]),
        class_names=["car"],
    )


def test_new_track_registered(tracker):
    objects = _make_update(tracker, tid=1)
    assert len(objects) == 1
    assert objects[0].track_id == 1


def test_trajectory_accumulates(tracker):
    for _ in range(5):
        _make_update(tracker, tid=1)
    assert len(tracker.tracks[1].trajectory) == 5


def test_id_persists_across_frames(tracker):
    for i in range(10):
        objects = _make_update(tracker, tid=42, x1=i*2, x2=i*2+100)
    assert 42 in tracker.tracks
    assert tracker.tracks[42].track_id == 42


def test_stale_track_pruned_after_gap(tracker):
    """Track should be removed after reid_gap_frames without update."""
    _make_update(tracker, tid=5)
    # Advance frame counter beyond gap by calling update with a different ID
    for _ in range(15):  # gap = 10 frames at fps=10
        _make_update(tracker, tid=99)
    # tid=5 should now be pruned
    assert 5 not in tracker.tracks


def test_speed_estimated(tracker):
    # Move vehicle 10px per frame for 6 frames
    for i in range(6):
        tracker.update(
            track_ids=np.array([10]),
            bboxes=np.array([[i*10, 100, i*10+100, 200]], dtype=float),
            class_ids=np.array([2]),
            class_names=["car"],
        )
    obj = tracker.tracks[10]
    # fps=10, movement=10px/frame → speed ≈ 100 px/s
    assert obj.speed_px_s > 0


def test_multiple_tracks(tracker):
    tracker.update(
        track_ids=np.array([1, 2, 3]),
        bboxes=np.array([[0,0,50,50],[100,100,150,150],[200,200,250,250]], dtype=float),
        class_ids=np.array([2, 3, 5]),
        class_names=["car", "motorcycle", "bus"],
    )
    assert len(tracker.tracks) == 3
