from __future__ import annotations

import numpy as np

from trackers.tracker_manager import TrackerManager


def update_one(tracker: TrackerManager, tid: int, x: float):
    return tracker.update(
        np.array([tid]),
        np.array([[x, 100, x + 80, 160]], dtype=float),
        np.array([2]),
        ["car"],
        np.array([0.9]),
    )


def update_empty(tracker: TrackerManager):
    return tracker.update(np.array([]), np.empty((0, 4)), np.array([]), [], np.array([]))


def test_track_keeps_same_id_during_short_occlusion():
    tracker = TrackerManager(reid_gap_seconds=1.0, fps=10.0, cluster_radius_px=60)
    update_one(tracker, 1, 100)
    update_one(tracker, 1, 106)
    update_empty(tracker)
    tracks = update_one(tracker, 99, 112)
    assert tracks[0].track_id == 1


def test_long_disappearance_creates_new_track_id():
    tracker = TrackerManager(reid_gap_seconds=0.3, fps=10.0, cluster_radius_px=60)
    update_one(tracker, 1, 100)
    for _ in range(5):
        update_empty(tracker)
    tracks = update_one(tracker, 99, 105)
    assert tracks[0].track_id == 99
    assert tracker.pop_exited_tracks()

