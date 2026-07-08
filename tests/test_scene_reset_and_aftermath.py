from __future__ import annotations

from collections import deque

import numpy as np

from accident.fusion import CollisionDetector
from accident.scene import AccidentSceneDetector, SceneChangeDetector
from trackers.tracker_manager import TrackedObject


FRAME = np.zeros((360, 640, 3), dtype=np.uint8)


def obj(
    tid: int,
    cx: float,
    cy: float,
    speed: float,
    class_id: int = 2,
    class_name: str = "car",
    w: float = 90,
    h: float = 70,
) -> TrackedObject:
    bbox = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)
    return TrackedObject(
        track_id=tid,
        class_id=class_id,
        class_name=class_name,
        bbox=bbox,
        centroid=np.array([cx, cy], dtype=float),
        speed_px_s=speed,
        trajectory=deque([(cx, cy)], maxlen=90),
    )


def test_overlap_only_never_confirms_dynamic_collision():
    det = CollisionDetector(
        fps=10,
        confirmation_seconds=0.5,
        min_confirming_frames=4,
        candidate_threshold=0.20,
        confirmed_threshold=0.30,
        min_track_history_frames=2,
    )
    events = []
    for fid in range(40):
        events.extend(det.process_frame(FRAME, [obj(1, 250, 180, 0), obj(2, 275, 180, 0)], fid))
    assert events == []


def test_transient_collision_like_signal_does_not_confirm_without_persistence():
    det = CollisionDetector(
        fps=10,
        confirmation_seconds=1.0,
        min_confirming_frames=6,
        candidate_threshold=0.20,
        confirmed_threshold=0.30,
        min_track_history_frames=2,
        require_post_impact_validation=False,
    )
    events = []
    for fid in range(12):
        if fid < 6:
            a, b, speed = 120 + fid * 20, 460 - fid * 20, 80
        else:
            a, b, speed = 300, 330, 1
        events.extend(det.process_frame(FRAME, [obj(11, a, 180, speed), obj(12, b, 180, speed)], fid))
    assert events == []


def test_vehicle_pedestrian_proximity_is_not_dynamic_accident():
    det = CollisionDetector(fps=10, min_track_history_frames=2, min_confirming_frames=3)
    events = []
    for fid in range(30):
        car = obj(21, 260 + fid * 2, 180, 20, class_id=2, class_name="car")
        person = obj(22, 290 + fid * 2, 185, 4, class_id=0, class_name="person", w=28, h=80)
        events.extend(det.process_frame(FRAME, [car, person], fid))
    assert events == []
    assert det.get_active_candidates() == {}


def test_scene_cut_reset_clears_dynamic_state():
    det = CollisionDetector(
        fps=10,
        confirmation_seconds=0.5,
        min_confirming_frames=4,
        candidate_threshold=0.20,
        confirmed_threshold=0.30,
        min_track_history_frames=2,
        require_post_impact_validation=False,
    )
    for fid in range(10):
        det.process_frame(FRAME, [obj(31, 120 + fid * 20, 180, 80), obj(32, 460 - fid * 20, 180, 80)], fid)
    assert det._pairs

    cut = SceneChangeDetector(enabled=True, threshold=0.25)
    assert not cut.update(np.zeros_like(FRAME)).is_cut
    white = np.full_like(FRAME, 255)
    result = cut.update(white)
    assert result.is_cut

    det.reset_state()
    assert det._pairs == {}
    assert det.get_active_candidates() == {}


def test_accident_scene_detector_fallback_labels_static_aftermath_only():
    detector = AccidentSceneDetector(
        enabled=True,
        heuristic_enabled=True,
        persistence_frames=3,
        min_stopped_vehicles=2,
        min_bbox_height_ratio=0.08,
        cluster_radius_px=180,
        score_threshold=0.45,
        region_cooldown_frames=20,
    )
    events = []
    for fid in range(8):
        tracks = [
            obj(41, 280, 190, 0, w=130, h=95),
            obj(42, 370, 195, 0, w=120, h=90),
        ]
        events.extend(detector.process_frame(FRAME, tracks, fid))

    assert len(events) == 1
    assert events[0].label == "ACCIDENT_SCENE_DETECTED"
    assert events[0].event_id.startswith("SCENE-")
    assert "persistent_stopped_vehicle_cluster" in events[0].reasons
