from __future__ import annotations

from collections import deque

import numpy as np

from accident.fusion import AccidentEvent, CollisionDetector
from trackers.tracker_manager import TrackedObject

BLANK = np.zeros((360, 640, 3), dtype=np.uint8)


def make_detector() -> CollisionDetector:
    return CollisionDetector(
        fps=10.0,
        confirmation_seconds=0.6,
        min_confirming_frames=4,
        min_signals_required=3,
        severity_threshold=0.28,
        proximity_threshold_px=150,
        iou_spike_threshold=0.05,
        min_contact_iou=0.03,
        contact_edge_px_abs=5,
        velocity_drop_threshold=0.25,
        min_relative_speed=0.12,
        optical_flow_threshold=0.12,
        trajectory_prediction_horizon=6,
        post_impact_seconds=0.4,
        pre_window_seconds=0.7,
        min_track_age_frames=3,
        min_peak_speed_for_collision=4.0,
        high_precision_mode=True,
        cooldown_seconds=30.0,
    )


def make_state_detector(**overrides) -> CollisionDetector:
    params = {
        "fps": 10.0,
        "confirmation_seconds": 0.5,
        "min_confirming_frames": 5,
        "min_signals_required": 3,
        "candidate_threshold": 0.25,
        "confirmed_threshold": 0.35,
        "severity_threshold": 0.35,
        "proximity_threshold_px": 150,
        "iou_spike_threshold": 0.05,
        "min_contact_iou": 0.03,
        "contact_edge_px_abs": 5,
        "velocity_drop_threshold": 0.25,
        "min_relative_speed": 0.12,
        "optical_flow_threshold": 0.12,
        "trajectory_prediction_horizon": 6,
        "post_impact_seconds": 0.4,
        "pre_window_seconds": 0.7,
        "min_track_age_frames": 3,
        "min_peak_speed_for_collision": 4.0,
        "stationary_speed_px_s": 4.0,
        "stationary_window_frames": 6,
        "parked_min_seconds": 0.5,
        "max_candidate_gap_frames": 6,
        "confirmed_display_seconds": 1.0,
        "high_precision_mode": True,
        "cooldown_seconds": 60.0,
    }
    params.update(overrides)
    return CollisionDetector(**params)


def obj(tid: int, cx: float, cy: float, speed: float, w: float = 80, h: float = 60) -> TrackedObject:
    bbox = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)
    return TrackedObject(
        track_id=tid,
        class_id=2,
        class_name="car",
        bbox=bbox,
        centroid=np.array([cx, cy], dtype=float),
        speed_px_s=speed,
        trajectory=deque([(cx, cy)], maxlen=90),
    )


def collision_sequence(det: CollisionDetector, extra_frames: int = 30) -> list[AccidentEvent]:
    events: list[AccidentEvent] = []
    for fid in range(55 + extra_frames):
        if fid < 26:
            ca = 110 + fid * 7.5
            cb = 530 - fid * 7.5
            speed_a = speed_b = 75.0
        else:
            ca = 302.0
            cb = 338.0
            speed_a = speed_b = 1.0
        events.extend(det.process_frame(BLANK, [obj(1, ca, 180, speed_a), obj(2, cb, 180, speed_b)], fid))
    return events


def moving_hits_parked_sequence(det: CollisionDetector) -> list[AccidentEvent]:
    events: list[AccidentEvent] = []
    for fid in range(70):
        if fid < 34:
            ca = 70 + fid * 8
            speed_a = 80.0
        else:
            ca = 342.0
            speed_a = 1.0
        parked = obj(22, 382.0, 180, 0.0)
        moving = obj(21, ca, 180, speed_a)
        events.extend(det.process_frame(BLANK, [moving, parked], fid))
    return events


def test_parallel_close_vehicles_do_not_trigger():
    det = make_detector()
    events: list[AccidentEvent] = []
    for fid in range(60):
        events.extend(
            det.process_frame(
                BLANK,
                [obj(1, 80 + fid * 5, 160, 50), obj(2, 85 + fid * 5, 220, 50)],
                fid,
            )
        )
    assert events == []


def test_collision_with_deceleration_and_stall_triggers():
    events = collision_sequence(make_detector())
    assert len(events) == 1
    event = events[0]
    assert event.confirmed is True
    assert event.event_id.startswith("ACC-")
    assert event.severity_score >= 0.28
    assert event.debug["post_impact_seen"] is True
    assert event.debug["post_impact_reasons"]


def test_occlusion_overlap_without_motion_anomaly_does_not_trigger():
    det = make_detector()
    events: list[AccidentEvent] = []
    for fid in range(60):
        # Same direction and same speed, overlapping boxes caused by perspective/occlusion.
        events.extend(
            det.process_frame(
                BLANK,
                [obj(10, 180 + fid * 3, 180, 30), obj(11, 205 + fid * 3, 180, 30)],
                fid,
            )
        )
    assert events == []


def test_same_direction_overlap_stable_speed_suppressed():
    det = make_state_detector()
    events: list[AccidentEvent] = []
    for fid in range(70):
        # Faster car follows/overlaps a slower car in the same lane perspective.
        # Relative motion + overlap can look severe, but no speed disruption occurs.
        a = obj(61, 120 + fid * 4.0, 180, 40.0, w=90, h=60)
        b = obj(62, 155 + fid * 4.2, 180, 42.0, w=90, h=60)
        events.extend(det.process_frame(BLANK, [a, b], fid))
    assert events == []
    active = det.get_active_candidates()
    assert not any(data["state"] == "CANDIDATE" for data in active.values())


def test_parallel_overtaking_not_candidate():
    det = make_state_detector()
    events: list[AccidentEvent] = []
    for fid in range(80):
        slow = obj(71, 160 + fid * 3.0, 180, 30.0, w=80, h=60)
        fast = obj(72, 80 + fid * 4.5, 185, 45.0, w=80, h=60)
        events.extend(det.process_frame(BLANK, [slow, fast], fid))
    assert events == []
    assert not any(data["state"] == "CANDIDATE" for data in det.get_active_candidates().values())


def test_optical_flow_alone_cannot_confirm():
    det = make_state_detector(
        require_post_impact_validation=True,
        confirmed_threshold=0.30,
        severity_threshold=0.30,
        candidate_threshold=0.20,
    )
    det._optical_flow_score = lambda *args, **kwargs: 1.0
    events: list[AccidentEvent] = []
    for fid in range(60):
        a = obj(81, 240 + fid * 1.0, 180, 10.0)
        b = obj(82, 300 + fid * 1.0, 180, 10.0)
        events.extend(det.process_frame(BLANK, [a, b], fid))
    assert events == []


def test_accident_event_triggers_once_due_to_closed_state_and_cooldown():
    det = make_detector()
    events = collision_sequence(det, extra_frames=80)
    assert len(events) == 1


def test_candidate_confirms_after_required_frames():
    det = make_state_detector()
    events = collision_sequence(det, extra_frames=40)
    assert len(events) == 1
    pair_state = det._pairs[(1, 2)]
    assert pair_state.state in {"CONFIRMED", "CLOSED"}
    assert pair_state.confirmed_frame >= pair_state.candidate_frame


def test_auto_confirmation_uses_k_out_of_n_window_at_30fps():
    det = CollisionDetector(fps=30.0, confirmation_seconds=1.2)
    assert det.confirm_frames == 36
    assert 15 <= det.min_confirming_frames <= 22
    assert det.min_confirming_frames < det.confirm_frames


def test_candidate_start_frame_not_reset():
    det = make_state_detector(confirmed_threshold=0.99, severity_threshold=0.99, require_post_impact_validation=False)
    first_candidate = None
    last_fid = 0
    for fid in range(35):
        ca = 110 + fid * 7.5
        cb = 530 - fid * 7.5
        det.process_frame(BLANK, [obj(31, ca, 180, 75.0), obj(32, cb, 180, 75.0)], fid)
        state = det._pairs.get((31, 32))
        if state and state.state == "CANDIDATE":
            first_candidate = state.candidate_frame
            last_fid = fid
            break
    assert first_candidate is not None
    for fid in range(last_fid + 1, last_fid + 5):
        ca = 110 + fid * 7.5
        cb = 530 - fid * 7.5
        det.process_frame(BLANK, [obj(31, ca, 180, 75.0), obj(32, cb, 180, 75.0)], fid)
    assert det._pairs[(31, 32)].candidate_frame == first_candidate
    assert det._pairs[(31, 32)].event_id


def test_static_static_pair_not_candidate():
    det = make_state_detector()
    for fid in range(30):
        det.process_frame(BLANK, [obj(41, 300, 180, 0.0), obj(42, 340, 180, 0.0)], fid)
    active = det.get_active_candidates()
    assert not any(data["state"] == "CANDIDATE" for data in active.values())


def test_moving_vehicle_hits_parked_vehicle_confirms():
    events = moving_hits_parked_sequence(make_state_detector())
    assert len(events) == 1
    assert events[0].confirmed


def test_normal_passing_near_parked_vehicle_suppressed():
    det = make_state_detector()
    for fid in range(60):
        parked = obj(51, 340.0, 180, 0.0)
        passing = obj(52, 60 + fid * 6, 245, 60.0)
        det.process_frame(BLANK, [parked, passing], fid)
    active = det.get_active_candidates()
    assert not any(data["state"] == "CANDIDATE" for data in active.values())


def test_cooldown_does_not_block_first_confirmation():
    det = make_state_detector(cooldown_seconds=999.0)
    events = collision_sequence(det, extra_frames=40)
    assert len(events) == 1
    assert events[0].confirmed


def test_debug_wait_reason_present(tmp_path):
    debug_path = tmp_path / "debug_events.jsonl"
    det = make_state_detector(
        confirmed_threshold=0.99,
        severity_threshold=0.99,
        require_post_impact_validation=False,
        debug_events=True,
        debug_events_path=str(debug_path),
    )
    collision_sequence(det, extra_frames=20)
    text = debug_path.read_text(encoding="utf-8")
    assert "candidate_wait" in text
    assert "wait_reason" in text


def test_geometry_helpers():
    det = make_detector()
    assert det._box_iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1.0
    assert det._bbox_edge_gap(np.array([0, 0, 10, 10]), np.array([12, 0, 20, 10])) == 2.0
    assert det._bbox_diagonal(np.array([0, 0, 3, 4])) > 5.0


def test_spatial_prefilter_skips_far_pair_explosion(monkeypatch):
    det = make_state_detector(min_track_age_frames=1, pair_grid_cell_px=80, max_pairs_per_frame=20)
    scored = {"count": 0}
    original = det._score_pair

    def counted(*args, **kwargs):
        scored["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(det, "_score_pair", counted)
    for fid in range(6):
        tracks = [obj(100 + i, 80 + i * 240, 180 + (i % 2) * 120, 0.0) for i in range(12)]
        det.process_frame(BLANK, tracks, fid)

    assert scored["count"] == 0


def test_candidate_confirmation_hysteresis_survives_short_dropout():
    det = make_state_detector(
        min_confirming_frames=3,
        confirmation_seconds=0.5,
        state_grace_frames=8,
        require_post_impact_validation=False,
    )
    candidate_seen_at = None
    for fid in range(50):
        if fid < 26:
            ca = 110 + fid * 7.5
            cb = 530 - fid * 7.5
            speed_a = speed_b = 75.0
        else:
            ca = 302.0
            cb = 338.0
            speed_a = speed_b = 1.0
        det.process_frame(BLANK, [obj(201, ca, 180, speed_a), obj(202, cb, 180, speed_b)], fid)
        state = det._pairs.get((201, 202))
        if state and state.state == "CANDIDATE" and state.confirming_frames > 0:
            candidate_seen_at = fid
            break

    assert candidate_seen_at is not None
    before = det._pairs[(201, 202)].confirming_frames
    det.process_frame(BLANK, [obj(201, 302.0, 180, 1.0)], candidate_seen_at + 1)
    state = det._pairs[(201, 202)]
    assert state.state == "CANDIDATE"
    assert 0 < state.confirming_frames <= before


def test_muted_track_rescue_maps_new_id_to_previous_collision_identity():
    det = make_state_detector(min_track_age_frames=1, muted_track_frames=6, muted_base_radius_px=80)
    for fid in range(5):
        det.process_frame(BLANK, [obj(301, 100 + fid * 12, 180, 120.0), obj(302, 380, 180, 0.0)], fid)

    det.process_frame(BLANK, [obj(399, 160, 180, 120.0), obj(302, 380, 180, 0.0)], 5)

    assert det._track_alias.get(399) == 301
