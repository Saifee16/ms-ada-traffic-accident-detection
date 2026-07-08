"""Multi-stage accident detection with pair-level event state.

The detector keeps the existing public API used by the pipeline:

    AccidentFusion(...).process_frame(frame, tracked_objects, frame_id)

Internally each vehicle pair moves through NORMAL -> SUSPECT -> CANDIDATE ->
CONFIRMED -> CLOSED.  Confirmation requires persistent multi-signal evidence
and a post-impact anomaly, which is the main guard against close traffic,
overtaking, occlusion overlap, and parked vehicles.
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from trackers.tracker_manager import TrackedObject
from utils.logger import get_logger

logger = get_logger(__name__)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class EventState(str, Enum):
    NORMAL = "NORMAL"
    SUSPECT = "SUSPECT"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    CLOSED = "CLOSED"


@dataclass
class AccidentEvent:
    track_id_a: int
    track_id_b: int
    timestamp: float
    frame_id: int
    severity_score: float
    iou_at_collision: float
    motion_anomaly_score: float
    signals_triggered: List[str] = field(default_factory=list)
    confirmed: bool = False
    event_id: str = ""
    signal_scores: Dict[str, float] = field(default_factory=dict)
    debug: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class FusionFrameResult:
    frame_id: int
    frame: np.ndarray
    tracked_objects: List[TrackedObject]
    events: List[AccidentEvent]


@dataclass
class SignalEvidence:
    frame_id: int
    pair_key: Tuple[int, int]
    state: EventState
    signal_scores: Dict[str, float]
    signal_flags: Dict[str, bool]
    severity: float
    iou: float
    distance_px: float
    edge_gap_px: float
    speed_a: float
    speed_b: float
    bbox_a: List[float]
    bbox_b: List[float]
    reasons: List[str] = field(default_factory=list)
    suppressed_reason: str = ""

    @property
    def active_signal_count(self) -> int:
        return sum(1 for v in self.signal_flags.values() if v)

    def to_json(self) -> Dict[str, Any]:
        return {
            "frame_number": self.frame_id,
            "pair": list(self.pair_key),
            "state": self.state.value,
            "signal_scores": self.signal_scores,
            "signal_flags": self.signal_flags,
            "severity": round(self.severity, 4),
            "iou": round(self.iou, 4),
            "distance_px": round(self.distance_px, 2),
            "edge_gap_px": round(self.edge_gap_px, 2),
            "speed_a": round(self.speed_a, 2),
            "speed_b": round(self.speed_b, 2),
            "bbox_a": [round(x, 2) for x in self.bbox_a],
            "bbox_b": [round(x, 2) for x in self.bbox_b],
            "reasons": self.reasons,
            "suppressed_reason": self.suppressed_reason,
            "active_signal_count": self.active_signal_count,
        }


@dataclass
class _PairState:
    track_id_a: int
    track_id_b: int
    state: EventState = EventState.NORMAL
    first_seen_frame: int = -1
    last_seen_frame: int = -1
    suspect_frame: int = -1
    candidate_frame: int = -1
    candidate_start_time: float = 0.0
    impact_frame: int = -1
    confirmed_frame: int = -1
    confirmed_until_frame: int = -1
    closed_frame: int = -1
    event_id: str = ""
    last_confirmed_time: float = 0.0
    evidence: Deque[SignalEvidence] = field(default_factory=lambda: deque(maxlen=180))
    iou_history: Deque[float] = field(default_factory=lambda: deque(maxlen=180))
    distance_history: Deque[float] = field(default_factory=lambda: deque(maxlen=180))
    confirmation_window: Deque[int] = field(default_factory=lambda: deque(maxlen=180))
    hard_impact_window: Deque[int] = field(default_factory=lambda: deque(maxlen=180))
    confirming_frames: int = 0
    hard_impact_frames: int = 0
    weak_frames: int = 0
    missing_frames: int = 0
    last_positive_frame: int = -1
    max_severity_seen: float = 0.0
    post_impact_seen: bool = False
    post_impact_reasons: List[str] = field(default_factory=list)
    emitted: bool = False
    last_wait_reason: str = ""
    cooldown_seconds: float = 30.0

    def recent(self, n: int) -> List[SignalEvidence]:
        if n <= 0:
            return []
        return list(self.evidence)[-n:]


@dataclass
class _TrackSnapshot:
    track_id: int
    class_id: int
    center: np.ndarray
    bbox: np.ndarray
    velocity_per_frame: np.ndarray
    frame_id: int

    @property
    def width(self) -> float:
        return max(1.0, float(self.bbox[2] - self.bbox[0]))

    @property
    def height(self) -> float:
        return max(1.0, float(self.bbox[3] - self.bbox[1]))

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1.0)

    @property
    def diagonal(self) -> float:
        return float(math.hypot(self.width, self.height)) + 1e-6


@dataclass
class _AsyncFusionTask:
    frame_id: int
    frame: np.ndarray
    tracked_objects: List[TrackedObject]
    debug: bool


class CollisionDetector:
    """Explainable, tunable multi-stage accident detector."""

    def __init__(
        self,
        fps: float = 15.0,
        confirmation_seconds: float = 1.5,
        min_confirming_frames: Optional[int] = None,
        min_signals_required: int = 3,
        severity_threshold: float = 0.45,
        candidate_threshold: Optional[float] = None,
        confirmed_threshold: Optional[float] = None,
        cooldown_seconds: float = 30.0,
        proximity_threshold_px: float = 120.0,
        iou_spike_threshold: float = 0.08,
        min_contact_iou: float = 0.05,
        contact_edge_px_abs: float = 6.0,
        velocity_drop_threshold: float = 0.30,
        min_relative_speed: float = 0.16,
        min_rel_delta_v: Optional[float] = None,
        optical_flow_threshold: float = 0.12,
        flow_spike_threshold: Optional[float] = None,
        trajectory_prediction_horizon: int = 8,
        prediction_error_threshold: float = 0.35,
        post_impact_seconds: float = 0.6,
        pre_window_seconds: float = 1.0,
        moving_threshold: float = 0.012,
        min_peak_speed_for_collision: float = 6.0,
        stationary_speed_px_s: float = 4.0,
        stationary_window_frames: int = 15,
        parked_min_seconds: float = 2.0,
        require_post_impact_validation: bool = True,
        require_hard_impact_signal: bool = True,
        min_hard_impact_signals: int = 1,
        min_supporting_impact_signals: int = 1,
        suppress_same_direction_traffic: bool = True,
        suppress_static_static: bool = True,
        suppress_normal_passing_parked: bool = True,
        closing_distance_enabled: bool = True,
        closing_distance_min_drop_px: float = 12.0,
        closing_distance_min_drop_ratio: float = 0.12,
        max_candidate_gap_frames: int = 12,
        state_grace_frames: Optional[int] = None,
        pair_grid_cell_px: Optional[float] = None,
        max_pair_distance_px: Optional[float] = None,
        max_pairs_per_frame: int = 80,
        debug_record_normal_interval: int = 30,
        min_bbox_height_ratio: float = 0.015,
        normalized_speed_enabled: bool = True,
        road_roi_enabled: bool = False,
        pair_cooldown_frames: Optional[int] = None,
        candidate_persistence_frames: Optional[int] = None,
        confirmation_window_frames: Optional[int] = None,
        confirmation_score_threshold: Optional[float] = None,
        muted_track_frames: int = 20,
        muted_base_radius_px: float = 60.0,
        muted_velocity_radius_scale: float = 0.75,
        muted_aspect_ratio_delta: float = 0.35,
        muted_area_ratio_delta: float = 0.60,
        async_queue_size: int = 4,
        confirmed_display_seconds: float = 3.0,
        same_direction_cosine: float = 0.88,
        high_precision_mode: bool = True,
        debug_events: bool = False,
        debug_events_path: str = "output/debug_events.jsonl",
        severity_weights: Optional[Dict[str, float]] = None,
        # Legacy names kept so the existing pipeline/config can still construct us.
        speed_drop_ratio: Optional[float] = None,
        optical_flow_zscore: Optional[float] = None,
        proximity_px: Optional[float] = None,
        min_signals: Optional[int] = None,
        min_track_age_frames: int = 5,
        min_track_history_frames: Optional[int] = None,
        fp_suppression_frames: int = 2,
        grazing_iou_limit: float = 0.08,
        approach_angle_deg: float = 20.0,
        debug_events_dir: Optional[str] = None,
        **_legacy_kwargs: Any,
    ) -> None:
        self.fps = max(float(fps), 1.0)
        self.confirmation_seconds = max(float(confirmation_seconds), 0.1)
        self.confirm_frames = max(
            1,
            int(confirmation_window_frames)
            if confirmation_window_frames is not None
            else int(math.ceil(self.confirmation_seconds * self.fps)),
        )
        if min_confirming_frames is None:
            min_confirming_frames = (
                int(candidate_persistence_frames)
                if candidate_persistence_frames is not None
                else max(5, int(math.ceil(self.confirm_frames * 0.55)))
            )
        self.min_confirming_frames = max(1, int(min_confirming_frames))

        self.min_signals_required = int(min_signals if min_signals is not None else min_signals_required)
        if high_precision_mode:
            self.min_signals_required = max(self.min_signals_required, 3)
        self.min_signals = self.min_signals_required  # backward-compatible attribute

        self.candidate_threshold = float(candidate_threshold if candidate_threshold is not None else min(0.35, severity_threshold))
        self.confirmed_threshold = float(
            confirmation_score_threshold
            if confirmation_score_threshold is not None
            else confirmed_threshold
            if confirmed_threshold is not None
            else severity_threshold
        )
        self.severity_threshold = self.confirmed_threshold
        self.min_severity = self.confirmed_threshold  # backward-compatible attribute
        self.cooldown_seconds = float(cooldown_seconds)

        self.proximity_threshold_px = float(proximity_px or proximity_threshold_px)
        self.iou_spike_threshold = float(iou_spike_threshold)
        self.min_contact_iou = float(min_contact_iou)
        self.contact_edge_px = float(contact_edge_px_abs)
        self.velocity_drop_threshold = float(min_rel_delta_v or velocity_drop_threshold)
        if speed_drop_ratio is not None:
            self.velocity_drop_threshold = min(self.velocity_drop_threshold, max(0.08, 1.0 - float(speed_drop_ratio)))
        self.min_relative_speed = float(min_relative_speed)
        self.optical_flow_threshold = float(flow_spike_threshold or optical_flow_threshold)
        self.trajectory_prediction_horizon = max(2, int(trajectory_prediction_horizon))
        self.prediction_error_threshold = float(prediction_error_threshold)
        self.post_impact_frames = max(2, int(math.ceil(post_impact_seconds * self.fps)))
        self.pre_window_frames = max(3, int(math.ceil(pre_window_seconds * self.fps)))
        self.n_pre = self.pre_window_frames  # backward-compatible tests/helpers
        self.moving_threshold = float(moving_threshold)
        self.min_peak_speed_for_collision = float(min_peak_speed_for_collision)
        self.stationary_speed_px_s = float(stationary_speed_px_s)
        self.stationary_window_frames = max(3, int(stationary_window_frames))
        self.parked_min_frames = max(1, int(math.ceil(float(parked_min_seconds) * self.fps)))
        self.require_post_impact_validation = bool(require_post_impact_validation)
        self.require_hard_impact_signal = bool(require_hard_impact_signal)
        self.min_hard_impact_signals = max(1, int(min_hard_impact_signals))
        self.min_supporting_impact_signals = max(0, int(min_supporting_impact_signals))
        self.suppress_same_direction_traffic = bool(suppress_same_direction_traffic)
        self.suppress_static_static = bool(suppress_static_static)
        self.suppress_normal_passing_parked = bool(suppress_normal_passing_parked)
        self.closing_distance_enabled = bool(closing_distance_enabled)
        self.closing_distance_min_drop_px = float(closing_distance_min_drop_px)
        self.closing_distance_min_drop_ratio = float(closing_distance_min_drop_ratio)
        self.max_candidate_gap_frames = max(1, int(max_candidate_gap_frames))
        self.state_grace_frames = max(
            self.max_candidate_gap_frames,
            int(state_grace_frames) if state_grace_frames is not None else int(max(12, round(self.fps * 0.8))),
        )
        self.pair_grid_cell_px = float(pair_grid_cell_px or max(self.proximity_threshold_px, 96.0))
        self.max_pair_distance_px = float(max_pair_distance_px or 0.0)
        self.max_pairs_per_frame = max(1, int(max_pairs_per_frame))
        self.debug_record_normal_interval = max(1, int(debug_record_normal_interval))
        self.min_bbox_height_ratio = max(0.0, float(min_bbox_height_ratio))
        self.normalized_speed_enabled = bool(normalized_speed_enabled)
        self.road_roi_enabled = bool(road_roi_enabled)
        self.pair_cooldown_frames = max(
            1,
            int(pair_cooldown_frames) if pair_cooldown_frames is not None else int(self.cooldown_seconds * self.fps),
        )
        self.muted_track_frames = max(1, int(muted_track_frames))
        self.muted_base_radius_px = float(muted_base_radius_px)
        self.muted_velocity_radius_scale = float(muted_velocity_radius_scale)
        self.muted_aspect_ratio_delta = float(muted_aspect_ratio_delta)
        self.muted_area_ratio_delta = float(muted_area_ratio_delta)
        self.async_queue_size = max(1, int(async_queue_size))
        self.confirmed_display_frames = max(1, int(math.ceil(float(confirmed_display_seconds) * self.fps)))
        self.same_direction_cosine = float(same_direction_cosine)
        self.min_track_age = int(min_track_history_frames if min_track_history_frames is not None else min_track_age_frames)
        self.fp_frames = int(fp_suppression_frames)
        self.grazing_iou_limit = float(grazing_iou_limit)
        self.approach_angle_deg = float(approach_angle_deg)
        self.high_precision_mode = bool(high_precision_mode)

        self.weights = severity_weights or {
            "iou_score": 0.14,
            "proximity_score": 0.10,
            "closing_distance_score": 0.12,
            "velocity_drop_score": 0.20,
            "trajectory_conflict_score": 0.17,
            "optical_flow_anomaly_score": 0.10,
            "post_impact_stagnation_score": 0.17,
            "direction_change_score": 0.12,
        }

        self.debug_events = bool(debug_events)
        self.debug_events_path = Path(debug_events_path)
        if debug_events_dir and debug_events_path == "output/debug_events.jsonl":
            self.debug_events_path = Path(debug_events_dir) / "debug_events.jsonl"

        self._pairs: Dict[Tuple[int, int], _PairState] = {}
        self._speed_history: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=240))
        self._centroid_history: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=240))
        self._bbox_history: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=240))
        self._heading_history: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=120))
        self._peak_speed: Dict[int, float] = defaultdict(float)
        self._velocity_per_frame: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=float))
        self._last_track_snapshot: Dict[int, _TrackSnapshot] = {}
        self._muted_tracks: Dict[int, _TrackSnapshot] = {}
        self._track_alias: Dict[int, int] = {}
        self._prev_gray: Optional[np.ndarray] = None
        self._flow_level_history: Deque[float] = deque(maxlen=120)
        self._event_counter = 0
        self._lock = threading.RLock()
        self._async_in: "queue.Queue[_AsyncFusionTask | None]" = queue.Queue(maxsize=self.async_queue_size)
        self._async_out: "queue.Queue[FusionFrameResult]" = queue.Queue(maxsize=self.async_queue_size)
        self._async_thread: Optional[threading.Thread] = None
        self._async_stop = threading.Event()

    def process_frame(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        frame_id: int,
        debug: bool = False,
    ) -> List[AccidentEvent]:
        """Evaluate spatially plausible vehicle pairs in one processed frame."""
        with self._lock:
            fusion_tracks = self._canonicalize_tracks(tracked_objects, frame_id)
            self._update_track_histories(fusion_tracks, frame_id)

            vehicle_classes = {1, 2, 3, 5, 7}
            frame_shape = frame.shape[:2]
            vehicles = [
                obj
                for obj in fusion_tracks
                if int(obj.class_id) in vehicle_classes and self._valid_dynamic_vehicle(obj, frame_shape)
            ]
            pair_candidates = self._preselect_pairs(vehicles, frame_id)
            flow_map = self._compute_flow_map(frame) if pair_candidates else self._prime_gray(frame)

            confirmed: List[AccidentEvent] = []
            seen_pairs: set[Tuple[int, int]] = set()

            for obj_a, obj_b in pair_candidates:
                pair_key = (min(obj_a.track_id, obj_b.track_id), max(obj_a.track_id, obj_b.track_id))
                seen_pairs.add(pair_key)
                event = self._evaluate_pair(obj_a, obj_b, frame_id, flow_map, debug)
                if event is not None:
                    confirmed.append(event)

            self._expire_unseen_pairs(seen_pairs, frame_id)
            return confirmed

    def reset_state(self, reset_frame_memory: bool = True) -> None:
        """Clear pair/event state after a source transition or scene cut."""
        with self._lock:
            self._pairs.clear()
            self._muted_tracks.clear()
            self._track_alias.clear()
            self._event_counter = 0
            if reset_frame_memory:
                self._speed_history.clear()
                self._centroid_history.clear()
                self._bbox_history.clear()
                self._heading_history.clear()
                self._peak_speed.clear()
                self._velocity_per_frame.clear()
                self._last_track_snapshot.clear()
                self._prev_gray = None
                self._flow_level_history.clear()

    def start_async(self) -> None:
        """Start a bounded async fusion worker for edge pipelines."""
        with self._lock:
            if self._async_thread and self._async_thread.is_alive():
                return
            self._async_stop.clear()
            self._async_thread = threading.Thread(target=self._async_loop, name="accident-fusion", daemon=True)
            self._async_thread.start()

    def submit_frame(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        frame_id: int,
        debug: bool = False,
        drop_oldest: bool = True,
    ) -> bool:
        """Queue a frame for fusion without blocking the caller.

        When the queue is full, the oldest pending fusion task is discarded.
        This keeps live video ingestion moving under CPU pressure.
        """
        if self._async_thread is None or not self._async_thread.is_alive():
            self.start_async()
        task = _AsyncFusionTask(
            frame_id=frame_id,
            frame=frame.copy(),
            tracked_objects=self._snapshot_tracks(tracked_objects),
            debug=debug,
        )
        if not drop_oldest:
            while not self._async_stop.is_set():
                try:
                    self._async_in.put(task, timeout=0.25)
                    return True
                except queue.Full:
                    continue
            return False

        try:
            self._async_in.put_nowait(task)
            return True
        except queue.Full:
            try:
                self._async_in.get_nowait()
            except queue.Empty:
                pass
            try:
                self._async_in.put_nowait(task)
                logger.warning("Accident fusion queue full; dropped oldest fusion task", frame_id=frame_id)
                return True
            except queue.Full:
                logger.warning("Accident fusion queue full; dropped current fusion task", frame_id=frame_id)
                return False

    def drain_results(self) -> List[FusionFrameResult]:
        results: List[FusionFrameResult] = []
        while True:
            try:
                results.append(self._async_out.get_nowait())
            except queue.Empty:
                return results

    def drain_events(self) -> List[AccidentEvent]:
        events: List[AccidentEvent] = []
        for result in self.drain_results():
            events.extend(result.events)
        return events

    def stop_async(self, timeout: float = 2.0) -> None:
        if self._async_thread and self._async_thread.is_alive():
            try:
                self._async_in.put_nowait(None)
            except queue.Full:
                try:
                    self._async_in.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._async_in.put_nowait(None)
                except queue.Full:
                    self._async_stop.set()
            self._async_thread.join(timeout=timeout)
        self._async_stop.set()

    def _async_loop(self) -> None:
        while not self._async_stop.is_set():
            try:
                task = self._async_in.get(timeout=0.25)
            except queue.Empty:
                continue
            if task is None:
                break
            try:
                events = self.process_frame(task.frame, task.tracked_objects, task.frame_id, task.debug)
                result = FusionFrameResult(task.frame_id, task.frame, task.tracked_objects, events)
                try:
                    self._async_out.put_nowait(result)
                except queue.Full:
                    try:
                        self._async_out.get_nowait()
                    except queue.Empty:
                        pass
                    self._async_out.put_nowait(result)
            except Exception as exc:  # pragma: no cover - defensive edge runtime guard
                logger.exception("Accident fusion async worker failed", exc=str(exc), frame_id=task.frame_id)

    def _valid_dynamic_vehicle(self, obj: TrackedObject, frame_shape: Tuple[int, int]) -> bool:
        if self.min_bbox_height_ratio <= 0:
            return True
        frame_h = max(float(frame_shape[0]), 1.0)
        bbox = np.array(obj.bbox, dtype=float)
        height_ratio = max(0.0, float(bbox[3] - bbox[1])) / frame_h
        if height_ratio < self.min_bbox_height_ratio:
            return False
        if self.road_roi_enabled:
            # ROI polygons are not part of the current config schema.  The flag is
            # kept for compatibility and future camera-specific road masks.
            return True
        return True

    def get_active_candidates(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Return candidate/suspect pair state for overlays and debugging."""
        with self._lock:
            active: Dict[Tuple[int, int], Dict[str, Any]] = {}
            for key, state in self._pairs.items():
                if state.state in {EventState.SUSPECT, EventState.CANDIDATE, EventState.CONFIRMED}:
                    last = state.evidence[-1].to_json() if state.evidence else {}
                    active[key] = {
                        "state": state.state.value,
                        "event_id": state.event_id,
                        "confirming_frames": state.confirming_frames,
                        "candidate_start_frame": state.candidate_frame,
                        "max_severity_seen": state.max_severity_seen,
                        "post_impact_seen": state.post_impact_seen,
                        "confirmed_until_frame": state.confirmed_until_frame,
                        "last": last,
                    }
            return active

    def explain_event(self, event: AccidentEvent) -> Dict[str, Any]:
        doc = {
            "event_id": event.event_id,
            "track_id_a": event.track_id_a,
            "track_id_b": event.track_id_b,
            "timestamp": event.timestamp,
            "frame_number": event.frame_id,
            "severity_score": event.severity_score,
            "iou_at_collision": event.iou_at_collision,
            "motion_anomaly_score": event.motion_anomaly_score,
            "signals_triggered": event.signals_triggered,
            "signal_scores": event.signal_scores,
            "debug": event.debug or {},
        }
        return doc

    def _snapshot_tracks(self, tracked_objects: Iterable[TrackedObject]) -> List[TrackedObject]:
        snapshots: List[TrackedObject] = []
        for obj in tracked_objects:
            snapshots.append(
                replace(
                    obj,
                    bbox=np.array(obj.bbox, dtype=float).copy(),
                    centroid=np.array(obj.centroid, dtype=float).copy(),
                    velocity_vector=np.array(obj.velocity_vector, dtype=float).copy(),
                    trajectory=deque(list(obj.trajectory), maxlen=getattr(obj.trajectory, "maxlen", 90) or 90),
                    confidence_history=deque(
                        list(obj.confidence_history),
                        maxlen=getattr(obj.confidence_history, "maxlen", 30) or 30,
                    ),
                )
            )
        return snapshots

    def _canonicalize_tracks(self, tracked_objects: Iterable[TrackedObject], frame_id: int) -> List[TrackedObject]:
        active_raw_ids = {int(obj.track_id) for obj in tracked_objects}
        for old_id, snapshot in list(self._last_track_snapshot.items()):
            if old_id in active_raw_ids:
                continue
            age = frame_id - snapshot.frame_id
            if age <= 0 or age > self.muted_track_frames:
                continue
            if self._track_in_active_pair(old_id) or self._peak_speed[old_id] >= self.min_peak_speed_for_collision:
                self._muted_tracks[old_id] = snapshot

        canonicalized: List[TrackedObject] = []
        used_muted: set[int] = set()
        for obj in tracked_objects:
            raw_id = int(obj.track_id)
            canonical_id = self._track_alias.get(raw_id, raw_id)
            if canonical_id == raw_id and raw_id not in self._last_track_snapshot:
                match_id = self._match_muted_track(obj, frame_id, used_muted)
                if match_id is not None:
                    canonical_id = match_id
                    self._track_alias[raw_id] = match_id
                    used_muted.add(match_id)
                    self._muted_tracks.pop(match_id, None)

            if canonical_id != raw_id:
                canonicalized.append(replace(obj, track_id=canonical_id))
            else:
                canonicalized.append(obj)

        self._expire_muted_tracks(frame_id)
        return canonicalized

    def _track_in_active_pair(self, track_id: int) -> bool:
        for pair_key, state in self._pairs.items():
            if track_id in pair_key and state.state in {EventState.SUSPECT, EventState.CANDIDATE, EventState.CONFIRMED}:
                return True
        return False

    def _match_muted_track(
        self,
        obj: TrackedObject,
        frame_id: int,
        used_muted: set[int],
    ) -> Optional[int]:
        center = np.array(obj.centroid, dtype=float)
        bbox = np.array(obj.bbox, dtype=float)
        width = max(1.0, float(bbox[2] - bbox[0]))
        height = max(1.0, float(bbox[3] - bbox[1]))
        area = width * height
        aspect = width / max(height, 1.0)

        best_id: Optional[int] = None
        best_dist = float("inf")
        for muted_id, snapshot in self._muted_tracks.items():
            if muted_id in used_muted:
                continue
            if int(obj.class_id) != int(snapshot.class_id):
                continue
            age = frame_id - snapshot.frame_id
            if age <= 0 or age > self.muted_track_frames:
                continue
            predicted = snapshot.center + snapshot.velocity_per_frame * age
            speed = float(np.linalg.norm(snapshot.velocity_per_frame))
            radius = self.muted_base_radius_px + self.muted_velocity_radius_scale * speed * age + snapshot.diagonal * 0.25
            dist = float(np.linalg.norm(center - predicted))
            if dist > radius:
                continue
            aspect_delta = abs(aspect - snapshot.aspect_ratio) / max(snapshot.aspect_ratio, 1e-6)
            area_delta = abs(area - snapshot.area) / max(snapshot.area, 1e-6)
            if aspect_delta > self.muted_aspect_ratio_delta or area_delta > self.muted_area_ratio_delta:
                continue
            if dist < best_dist:
                best_dist = dist
                best_id = muted_id
        return best_id

    def _expire_muted_tracks(self, frame_id: int) -> None:
        expired = [
            tid for tid, snapshot in self._muted_tracks.items()
            if frame_id - snapshot.frame_id > self.muted_track_frames
        ]
        for tid in expired:
            self._muted_tracks.pop(tid, None)
        self._track_alias = {
            raw: canonical
            for raw, canonical in self._track_alias.items()
            if canonical in self._last_track_snapshot
            and frame_id - self._last_track_snapshot[canonical].frame_id <= self.muted_track_frames * 2
        }

    def _update_track_histories(self, tracked_objects: Iterable[TrackedObject], frame_id: int) -> None:
        for obj in tracked_objects:
            tid = int(obj.track_id)
            cx, cy = float(obj.centroid[0]), float(obj.centroid[1])
            self._centroid_history[tid].append((cx, cy))
            self._bbox_history[tid].append(np.array(obj.bbox, dtype=float).copy())

            velocity_per_frame = self._velocity_from_history(tid, frames=min(5, self.pre_window_frames))
            self._velocity_per_frame[tid] = velocity_per_frame
            hist_velocity = velocity_per_frame * self.fps
            hist_speed = float(np.linalg.norm(hist_velocity))
            speed = max(float(obj.speed_px_s or 0.0), hist_speed)
            self._speed_history[tid].append(speed)
            self._peak_speed[tid] = max(self._peak_speed[tid], speed)
            if hist_speed > 0.5:
                self._heading_history[tid].append(hist_velocity / max(hist_speed, 1e-6))
            self._last_track_snapshot[tid] = _TrackSnapshot(
                track_id=tid,
                class_id=int(obj.class_id),
                center=np.array([cx, cy], dtype=float),
                bbox=np.array(obj.bbox, dtype=float).copy(),
                velocity_per_frame=velocity_per_frame.copy(),
                frame_id=frame_id,
            )

    def _preselect_pairs(self, vehicles: List[TrackedObject], frame_id: int) -> List[Tuple[TrackedObject, TrackedObject]]:
        if len(vehicles) < 2:
            return []

        by_id = {int(obj.track_id): obj for obj in vehicles}
        selected: Dict[Tuple[int, int], Tuple[float, TrackedObject, TrackedObject]] = {}

        for pair_key, state in self._pairs.items():
            if state.state not in {EventState.SUSPECT, EventState.CANDIDATE, EventState.CONFIRMED}:
                continue
            if frame_id - state.last_seen_frame > self.state_grace_frames:
                continue
            a = by_id.get(pair_key[0])
            b = by_id.get(pair_key[1])
            if a is not None and b is not None:
                selected[pair_key] = (0.0, a, b)

        cell_size = max(self.pair_grid_cell_px, 1.0)
        cells: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        centers: List[np.ndarray] = []
        bboxes: List[np.ndarray] = []
        diags: List[float] = []
        static_flags: List[bool] = []

        for idx, obj in enumerate(vehicles):
            center = np.array(obj.centroid, dtype=float)
            bbox = np.array(obj.bbox, dtype=float)
            centers.append(center)
            bboxes.append(bbox)
            diags.append(self._bbox_diagonal(bbox))
            static_flags.append(self.is_stationary(obj.track_id) or self.is_parked(obj.track_id))
            cell = (int(center[0] // cell_size), int(center[1] // cell_size))
            cells[cell].append(idx)

        for idx, obj_a in enumerate(vehicles):
            center_a = centers[idx]
            max_diag = diags[idx]
            speed_a = float(self._speed_history[obj_a.track_id][-1]) if self._speed_history[obj_a.track_id] else 0.0
            search_radius = self._pair_search_radius(max_diag, speed_a, 0.0)
            cell = (int(center_a[0] // cell_size), int(center_a[1] // cell_size))
            cell_span = max(1, int(math.ceil(search_radius / cell_size)))

            for dx in range(-cell_span, cell_span + 1):
                for dy in range(-cell_span, cell_span + 1):
                    for jdx in cells.get((cell[0] + dx, cell[1] + dy), []):
                        if jdx <= idx:
                            continue
                        obj_b = vehicles[jdx]
                        pair_key = (min(obj_a.track_id, obj_b.track_id), max(obj_a.track_id, obj_b.track_id))
                        if pair_key in selected:
                            continue
                        state = self._pairs.get(pair_key)
                        active_state = state is not None and state.state in {
                            EventState.SUSPECT,
                            EventState.CANDIDATE,
                            EventState.CONFIRMED,
                        }
                        if static_flags[idx] and static_flags[jdx] and not active_state:
                            continue

                        speed_b = float(self._speed_history[obj_b.track_id][-1]) if self._speed_history[obj_b.track_id] else 0.0
                        avg_diag = max(1.0, (diags[idx] + diags[jdx]) / 2.0)
                        radius = self._pair_search_radius(avg_diag, speed_a, speed_b)
                        distance = float(np.linalg.norm(center_a - centers[jdx]))
                        if distance > radius and not active_state:
                            edge_gap = self._bbox_edge_gap(bboxes[idx], bboxes[jdx])
                            if edge_gap > max(self.contact_edge_px * 3.0, avg_diag * 0.15):
                                continue
                        if self._early_same_direction_stable(obj_a, obj_b) and not active_state:
                            continue
                        selected[pair_key] = (distance, obj_a, obj_b)

        ranked = sorted(selected.values(), key=lambda item: item[0])
        if len(ranked) > self.max_pairs_per_frame:
            ranked = ranked[: self.max_pairs_per_frame]
        return [(a, b) for _, a, b in ranked]

    def _pair_search_radius(self, avg_diag: float, speed_a: float, speed_b: float) -> float:
        if self.max_pair_distance_px > 0:
            return self.max_pair_distance_px
        motion_allowance = (max(speed_a, 0.0) + max(speed_b, 0.0)) / max(self.fps, 1.0) * 3.0
        return max(self.proximity_threshold_px, avg_diag * 1.6, self.contact_edge_px * 6.0) + motion_allowance

    def _early_same_direction_stable(self, a: TrackedObject, b: TrackedObject) -> bool:
        va = self._velocity_per_frame.get(a.track_id, np.zeros(2, dtype=float))
        vb = self._velocity_per_frame.get(b.track_id, np.zeros(2, dtype=float))
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na < 0.2 or nb < 0.2:
            return False
        cos = float(np.dot(va, vb) / max(na * nb, 1e-6))
        if cos < self.same_direction_cosine:
            return False
        speed_a = self._speed_history[a.track_id]
        speed_b = self._speed_history[b.track_id]
        if len(speed_a) < 4 or len(speed_b) < 4:
            return False
        recent_a = list(speed_a)[-4:]
        recent_b = list(speed_b)[-4:]
        stable_a = float(np.std(recent_a)) <= max(2.0, float(np.mean(recent_a)) * 0.12)
        stable_b = float(np.std(recent_b)) <= max(2.0, float(np.mean(recent_b)) * 0.12)
        return stable_a and stable_b

    def _evaluate_pair(
        self,
        a: TrackedObject,
        b: TrackedObject,
        frame_id: int,
        flow_map: Optional[np.ndarray],
        debug: bool,
    ) -> Optional[AccidentEvent]:
        pair_key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
        state = self._pairs.get(pair_key)
        if state is None:
            state = _PairState(
                track_id_a=pair_key[0],
                track_id_b=pair_key[1],
                first_seen_frame=frame_id,
                cooldown_seconds=self.cooldown_seconds,
            )
            self._pairs[pair_key] = state
        state.last_seen_frame = frame_id

        evidence = self._score_pair(a, b, pair_key, frame_id, flow_map)
        state.evidence.append(evidence)
        state.iou_history.append(evidence.iou)
        state.distance_history.append(evidence.distance_px)

        if len(self._centroid_history[a.track_id]) < self.min_track_age or len(self._centroid_history[b.track_id]) < self.min_track_age:
            return None

        suppressed = self.should_suppress_candidate(a, b, state, evidence)
        if suppressed:
            evidence.suppressed_reason = suppressed
            state.weak_frames += 1
            if state.state == EventState.CANDIDATE:
                self._record_confirmation_sample(state, evidence, positive=False)
            if state.state in {EventState.SUSPECT, EventState.CANDIDATE} and state.weak_frames > self.state_grace_frames:
                state.state = EventState.CLOSED if state.state == EventState.CANDIDATE else EventState.NORMAL
            self._write_debug_record("suppressed_candidate", state, evidence, debug, wait_reason=suppressed)
            return None

        preliminary = self._is_preliminary_candidate(evidence)
        strong_candidate = self._is_strong_candidate(evidence)

        if state.state == EventState.NORMAL:
            if preliminary:
                state.state = EventState.SUSPECT
                state.suspect_frame = frame_id
                state.weak_frames = 0
                state.missing_frames = 0
        elif state.state == EventState.SUSPECT:
            if self._candidate_entry_ok(evidence):
                state.state = EventState.CANDIDATE
                if state.candidate_frame < 0:
                    state.candidate_frame = frame_id
                    state.candidate_start_time = time.time()
                    state.impact_frame = frame_id
                    state.confirmation_window.clear()
                    state.hard_impact_window.clear()
                state.weak_frames = 0
                state.missing_frames = 0
                state.max_severity_seen = evidence.severity
                state.post_impact_seen = False
                state.post_impact_reasons = []
                state.last_wait_reason = ""
                if not state.event_id:
                    state.event_id = self._next_event_id(frame_id, pair_key)
            elif preliminary:
                state.weak_frames = 0
                state.missing_frames = 0
            else:
                state.weak_frames += 1
                if state.weak_frames > self.state_grace_frames:
                    state.state = EventState.NORMAL
        elif state.state == EventState.CANDIDATE:
            if self._candidate_frame_ok(evidence):
                state.weak_frames = 0
                state.missing_frames = 0
            else:
                state.weak_frames += 1
                self._record_confirmation_sample(state, evidence, positive=False)
                if state.weak_frames > self.state_grace_frames:
                    state.state = EventState.CLOSED
                    state.closed_frame = frame_id
                    self._write_debug_record("candidate_closed", state, evidence, debug, wait_reason="candidate_evidence_disappeared")
                    return None
        elif state.state == EventState.CONFIRMED:
            if frame_id > state.confirmed_until_frame:
                state.state = EventState.CLOSED
                state.closed_frame = frame_id
            return None
        elif state.state == EventState.CLOSED:
            return None

        if state.state != EventState.CANDIDATE:
            self._write_debug_record("state", state, evidence, debug)
            return None

        state.max_severity_seen = max(state.max_severity_seen, evidence.severity)
        if self._candidate_frame_ok(evidence):
            self._record_confirmation_sample(state, evidence, positive=True)
        else:
            self._record_confirmation_sample(state, evidence, positive=False)

        post_ok, post_reasons = self._post_impact_validated(state)
        if post_ok:
            state.post_impact_seen = True
            for reason in post_reasons:
                if reason not in state.post_impact_reasons:
                    state.post_impact_reasons.append(reason)

        window_ready = len(state.confirmation_window) >= min(self.confirm_frames, self.min_confirming_frames)
        hits_ok = state.confirming_frames >= self.min_confirming_frames
        if not (window_ready and hits_ok):
            self._write_debug_record(
                "candidate_wait",
                state,
                evidence,
                debug,
                wait_reason="confirmation_window_not_satisfied",
            )
            return None

        severity_ok = state.max_severity_seen >= self.confirmed_threshold or evidence.severity >= self.confirmed_threshold
        if not severity_ok:
            self._write_debug_record(
                "candidate_wait",
                state,
                evidence,
                debug,
                wait_reason="severity_below_confirmed_threshold",
            )
            return None

        if self.require_post_impact_validation and not state.post_impact_seen:
            self._write_debug_record(
                "candidate_wait",
                state,
                evidence,
                debug,
                wait_reason="missing_hard_impact_signal",
            )
            return None

        cooldown_ok = (
            state.last_confirmed_time <= 0.0
            or (
                time.time() - state.last_confirmed_time >= state.cooldown_seconds
                and (state.confirmed_frame < 0 or frame_id - state.confirmed_frame >= self.pair_cooldown_frames)
            )
        )
        if not cooldown_ok:
            self._write_debug_record("candidate_wait", state, evidence, debug, wait_reason="cooldown_duplicate_alert_suppressed")
            return None

        state.state = EventState.CONFIRMED
        state.confirmed_frame = frame_id
        state.confirmed_until_frame = frame_id + self.confirmed_display_frames
        state.last_confirmed_time = time.time()
        state.last_wait_reason = ""
        evidence.reasons.extend(state.post_impact_reasons)

        severity = max(state.max_severity_seen, evidence.severity)
        event = self._make_event(state, evidence, severity, state.post_impact_reasons)
        self._write_debug_record("confirmed", state, evidence, True)
        state.emitted = True
        return event

    def _record_confirmation_sample(self, state: _PairState, evidence: SignalEvidence, positive: bool) -> None:
        if state.confirmation_window.maxlen != self.confirm_frames:
            state.confirmation_window = deque(state.confirmation_window, maxlen=self.confirm_frames)
        if state.hard_impact_window.maxlen != self.confirm_frames:
            state.hard_impact_window = deque(state.hard_impact_window, maxlen=self.confirm_frames)
        hit = 1 if positive else 0
        hard_hit = 1 if positive and self._hard_impact_frame(evidence) else 0
        state.confirmation_window.append(hit)
        state.hard_impact_window.append(hard_hit)
        state.confirming_frames = int(sum(state.confirmation_window))
        state.hard_impact_frames = int(sum(state.hard_impact_window))
        if positive:
            state.last_positive_frame = evidence.frame_id

    def _hard_impact_frame(self, evidence: SignalEvidence) -> bool:
        flags = evidence.signal_flags
        return bool(
            flags["velocity_drop"]
            or flags["post_impact_stagnation"]
            or flags["direction_change"]
            or (flags["closing_distance_before_contact"] and flags["contact_or_near_overlap"])
        )

    def _score_pair(
        self,
        a: TrackedObject,
        b: TrackedObject,
        pair_key: Tuple[int, int],
        frame_id: int,
        flow_map: Optional[np.ndarray],
    ) -> SignalEvidence:
        bbox_a = np.array(a.bbox, dtype=float)
        bbox_b = np.array(b.bbox, dtype=float)
        diag_a = self._bbox_diagonal(bbox_a)
        diag_b = self._bbox_diagonal(bbox_b)
        avg_diag = max(1.0, (diag_a + diag_b) / 2.0)

        iou = self._box_iou(bbox_a, bbox_b)
        edge_gap = self._bbox_edge_gap(bbox_a, bbox_b)
        distance = float(np.linalg.norm(np.array(a.centroid, dtype=float) - np.array(b.centroid, dtype=float)))
        dynamic_proximity = max(self.proximity_threshold_px, avg_diag * 1.4)

        iou_score = _clamp01(iou / max(self.iou_spike_threshold, self.min_contact_iou, 1e-6))
        proximity_score = _clamp01(1.0 - (distance / max(dynamic_proximity, 1.0)))
        if edge_gap < 0:
            near_overlap_score = 1.0
        else:
            near_overlap_score = _clamp01(1.0 - edge_gap / max(self.contact_edge_px, avg_diag * 0.08, 1.0))

        velocity_a = self._velocity_from_history(a.track_id, frames=self.pre_window_frames) * self.fps
        velocity_b = self._velocity_from_history(b.track_id, frames=self.pre_window_frames) * self.fps
        relative_speed_norm = float(np.linalg.norm(velocity_a - velocity_b)) / avg_diag
        relative_speed_score = _clamp01(relative_speed_norm / max(self.min_relative_speed * 2.5, 1e-6))

        velocity_drop_score = max(
            self._speed_drop_score(a.track_id, avg_diag),
            self._speed_drop_score(b.track_id, avg_diag),
        )
        trajectory_score, trajectory_reasons = self._trajectory_conflict_score(a.track_id, b.track_id, avg_diag)
        flow_score = self._optical_flow_score(flow_map, bbox_a, bbox_b)
        stagnation_score = max(self._stagnation_score(a.track_id), self._stagnation_score(b.track_id))
        direction_change_score = max(
            self._direction_change_score(a.track_id),
            self._direction_change_score(b.track_id),
        )
        closing_distance_score = self._closing_distance_score(
            pair_key=pair_key,
            current_distance=distance,
            edge_gap=edge_gap,
            near_overlap_score=near_overlap_score,
        )

        signal_scores = {
            "iou_score": round(iou_score, 4),
            "proximity_score": round(proximity_score, 4),
            "near_overlap_score": round(near_overlap_score, 4),
            "closing_distance_score": round(closing_distance_score, 4),
            "relative_speed_score": round(relative_speed_score, 4),
            "velocity_drop_score": round(velocity_drop_score, 4),
            "trajectory_conflict_score": round(trajectory_score, 4),
            "optical_flow_anomaly_score": round(flow_score, 4),
            "post_impact_stagnation_score": round(stagnation_score, 4),
            "direction_change_score": round(direction_change_score, 4),
        }

        signal_flags = {
            "proximity": proximity_score >= 0.18,
            "contact_or_near_overlap": iou >= self.min_contact_iou or near_overlap_score >= 0.55,
            "relative_motion": relative_speed_norm >= self.min_relative_speed,
            "closing_distance_before_contact": closing_distance_score >= 0.55,
            "approach_to_contact": closing_distance_score >= 0.70 and (iou >= self.min_contact_iou or near_overlap_score >= 0.65),
            "velocity_drop": velocity_drop_score >= self.velocity_drop_threshold,
            "trajectory_conflict": trajectory_score >= 0.45,
            "optical_flow_anomaly": flow_score >= self.optical_flow_threshold,
            "post_impact_stagnation": stagnation_score >= 0.55,
            "direction_change": direction_change_score >= 0.45,
        }

        severity = self._severity(signal_scores)
        reasons = trajectory_reasons
        if iou >= self.min_contact_iou:
            reasons.append("bbox_iou_contact")
        elif near_overlap_score >= 0.55:
            reasons.append("near_bbox_overlap")
        if velocity_drop_score >= self.velocity_drop_threshold:
            reasons.append("abrupt_deceleration")
        if closing_distance_score >= 0.55:
            reasons.append("closing_distance_before_contact")
        if stagnation_score >= 0.55:
            reasons.append("post_impact_stagnation")

        return SignalEvidence(
            frame_id=frame_id,
            pair_key=pair_key,
            state=self._pairs[pair_key].state if pair_key in self._pairs else EventState.NORMAL,
            signal_scores=signal_scores,
            signal_flags=signal_flags,
            severity=severity,
            iou=iou,
            distance_px=distance,
            edge_gap_px=edge_gap,
            speed_a=float(self._speed_history[a.track_id][-1]) if self._speed_history[a.track_id] else float(a.speed_px_s),
            speed_b=float(self._speed_history[b.track_id][-1]) if self._speed_history[b.track_id] else float(b.speed_px_s),
            bbox_a=bbox_a.tolist(),
            bbox_b=bbox_b.tolist(),
            reasons=reasons,
        )

    def _severity(self, scores: Dict[str, float]) -> float:
        total_weight = sum(self.weights.values()) or 1.0
        total = sum(self.weights.get(name, 0.0) * scores.get(name, 0.0) for name in self.weights)
        return _clamp01(total / total_weight)

    def _closing_distance_score(
        self,
        pair_key: Tuple[int, int],
        current_distance: float,
        edge_gap: float,
        near_overlap_score: float,
    ) -> float:
        if not self.closing_distance_enabled:
            return 0.0
        history = list(self._pairs[pair_key].distance_history) if pair_key in self._pairs else []
        distances = history[-max(4, self.pre_window_frames):] + [float(current_distance)]
        if len(distances) < 4:
            return 0.0
        half = max(2, len(distances) // 2)
        first = float(np.median(distances[:half]))
        second = float(np.median(distances[half:]))
        drop_px = max(0.0, first - second)
        drop_ratio = drop_px / max(first, 1.0)
        small_gap = edge_gap < 0 or edge_gap <= max(self.contact_edge_px * 2.0, 10.0) or near_overlap_score >= 0.65
        if not small_gap:
            return 0.0
        px_score = _clamp01(drop_px / max(self.closing_distance_min_drop_px, 1.0))
        ratio_score = _clamp01(drop_ratio / max(self.closing_distance_min_drop_ratio, 1e-6))
        consistency = 0.0
        pairs = list(zip(distances, distances[1:]))
        if pairs:
            consistency = sum(1 for prev, cur in pairs if cur <= prev + 1.0) / len(pairs)
        return _clamp01(0.45 * px_score + 0.40 * ratio_score + 0.15 * consistency)

    def _is_preliminary_candidate(self, ev: SignalEvidence) -> bool:
        flags = ev.signal_flags
        return bool(
            flags["proximity"]
            and ev.active_signal_count >= 2
        )

    def _is_strong_candidate(self, ev: SignalEvidence) -> bool:
        flags = ev.signal_flags
        contact_evidence = flags["proximity"] or flags["contact_or_near_overlap"]
        geometry_evidence = (
            flags["trajectory_conflict"]
            or flags["closing_distance_before_contact"]
            or flags["approach_to_contact"]
        )
        impact_precursor = (
            flags["velocity_drop"]
            or flags["post_impact_stagnation"]
            or flags["direction_change"]
            or flags["approach_to_contact"]
            or (flags["optical_flow_anomaly"] and ev.signal_scores["optical_flow_anomaly_score"] >= self.optical_flow_threshold * 1.5)
        )
        if not contact_evidence:
            return False
        if not geometry_evidence:
            return False
        if not impact_precursor:
            return False
        return ev.active_signal_count >= self.min_signals_required

    def _candidate_entry_ok(self, ev: SignalEvidence) -> bool:
        return (
            ev.active_signal_count >= self.min_signals_required
            and ev.severity >= self.candidate_threshold
            and self._is_strong_candidate(ev)
        )

    def _candidate_frame_ok(self, ev: SignalEvidence) -> bool:
        hard_motion = (
            ev.signal_flags["velocity_drop"]
            or ev.signal_flags["post_impact_stagnation"]
            or ev.signal_flags["direction_change"]
            or (ev.signal_flags["closing_distance_before_contact"] and ev.signal_flags["contact_or_near_overlap"])
        )
        geometry_or_impact = (
            hard_motion
            or ev.signal_flags["trajectory_conflict"]
            or ev.signal_flags["closing_distance_before_contact"]
            or ev.signal_flags["approach_to_contact"]
        )
        return (
            ev.severity >= self.candidate_threshold * 0.85
            and ev.active_signal_count >= max(2, self.min_signals_required - 1)
            and ev.signal_flags["proximity"]
            and ev.signal_flags["contact_or_near_overlap"]
            and geometry_or_impact
        )

    def is_stationary(self, track_id: int) -> bool:
        hist = list(self._speed_history[track_id])[-self.stationary_window_frames:]
        if len(hist) < max(3, self.stationary_window_frames // 2):
            return False
        return float(np.mean(hist)) <= self.stationary_speed_px_s

    def is_parked(self, track_id: int) -> bool:
        speed_hist = list(self._speed_history[track_id])[-self.parked_min_frames:]
        centroid_hist = list(self._centroid_history[track_id])[-self.parked_min_frames:]
        if len(speed_hist) < self.parked_min_frames or len(centroid_hist) < self.parked_min_frames:
            return False
        if float(np.mean(speed_hist)) > self.stationary_speed_px_s:
            return False
        displacement = float(np.linalg.norm(np.array(centroid_hist[-1]) - np.array(centroid_hist[0])))
        return displacement <= max(3.0, self.stationary_speed_px_s * 0.5)

    def is_static_static_pair(self, a: TrackedObject, b: TrackedObject) -> bool:
        a_static = self.is_stationary(a.track_id) or self.is_parked(a.track_id)
        b_static = self.is_stationary(b.track_id) or self.is_parked(b.track_id)
        return a_static and b_static

    def is_normal_passing_pair(self, a: TrackedObject, b: TrackedObject, ev: SignalEvidence) -> bool:
        a_static = self.is_stationary(a.track_id) or self.is_parked(a.track_id)
        b_static = self.is_stationary(b.track_id) or self.is_parked(b.track_id)
        if a_static == b_static:
            return False
        if self.has_moving_to_stationary_impact(a, b, ev):
            return False
        low_disruption = (
            ev.signal_scores["velocity_drop_score"] < self.velocity_drop_threshold
            and ev.signal_scores["post_impact_stagnation_score"] < 0.45
            and ev.signal_scores["direction_change_score"] < 0.35
            and ev.signal_scores["optical_flow_anomaly_score"] < self.optical_flow_threshold
        )
        no_approach_impact = not ev.signal_flags["approach_to_contact"] and not ev.signal_flags["closing_distance_before_contact"]
        return bool(low_disruption and no_approach_impact)

    def has_moving_to_stationary_impact(self, a: TrackedObject, b: TrackedObject, ev: SignalEvidence) -> bool:
        a_static = self.is_stationary(a.track_id) or self.is_parked(a.track_id)
        b_static = self.is_stationary(b.track_id) or self.is_parked(b.track_id)
        if a_static == b_static:
            return False
        moving_track = b.track_id if a_static else a.track_id
        moving_peak = self._peak_speed[moving_track]
        impact_contact = ev.signal_flags["contact_or_near_overlap"] and ev.signal_flags["proximity"]
        moving_disruption = (
            ev.signal_flags["velocity_drop"]
            or ev.signal_flags["post_impact_stagnation"]
            or ev.signal_flags["direction_change"]
            or ev.signal_scores["velocity_drop_score"] >= self.velocity_drop_threshold * 0.75
        )
        path_conflict = (
            ev.signal_flags["trajectory_conflict"]
            or ev.signal_flags["closing_distance_before_contact"]
            or ev.signal_flags["approach_to_contact"]
            or ev.signal_scores["near_overlap_score"] >= 0.85
        )
        return bool(moving_peak >= self.min_peak_speed_for_collision and impact_contact and path_conflict and moving_disruption)

    def should_suppress_candidate(
        self,
        a: TrackedObject,
        b: TrackedObject,
        state: _PairState,
        ev: SignalEvidence,
    ) -> str:
        if self.suppress_static_static and self.is_static_static_pair(a, b):
            return "static_static_pair"
        if self.suppress_normal_passing_parked and self.is_normal_passing_pair(a, b, ev):
            return "normal_passing_pair"
        return self._false_positive_reason(a, b, state, ev)

    def _false_positive_reason(
        self,
        a: TrackedObject,
        b: TrackedObject,
        state: _PairState,
        ev: SignalEvidence,
    ) -> str:
        peak_a = self._peak_speed[a.track_id]
        peak_b = self._peak_speed[b.track_id]
        if max(peak_a, peak_b) < self.min_peak_speed_for_collision:
            return "both_vehicles_stationary_or_parked"

        if self.has_moving_to_stationary_impact(a, b, ev):
            return ""

        flags = ev.signal_flags
        scores = ev.signal_scores
        vel_a = self._velocity_from_history(a.track_id, frames=self.pre_window_frames)
        vel_b = self._velocity_from_history(b.track_id, frames=self.pre_window_frames)
        same_dir = self._same_direction(vel_a, vel_b)
        low_disruption = (
            scores["velocity_drop_score"] < self.velocity_drop_threshold * 0.65
            and scores["post_impact_stagnation_score"] < 0.45
            and scores["direction_change_score"] < 0.35
            and scores["optical_flow_anomaly_score"] < self.optical_flow_threshold
        )

        if self.suppress_same_direction_traffic and same_dir and low_disruption:
            if not ev.signal_flags["closing_distance_before_contact"] and not ev.signal_flags["approach_to_contact"]:
                return "same_direction_flow_without_impact"
            return "same_direction_close_traffic_or_overtaking"

        if (
            ev.iou >= self.min_contact_iou
            and low_disruption
            and not ev.signal_flags["closing_distance_before_contact"]
            and not ev.signal_flags["approach_to_contact"]
            and not flags["trajectory_conflict"]
        ):
            return "occlusion_overlap_without_motion_anomaly"

        if ev.iou < self.grazing_iou_limit and not flags["velocity_drop"] and not flags["post_impact_stagnation"]:
            if flags["contact_or_near_overlap"] and flags["relative_motion"] and not flags["trajectory_conflict"]:
                return "grazing_or_visual_crossing_without_post_impact_anomaly"

        if state.state == EventState.CANDIDATE and state.weak_frames > self.confirm_frames + self.fp_frames:
            return "candidate_timed_out_without_persistent_evidence"

        return ""

    def _post_impact_validated(self, state: _PairState) -> Tuple[bool, List[str]]:
        recent = state.recent(max(self.post_impact_frames, self.min_confirming_frames))
        if not recent:
            return False, []
        reasons: List[str] = []

        hard_reasons: List[str] = []
        supporting_reasons: List[str] = []
        if any(ev.signal_flags["velocity_drop"] for ev in recent):
            hard_reasons.append("post_impact_abrupt_deceleration")
        if any(ev.signal_flags["post_impact_stagnation"] for ev in recent):
            hard_reasons.append("post_impact_stalled_or_stopped")
        if any(ev.signal_flags["direction_change"] for ev in recent):
            hard_reasons.append("post_impact_direction_change")
        if any(ev.signal_flags["closing_distance_before_contact"] and ev.signal_flags["contact_or_near_overlap"] for ev in recent):
            hard_reasons.append("prediction_error_spike_after_contact")
        if any(ev.signal_flags["optical_flow_anomaly"] for ev in recent):
            supporting_reasons.append("post_impact_flow_anomaly")

        close_frames = sum(1 for ev in recent if ev.signal_flags["proximity"] and ev.signal_flags["contact_or_near_overlap"])
        contact_persistence = False
        if close_frames >= max(2, min(self.post_impact_frames, len(recent)) // 2):
            contact_persistence = True
            supporting_reasons.append("contact_persistence")

        if contact_persistence and any(ev.signal_flags["trajectory_conflict"] for ev in recent):
            supporting_reasons.append("predicted_path_conflict_persisted")

        reasons = hard_reasons + supporting_reasons
        hard_count = len(set(hard_reasons))
        support_count = len(set(supporting_reasons))
        enough_hard = hard_count >= self.min_hard_impact_signals
        enough_support = support_count >= self.min_supporting_impact_signals
        if self.require_hard_impact_signal and not (
            (enough_hard and enough_support) or hard_count >= max(2, self.min_hard_impact_signals + 1)
        ):
            return False, []
        return bool(reasons), reasons

    def _make_event(
        self,
        state: _PairState,
        evidence: SignalEvidence,
        severity: float,
        post_reasons: List[str],
    ) -> AccidentEvent:
        recent = state.recent(self.confirm_frames)
        iou_max = max([ev.iou for ev in recent], default=evidence.iou)
        motion_score = max(
            evidence.signal_scores.get("velocity_drop_score", 0.0),
            evidence.signal_scores.get("optical_flow_anomaly_score", 0.0),
            evidence.signal_scores.get("post_impact_stagnation_score", 0.0),
            evidence.signal_scores.get("direction_change_score", 0.0),
        )
        triggered = [name for name, fired in evidence.signal_flags.items() if fired]
        debug_doc = {
            "state": state.state.value,
            "candidate_frame": state.candidate_frame,
            "impact_frame": state.impact_frame,
            "confirmed_frame": state.confirmed_frame,
            "confirmation_count": state.confirming_frames,
            "max_severity_seen": state.max_severity_seen,
            "post_impact_seen": state.post_impact_seen,
            "post_impact_reasons": post_reasons,
            "recent_evidence": [ev.to_json() for ev in recent[-10:]],
        }
        logger.warning(
            "ACCIDENT CONFIRMED",
            event_id=state.event_id,
            pair=list(evidence.pair_key),
            severity=round(severity, 3),
            signals=triggered,
            frame_id=evidence.frame_id,
        )
        return AccidentEvent(
            track_id_a=state.track_id_a,
            track_id_b=state.track_id_b,
            timestamp=time.time(),
            frame_id=evidence.frame_id,
            severity_score=round(float(severity), 4),
            iou_at_collision=round(float(iou_max), 4),
            motion_anomaly_score=round(float(motion_score), 4),
            signals_triggered=triggered,
            confirmed=True,
            event_id=state.event_id,
            signal_scores=dict(evidence.signal_scores),
            debug=debug_doc,
        )

    def _next_event_id(self, frame_id: int, pair_key: Tuple[int, int]) -> str:
        self._event_counter += 1
        return f"ACC-{frame_id:06d}-{pair_key[0]}-{pair_key[1]}-{self._event_counter:03d}"

    def _write_debug_record(
        self,
        record_type: str,
        state: _PairState,
        ev: SignalEvidence,
        debug: bool,
        wait_reason: str = "",
    ) -> None:
        if wait_reason:
            state.last_wait_reason = wait_reason
        if not self._should_emit_debug_record(record_type, state, ev, debug):
            return
        if debug:
            logger.debug(
                "ACCIDENT_DEBUG",
                record_type=record_type,
                event_id=state.event_id,
                pair=list(ev.pair_key),
                frame=ev.frame_id,
                state=state.state.value,
                severity=round(ev.severity, 3),
                max_severity_seen=round(state.max_severity_seen, 3),
                active_signals=ev.active_signal_count,
                confirming_frame_count=state.confirming_frames,
                required=self.min_confirming_frames,
                post_impact_seen=state.post_impact_seen,
                wait_reason=state.last_wait_reason,
                suppressed=ev.suppressed_reason,
            )
        if not self.debug_events and record_type != "confirmed":
            return
        try:
            self.debug_events_path.parent.mkdir(parents=True, exist_ok=True)
            payload = ev.to_json()
            payload.update({
                "type": record_type,
                "event_id": state.event_id,
                "confirmation_count": state.confirming_frames,
                "confirming_frame_count": state.confirming_frames,
                "required_confirming_frames": self.min_confirming_frames,
                "candidate_start_frame": state.candidate_frame,
                "candidate_start_time": state.candidate_start_time,
                "max_severity_seen": round(state.max_severity_seen, 4),
                "post_impact_seen": state.post_impact_seen,
                "post_impact_reasons": list(state.post_impact_reasons),
                "suppressed": bool(ev.suppressed_reason),
                "suppressed_reason": ev.suppressed_reason,
                "wait_reason": state.last_wait_reason,
                "timestamp": time.time(),
            })
            with open(self.debug_events_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
        except OSError as exc:
            logger.warning("Failed to write accident debug record", exc=str(exc))

    def _should_emit_debug_record(
        self,
        record_type: str,
        state: _PairState,
        ev: SignalEvidence,
        debug: bool,
    ) -> bool:
        if record_type == "confirmed":
            return True
        if not debug and not self.debug_events:
            return False
        if state.state in {EventState.SUSPECT, EventState.CANDIDATE, EventState.CONFIRMED}:
            return True
        if record_type in {"candidate_wait", "candidate_closed"}:
            return True
        if ev.severity >= max(0.05, self.candidate_threshold * 0.85):
            return True
        return ev.frame_id % self.debug_record_normal_interval == 0

    def _expire_unseen_pairs(self, seen_pairs: set[Tuple[int, int]], frame_id: int) -> None:
        for key, state in list(self._pairs.items()):
            if key in seen_pairs:
                continue
            if state.state in {EventState.SUSPECT, EventState.CANDIDATE}:
                state.weak_frames += 1
                state.missing_frames += 1
                if state.state == EventState.CANDIDATE:
                    state.confirmation_window.append(0)
                    state.hard_impact_window.append(0)
                    state.confirming_frames = int(sum(state.confirmation_window))
                    state.hard_impact_frames = int(sum(state.hard_impact_window))
                if state.weak_frames > self.state_grace_frames:
                    state.state = EventState.CLOSED
                    state.closed_frame = frame_id
            elif state.state == EventState.CONFIRMED and frame_id > state.confirmed_until_frame:
                state.state = EventState.CLOSED
                state.closed_frame = frame_id
            if frame_id - state.last_seen_frame > int(self.cooldown_seconds * self.fps * 2):
                self._pairs.pop(key, None)

    def _velocity_from_history(self, track_id: int, frames: int) -> np.ndarray:
        hist = list(self._centroid_history[track_id])
        n = min(max(2, frames), len(hist))
        if n < 2:
            return np.zeros(2, dtype=float)
        pts = hist[-n:]
        first = np.array(pts[0], dtype=float)
        last = np.array(pts[-1], dtype=float)
        return (last - first) / max(n - 1, 1)

    def _speed_drop_score(self, track_id: int, avg_diag: float) -> float:
        hist = list(self._speed_history[track_id])
        if len(hist) < 4:
            return 0.0
        split = max(2, min(len(hist) // 2, self.pre_window_frames // 2))
        before = hist[-self.pre_window_frames:-split] if len(hist) > self.pre_window_frames else hist[:split]
        after = hist[-min(4, len(hist)):]
        if not before or not after:
            return 0.0
        pre = float(np.median(before))
        post = float(np.median(after))
        if pre < 1.0:
            return 0.0
        drop_ratio = max(0.0, (pre - post) / max(pre, 1.0))
        drop_px = max(0.0, pre - post) / max(avg_diag, 1.0)
        return _clamp01(max(drop_ratio, drop_px))

    def _stagnation_score(self, track_id: int) -> float:
        hist = list(self._speed_history[track_id])
        if len(hist) < 4:
            return 0.0
        peak = self._peak_speed[track_id]
        recent = float(np.median(hist[-min(5, len(hist)):]))
        if peak < self.min_peak_speed_for_collision:
            return 0.0
        return _clamp01((peak - recent) / max(peak, 1.0))

    def _direction_change_score(self, track_id: int) -> float:
        hist = list(self._centroid_history[track_id])
        if len(hist) < max(6, self.pre_window_frames // 2):
            return 0.0
        n = min(self.pre_window_frames, len(hist))
        window = hist[-n:]
        early = np.array(window[min(2, len(window) - 1)], dtype=float) - np.array(window[0], dtype=float)
        late = np.array(window[-1], dtype=float) - np.array(window[max(0, len(window) - 3)], dtype=float)
        early_norm = float(np.linalg.norm(early))
        late_norm = float(np.linalg.norm(late))
        if early_norm < 1.0 or late_norm < 1.0:
            return 0.0
        cos = float(np.clip(np.dot(early, late) / (early_norm * late_norm), -1.0, 1.0))
        angle = math.degrees(math.acos(cos))
        return _clamp01(angle / 120.0)

    def _trajectory_conflict_score(self, track_a: int, track_b: int, avg_diag: float) -> Tuple[float, List[str]]:
        hist_a = list(self._centroid_history[track_a])
        hist_b = list(self._centroid_history[track_b])
        if len(hist_a) < 3 or len(hist_b) < 3:
            return 0.0, []

        va = self._velocity_from_history(track_a, frames=self.pre_window_frames)
        vb = self._velocity_from_history(track_b, frames=self.pre_window_frames)
        ca = np.array(hist_a[-1], dtype=float)
        cb = np.array(hist_b[-1], dtype=float)
        speed_a = float(np.linalg.norm(va))
        speed_b = float(np.linalg.norm(vb))
        if max(speed_a, speed_b) < 0.15:
            return 0.0, []

        d0 = float(np.linalg.norm(ca - cb))
        predicted_distances = [
            float(np.linalg.norm((ca + va * step) - (cb + vb * step)))
            for step in range(1, self.trajectory_prediction_horizon + 1)
        ]
        min_future = min(predicted_distances) if predicted_distances else d0
        convergence = _clamp01((d0 - min_future) / max(d0, avg_diag, 1.0))
        conflict = _clamp01(1.0 - min_future / max(avg_diag * 0.9, 1.0))

        angle_score = 0.0
        if speed_a > 0.2 and speed_b > 0.2:
            cos = float(np.clip(np.dot(va, vb) / (speed_a * speed_b), -1.0, 1.0))
            angle = math.degrees(math.acos(cos))
            angle_score = _clamp01((angle - self.approach_angle_deg) / max(120.0 - self.approach_angle_deg, 1.0))

        score = _clamp01(0.45 * conflict + 0.35 * convergence + 0.20 * angle_score)
        reasons = []
        if conflict >= 0.45:
            reasons.append("predicted_path_conflict")
        if convergence >= 0.35:
            reasons.append("trajectory_convergence")
        return score, reasons

    def _optical_flow_score(self, flow_map: Optional[np.ndarray], bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
        if flow_map is None:
            return 0.0
        global_level = float(np.median(flow_map))
        self._flow_level_history.append(global_level)
        baseline = float(np.median(self._flow_level_history)) if self._flow_level_history else global_level

        def region_level(bbox: np.ndarray) -> float:
            h, w = flow_map.shape[:2]
            x1 = int(max(0, min(w, bbox[0])))
            y1 = int(max(0, min(h, bbox[1])))
            x2 = int(max(0, min(w, bbox[2])))
            y2 = int(max(0, min(h, bbox[3])))
            if x2 <= x1 or y2 <= y1:
                return 0.0
            return float(np.median(flow_map[y1:y2, x1:x2]))

        pair_level = max(region_level(bbox_a), region_level(bbox_b))
        camera_shake_adjusted = max(0.0, pair_level - max(global_level, baseline))
        return _clamp01(camera_shake_adjusted / max(self.optical_flow_threshold * 20.0, 1.0))

    def _compute_flow_map(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return None
        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            self._prev_gray = gray
            return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        except cv2.error:
            self._prev_gray = gray
            return None

    def _prime_gray(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            self._prev_gray = None
        return None

    def _same_direction(self, va: np.ndarray, vb: np.ndarray) -> bool:
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na < 0.2 or nb < 0.2:
            return False
        cos = float(np.dot(va, vb) / (na * nb))
        return cos >= self.same_direction_cosine

    def _is_moving(self, track_id: int, bbox: np.ndarray) -> bool:
        hist = list(self._centroid_history[track_id])
        if len(hist) < 2:
            return False
        diag = self._bbox_diagonal(bbox)
        n = min(self.pre_window_frames, len(hist))
        window = hist[-n:]
        disps = [
            float(np.linalg.norm(np.array(window[i + 1]) - np.array(window[i]))) / diag
            for i in range(len(window) - 1)
        ]
        return bool(disps and float(np.median(disps)) >= self.moving_threshold)

    def _was_moving_before(self, track_id: int, bbox: np.ndarray) -> bool:
        return self._is_moving(track_id, bbox) or self._peak_speed[track_id] >= self.min_peak_speed_for_collision

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
        xa1, ya1, xa2, ya2 = [float(x) for x in a]
        xb1, yb1, xb2, yb2 = [float(x) for x in b]
        ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
        ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        return float(inter / max(area_a + area_b - inter, 1e-6))

    @staticmethod
    def _bbox_edge_gap(a: np.ndarray, b: np.ndarray) -> float:
        xa1, ya1, xa2, ya2 = [float(x) for x in a]
        xb1, yb1, xb2, yb2 = [float(x) for x in b]
        h_gap = max(xb1 - xa2, xa1 - xb2, 0.0)
        v_gap = max(yb1 - ya2, ya1 - yb2, 0.0)
        if h_gap == 0 and v_gap == 0:
            return -1.0
        return float(h_gap + v_gap)

    @staticmethod
    def _bbox_diagonal(bbox: np.ndarray) -> float:
        x1, y1, x2, y2 = [float(x) for x in bbox]
        return float(math.sqrt(max(0.0, x2 - x1) ** 2 + max(0.0, y2 - y1) ** 2)) + 1e-6


AccidentFusionEngine = CollisionDetector
AccidentFusion = AccidentFusionEngine
