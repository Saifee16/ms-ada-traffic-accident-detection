"""Scene-cut and static accident-aftermath helpers.

Dynamic collision detection belongs in ``accident.fusion``.  This module handles
two separate concerns that should not create dynamic collision alerts:

* scene transition resets, so stale pair state does not leak across clips;
* conservative static aftermath detection for stopped damaged-scene clusters.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from trackers.tracker_manager import TrackedObject


@dataclass
class SceneCutResult:
    is_cut: bool
    score: float
    reason: str = ""


class SceneChangeDetector:
    """Lightweight frame-difference based scene-cut detector."""

    def __init__(
        self,
        enabled: bool = True,
        threshold: float = 0.45,
        consecutive_frames: int = 1,
        size: Tuple[int, int] = (64, 36),
    ) -> None:
        self.enabled = bool(enabled)
        self.threshold = float(threshold)
        self.consecutive_frames = max(1, int(consecutive_frames))
        self.size = size
        self._prev_small: Optional[np.ndarray] = None
        self._hits = 0

    def update(self, frame: np.ndarray) -> SceneCutResult:
        if not self.enabled:
            return SceneCutResult(False, 0.0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA)
        small = cv2.equalizeHist(small)
        if self._prev_small is None:
            self._prev_small = small
            return SceneCutResult(False, 0.0)
        score = float(np.mean(cv2.absdiff(self._prev_small, small)) / 255.0)
        self._prev_small = small
        if score >= self.threshold:
            self._hits += 1
        else:
            self._hits = 0
        is_cut = self._hits >= self.consecutive_frames
        if is_cut:
            self._hits = 0
        return SceneCutResult(is_cut, score, "frame_histogram_shift" if is_cut else "")

    def reset(self) -> None:
        self._prev_small = None
        self._hits = 0


@dataclass
class AccidentSceneEvent:
    event_id: str
    frame_id: int
    timestamp: float
    score: float
    bbox: Tuple[int, int, int, int]
    involved_track_ids: List[int] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    label: str = "ACCIDENT_SCENE_DETECTED"


@dataclass
class _SceneCandidate:
    region_key: Tuple[int, int]
    first_frame: int
    last_frame: int
    hits: int = 0
    max_score: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    track_ids: List[int] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    emitted_until_frame: int = -1
    event_id: str = ""


class AccidentSceneDetector:
    """Conservative fallback for static crash aftermath scenes.

    This is not a damaged-vehicle classifier.  It only detects persistent,
    abnormal stopped-vehicle clusters and labels them separately from dynamic
    collisions.
    """

    def __init__(
        self,
        enabled: bool = False,
        heuristic_enabled: bool = True,
        persistence_frames: int = 45,
        min_stopped_vehicles: int = 2,
        min_bbox_height_ratio: float = 0.06,
        stopped_speed_px_s: float = 4.0,
        cluster_radius_px: float = 150.0,
        score_threshold: float = 0.68,
        region_cooldown_frames: int = 300,
        display_frames: int = 90,
    ) -> None:
        self.enabled = bool(enabled)
        self.heuristic_enabled = bool(heuristic_enabled)
        self.persistence_frames = max(1, int(persistence_frames))
        self.min_stopped_vehicles = max(2, int(min_stopped_vehicles))
        self.min_bbox_height_ratio = float(min_bbox_height_ratio)
        self.stopped_speed_px_s = float(stopped_speed_px_s)
        self.cluster_radius_px = float(cluster_radius_px)
        self.score_threshold = float(score_threshold)
        self.region_cooldown_frames = max(1, int(region_cooldown_frames))
        self.display_frames = max(1, int(display_frames))
        self._speed_history: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=60))
        self._candidates: Dict[Tuple[int, int], _SceneCandidate] = {}
        self._event_counter = 0

    def reset(self) -> None:
        self._speed_history.clear()
        self._candidates.clear()

    def get_active_events(self, frame_id: int) -> List[AccidentSceneEvent]:
        events: List[AccidentSceneEvent] = []
        for candidate in self._candidates.values():
            if candidate.emitted_until_frame >= frame_id:
                events.append(self._make_event(candidate, frame_id, reuse_id=True))
        return events

    def process_frame(
        self,
        frame: np.ndarray,
        tracked_objects: Iterable[TrackedObject],
        frame_id: int,
    ) -> List[AccidentSceneEvent]:
        if not self.enabled or not self.heuristic_enabled:
            return []
        h, w = frame.shape[:2]
        vehicles = []
        for obj in tracked_objects:
            if int(obj.class_id) not in {1, 2, 3, 5, 7}:
                continue
            x1, y1, x2, y2 = [float(v) for v in obj.bbox]
            height_ratio = max(0.0, y2 - y1) / max(float(h), 1.0)
            if height_ratio < self.min_bbox_height_ratio:
                continue
            self._speed_history[obj.track_id].append(float(obj.speed_px_s or 0.0))
            if len(self._speed_history[obj.track_id]) < min(5, self.persistence_frames):
                continue
            median_speed = float(np.median(self._speed_history[obj.track_id]))
            if median_speed <= self.stopped_speed_px_s:
                vehicles.append(obj)

        clusters = self._cluster_stopped_vehicles(vehicles)
        events: List[AccidentSceneEvent] = []
        active_keys: set[Tuple[int, int]] = set()
        for cluster in clusters:
            score, bbox, reasons = self._score_cluster(cluster, frame.shape)
            if score < self.score_threshold:
                continue
            region_key = self._region_key(bbox, w, h)
            active_keys.add(region_key)
            candidate = self._candidates.get(region_key)
            if candidate is None:
                candidate = _SceneCandidate(region_key, frame_id, frame_id)
                self._candidates[region_key] = candidate
            candidate.last_frame = frame_id
            candidate.hits += 1
            candidate.max_score = max(candidate.max_score, score)
            candidate.bbox = bbox
            candidate.track_ids = [int(obj.track_id) for obj in cluster]
            candidate.reasons = reasons

            cooldown_active = candidate.emitted_until_frame >= frame_id
            if candidate.hits >= self.persistence_frames and not cooldown_active:
                candidate.event_id = self._next_event_id(candidate)
                candidate.emitted_until_frame = frame_id + self.region_cooldown_frames
                events.append(self._make_event(candidate, frame_id))

        for key, candidate in list(self._candidates.items()):
            if key in active_keys:
                continue
            if frame_id - candidate.last_frame > self.persistence_frames:
                self._candidates.pop(key, None)
        return events

    def _cluster_stopped_vehicles(self, vehicles: List[TrackedObject]) -> List[List[TrackedObject]]:
        clusters: List[List[TrackedObject]] = []
        for obj in vehicles:
            added = False
            center = np.array(obj.centroid, dtype=float)
            for cluster in clusters:
                distances = [float(np.linalg.norm(center - np.array(other.centroid, dtype=float))) for other in cluster]
                if min(distances, default=float("inf")) <= self.cluster_radius_px:
                    cluster.append(obj)
                    added = True
                    break
            if not added:
                clusters.append([obj])
        return [cluster for cluster in clusters if len(cluster) >= self.min_stopped_vehicles]

    def _score_cluster(
        self,
        cluster: List[TrackedObject],
        frame_shape: Tuple[int, ...],
    ) -> Tuple[float, Tuple[int, int, int, int], List[str]]:
        h, w = frame_shape[:2]
        x1 = min(float(obj.bbox[0]) for obj in cluster)
        y1 = min(float(obj.bbox[1]) for obj in cluster)
        x2 = max(float(obj.bbox[2]) for obj in cluster)
        y2 = max(float(obj.bbox[3]) for obj in cluster)
        bbox = (
            int(max(0, x1)),
            int(max(0, y1)),
            int(min(w - 1, x2)),
            int(min(h - 1, y2)),
        )
        count_score = min(1.0, len(cluster) / max(self.min_stopped_vehicles + 1, 1))
        heights = [max(0.0, float(obj.bbox[3] - obj.bbox[1])) / max(float(h), 1.0) for obj in cluster]
        size_score = min(1.0, float(np.mean(heights)) / max(self.min_bbox_height_ratio * 1.8, 1e-6))
        speeds = [float(np.median(self._speed_history[obj.track_id])) for obj in cluster]
        stopped_score = 1.0 - min(1.0, float(np.mean(speeds)) / max(self.stopped_speed_px_s * 1.5, 1.0))
        spread = math.hypot(max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1]))
        compact_score = 1.0 - min(1.0, spread / max(self.cluster_radius_px * 3.0, 1.0))
        score = 0.32 * count_score + 0.28 * size_score + 0.25 * stopped_score + 0.15 * compact_score
        reasons = ["persistent_stopped_vehicle_cluster"]
        if len(cluster) >= self.min_stopped_vehicles + 1:
            reasons.append("multiple_stopped_vehicles")
        if size_score >= 0.7:
            reasons.append("large_stationary_vehicle_boxes")
        return float(score), bbox, reasons

    def _region_key(self, bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int]:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        cell_w = max(1.0, width / 6.0)
        cell_h = max(1.0, height / 4.0)
        return int(cx // cell_w), int(cy // cell_h)

    def _next_event_id(self, candidate: _SceneCandidate) -> str:
        self._event_counter += 1
        return f"SCENE-{candidate.first_frame:06d}-{candidate.region_key[0]}-{candidate.region_key[1]}-{self._event_counter:03d}"

    def _make_event(self, candidate: _SceneCandidate, frame_id: int, reuse_id: bool = False) -> AccidentSceneEvent:
        if not candidate.event_id:
            candidate.event_id = self._next_event_id(candidate)
        return AccidentSceneEvent(
            event_id=candidate.event_id,
            frame_id=frame_id,
            timestamp=time.time(),
            score=round(candidate.max_score, 4),
            bbox=candidate.bbox,
            involved_track_ids=list(candidate.track_ids),
            reasons=list(candidate.reasons),
        )
