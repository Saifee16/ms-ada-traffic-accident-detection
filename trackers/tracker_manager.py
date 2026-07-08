"""Track registry and kinematic state for YOLO/ByteTrack outputs."""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    bbox: np.ndarray
    centroid: np.ndarray
    speed_px_s: float = 0.0
    detection_confidence: float = 0.0
    trajectory: deque = field(default_factory=lambda: deque(maxlen=90))
    last_seen_frame: int = 0
    first_seen_frame: int = 0
    first_seen_timestamp: float = field(default_factory=time.time)
    last_seen_timestamp: float = field(default_factory=time.time)
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=30))
    active: bool = True
    appearance_embedding: Optional[np.ndarray] = None
    plate_text: str = ""
    ocr_confidence: float = 0.0
    vehicle_identifier: str = ""
    fallback_used: bool = False
    velocity_vector: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    acceleration: float = 0.0
    heading_deg: float = 0.0


class SpeedEstimator:
    def __init__(self, pixels_per_meter: float = 8.0, fps: float = 25.0, window: int = 5):
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.window = max(2, int(window))

    def estimate(self, trajectory: deque) -> float:
        if len(trajectory) < 2:
            return 0.0
        pts = list(trajectory)[-self.window:]
        if len(pts) < 2:
            return 0.0
        distances = [
            float(np.linalg.norm(np.array(pts[i], dtype=float) - np.array(pts[i - 1], dtype=float)))
            for i in range(1, len(pts))
        ]
        if not distances:
            return 0.0
        return float(np.mean(distances) * self.fps)


class TrackerManager:
    """Maintains stable track objects and short-occlusion ID continuity."""

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        reid_gap_seconds: float = 10.0,
        fps: float = 25.0,
        pixels_per_meter: float = 8.0,
        speed_window: int = 5,
        cluster_radius_px: float = 50.0,
    ) -> None:
        self.tracker_type = tracker_type
        self.fps = max(float(fps), 1.0)
        self.reid_gap_frames = max(1, int(reid_gap_seconds * self.fps))
        self.speed_est = SpeedEstimator(pixels_per_meter, self.fps, speed_window)
        self.tracks: Dict[int, TrackedObject] = {}
        self._frame_counter = 0
        self._prev_speed: Dict[int, float] = {}
        self._canonical_id: Dict[int, int] = {}
        self._direct_last_seen: Dict[int, int] = {}
        self._exited_tracks: List[TrackedObject] = []
        self.cluster_radius_px = float(cluster_radius_px)

    def update(
        self,
        track_ids: np.ndarray,
        bboxes: np.ndarray,
        class_ids: np.ndarray,
        class_names: List[str],
        confidences: Optional[np.ndarray] = None,
    ) -> List[TrackedObject]:
        self._frame_counter += 1
        active_ids: set[int] = set()
        direct_active_ids: set[int] = set()

        if confidences is None:
            confidences = np.ones(len(track_ids), dtype=float)

        for tid_raw, bbox, cid_raw, cname, conf_raw in zip(track_ids, bboxes, class_ids, class_names, confidences):
            original_tid = int(tid_raw)
            cid = int(cid_raw)
            conf = float(conf_raw)
            bbox = np.array(bbox, dtype=float)
            centroid = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=float)

            if original_tid not in self.tracks and original_tid not in self._canonical_id:
                merge_window = max(3, self.reid_gap_frames // 2)
                for existing_id, existing_obj in list(self.tracks.items()):
                    if existing_obj.class_id != cid:
                        continue
                    direct_age = self._frame_counter - self._direct_last_seen.get(
                        existing_id, existing_obj.last_seen_frame
                    )
                    dist = float(np.linalg.norm(centroid - existing_obj.centroid))
                    if direct_age <= merge_window and dist <= self.cluster_radius_px:
                        self._canonical_id[original_tid] = existing_id
                        break

            direct_active_ids.add(original_tid)
            tid = self._canonical_id.get(original_tid, original_tid)
            if tid == original_tid:
                self._direct_last_seen[tid] = self._frame_counter

            if tid not in self.tracks:
                self.tracks[tid] = TrackedObject(
                    track_id=tid,
                    class_id=cid,
                    class_name=cname,
                    bbox=bbox.copy(),
                    centroid=centroid.copy(),
                    detection_confidence=conf,
                    first_seen_frame=self._frame_counter,
                    last_seen_frame=self._frame_counter,
                )

            obj = self.tracks[tid]
            obj.class_id = cid
            obj.class_name = cname
            obj.bbox = bbox.copy()
            obj.centroid = centroid.copy()
            obj.detection_confidence = conf
            obj.confidence_history.append(conf)
            obj.trajectory.append((float(centroid[0]), float(centroid[1])))
            obj.last_seen_frame = self._frame_counter
            obj.last_seen_timestamp = time.time()
            obj.active = True
            obj.speed_px_s = self.speed_est.estimate(obj.trajectory)
            self._update_kinematics(obj)
            active_ids.add(tid)

        self._prune_stale(direct_active_ids)
        return [self.tracks[tid] for tid in active_ids if tid in self.tracks]

    def _update_kinematics(self, obj: TrackedObject) -> None:
        pts = list(obj.trajectory)
        if len(pts) < 4:
            obj.velocity_vector = np.zeros(2, dtype=float)
            obj.acceleration = 0.0
            return
        disp = np.array(pts[-1], dtype=float) - np.array(pts[-4], dtype=float)
        obj.velocity_vector = disp * (self.fps / 3.0)
        speed = float(np.linalg.norm(obj.velocity_vector))
        if speed > 0.5:
            obj.heading_deg = float(math.degrees(math.atan2(obj.velocity_vector[1], obj.velocity_vector[0])) % 360)
        prev = self._prev_speed.get(obj.track_id, speed)
        obj.acceleration = (speed - prev) * self.fps
        self._prev_speed[obj.track_id] = speed

    def _prune_stale(self, direct_active_ids: set[int]) -> None:
        stale = [
            tid for tid, obj in self.tracks.items()
            if self._frame_counter - self._direct_last_seen.get(tid, obj.last_seen_frame) > self.reid_gap_frames
            and tid not in direct_active_ids
        ]
        for tid in stale:
            obj = self.tracks.pop(tid)
            obj.active = False
            self._exited_tracks.append(obj)
            self._prev_speed.pop(tid, None)

        self._canonical_id = {k: v for k, v in self._canonical_id.items() if v in self.tracks}
        self._direct_last_seen = {k: v for k, v in self._direct_last_seen.items() if k in self.tracks}

    def pop_exited_tracks(self) -> List[TrackedObject]:
        exited = list(self._exited_tracks)
        self._exited_tracks = []
        return exited

    def reset(self) -> None:
        """Clear local track registry after a source transition/scene cut."""
        self.tracks.clear()
        self._prev_speed.clear()
        self._canonical_id.clear()
        self._direct_last_seen.clear()
        self._exited_tracks.clear()

    def set_vehicle_identifier(
        self,
        track_id: int,
        identifier: str,
        ocr_confidence: float = 0.0,
        fallback_used: bool = False,
    ) -> None:
        obj = self.tracks.get(track_id)
        if obj is None:
            return
        obj.vehicle_identifier = identifier
        obj.fallback_used = bool(fallback_used)
        obj.ocr_confidence = float(ocr_confidence)
        if not fallback_used:
            obj.plate_text = identifier

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        return self.tracks.get(track_id)

    def get_recent(self, track_id: int, n_frames: int) -> Dict[str, list]:
        obj = self.tracks.get(track_id)
        if obj is None:
            return {"bbox": [], "centroid": [], "speed": []}
        return {
            "bbox": [obj.bbox.tolist()],
            "centroid": list(obj.trajectory)[-n_frames:],
            "speed": [obj.speed_px_s],
        }

    def get_bbox_diagonal(self, track_id: int, idx: int = -1) -> float:
        obj = self.tracks.get(track_id)
        if obj is None:
            return 1.0
        x1, y1, x2, y2 = obj.bbox
        return float(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) + 1e-6

    def get_all_active(self, within_frames: int = 5) -> List[TrackedObject]:
        threshold = self._frame_counter - within_frames
        return [obj for obj in self.tracks.values() if obj.last_seen_frame >= threshold]

    @staticmethod
    def ultralytics_tracker_cfg(tracker_type: str) -> str:
        mapping = {
            "bytetrack": "bytetrack.yaml",
            "strongsort": "strongsort.yaml",
        }
        return mapping.get(tracker_type, "bytetrack.yaml")
