"""trackers/tracker_manager.py — ByteTrack (default) + StrongSORT wrapper.

Choice rationale:
  ByteTrack: pure IoU + Kalman, no per-frame appearance embedding → fastest on CPU.
  StrongSORT: adds OSNet Re-ID appearance embedding for higher ID stability at cost
              of ~12ms extra per frame — enabled when reid_gap_seconds matters.
  Both integrated via Ultralytics built-in BoT-SORT/ByteTrack support.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    bbox: np.ndarray          # [x1,y1,x2,y2]
    centroid: np.ndarray      # [cx,cy]
    speed_px_s: float = 0.0
    trajectory: deque = field(default_factory=lambda: deque(maxlen=60))
    last_seen_frame: int = 0
    appearance_embedding: Optional[np.ndarray] = None
    plate_text: str = ""
    ocr_confidence: float = 0.0


class SpeedEstimator:
    """Rolling-average speed estimator from trajectory."""

    def __init__(self, pixels_per_meter: float = 8.0, fps: float = 25.0, window: int = 5):
        self.ppm = pixels_per_meter
        self.fps = fps
        self.window = window

    def estimate(self, trajectory: deque) -> float:
        pts = list(trajectory)
        if len(pts) < 2:
            return 0.0
        recent = pts[-min(self.window, len(pts)):]
        dists = [
            float(np.linalg.norm(np.array(b) - np.array(a)))
            for a, b in zip(recent, recent[1:])
        ]
        if not dists:
            return 0.0
        avg_px_per_frame = sum(dists) / len(dists)
        return avg_px_per_frame * self.fps  # px/s


class TrackerManager:
    """
    Wraps Ultralytics ByteTrack via model.track() results.
    Maintains TrackedObject registry with trajectory history and speed.
    Handles re-ID gap logic.
    """

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        reid_gap_seconds: float = 10.0,
        fps: float = 25.0,
        pixels_per_meter: float = 8.0,
        speed_window: int = 5,
    ) -> None:
        self.tracker_type = tracker_type
        self.reid_gap_frames = int(reid_gap_seconds * fps)
        self.fps = fps
        self.speed_est = SpeedEstimator(pixels_per_meter, fps, speed_window)
        self.tracks: Dict[int, TrackedObject] = {}
        self._frame_counter: int = 0

    # ──────────────────────────────────────────────────────────
    def update(
        self,
        track_ids: np.ndarray,        # shape (N,) int
        bboxes: np.ndarray,           # shape (N,4) float [x1,y1,x2,y2]
        class_ids: np.ndarray,        # shape (N,) int
        class_names: List[str],
    ) -> List[TrackedObject]:
        """Called once per processed frame with tracker output."""
        self._frame_counter += 1
        active_ids = set()

        for tid, bbox, cid, cname in zip(track_ids, bboxes, class_ids, class_names):
            tid = int(tid)
            cx, cy = float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)
            centroid = np.array([cx, cy])

            if tid not in self.tracks:
                self.tracks[tid] = TrackedObject(
                    track_id=tid,
                    class_id=int(cid),
                    class_name=cname,
                    bbox=bbox.copy(),
                    centroid=centroid,
                )

            obj = self.tracks[tid]
            obj.bbox = bbox.copy()
            obj.centroid = centroid
            obj.trajectory.append((cx, cy))
            obj.last_seen_frame = self._frame_counter
            obj.speed_px_s = self.speed_est.estimate(obj.trajectory)
            active_ids.add(tid)

        # Prune stale tracks beyond re-ID gap
        stale = [
            tid for tid, obj in self.tracks.items()
            if self._frame_counter - obj.last_seen_frame > self.reid_gap_frames
            and tid not in active_ids
        ]
        for tid in stale:
            del self.tracks[tid]

        return [self.tracks[tid] for tid in active_ids]

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        return self.tracks.get(track_id)

    def get_all_active(self, within_frames: int = 5) -> List[TrackedObject]:
        threshold = self._frame_counter - within_frames
        return [o for o in self.tracks.values() if o.last_seen_frame >= threshold]

    @staticmethod
    def ultralytics_tracker_cfg(tracker_type: str) -> str:
        """Return the tracker config name for model.track(tracker=...)."""
        mapping = {
            "bytetrack": "bytetrack.yaml",
            "strongsort": "strongsort.yaml",
            "botsort": "botsort.yaml",
        }
        return mapping.get(tracker_type, "bytetrack.yaml")
