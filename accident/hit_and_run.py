"""accident/hit_and_run.py — Post-collision Hit-and-Run detection.

Logic:
  After a confirmed AccidentEvent between tracks A and B:
  - Monitor both tracks for the next `check_window_frames` frames.
  - If one track (suspect) continues moving at speed > flee_speed_px_s
    while the other (victim) is incapacitated (speed < incapacitated_speed_px_s),
    AND the suspect's centroid is moving toward the frame edge
    (or is already within edge_margin_px of any edge):
    → emit a HitAndRunEvent with the suspect track_id and last-known plate.

Designed to be called from the main pipeline worker thread after
accident_events = fusion.process_frame(...) returns confirmed events.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from trackers.tracker_manager import TrackedObject
from accident.fusion import AccidentEvent
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HitAndRunEvent:
    """Emitted when a suspect track flees after a confirmed accident."""
    suspect_track_id: int
    victim_track_id: int
    accident_frame_id: int       # original AccidentEvent frame
    detected_frame_id: int       # frame when flee was confirmed
    timestamp: float
    suspect_plate: str           # best OCR reading, empty if unknown
    suspect_last_centroid: Tuple[float, float]
    flee_speed_px_s: float
    severity_original: float     # severity from the triggering AccidentEvent


@dataclass
class _WatchEntry:
    """Internal state for a pair being monitored post-collision."""
    track_a: int
    track_b: int
    accident_frame: int
    severity: float
    # frame when each track was last confirmed incapacitated / fleeing
    frames_a_stationary: int = 0
    frames_b_stationary: int = 0
    frames_a_moving: int = 0
    frames_b_moving: int = 0
    emitted: bool = False


class HitAndRunMonitor:
    """
    Post-collision monitor.  Call register_accident() for each new AccidentEvent,
    then process_frame() every frame with current tracked objects.

    Parameters
    ----------
    frame_width, frame_height:
        Pixel dimensions of the inference frame (default 640×480).
        Used to detect edge-proximity of the suspect vehicle.
    edge_margin_px:
        How close to any frame edge (px) for a vehicle to be considered
        "heading for the edge / fleeing".  Default 80px (~12% of 640px).
    flee_speed_px_s:
        Minimum speed (px/s) for a post-collision vehicle to be classified
        as "actively fleeing".  Default 12 px/s at 15fps ≈ 0.8px/frame.
    incapacitated_speed_px_s:
        Maximum speed (px/s) for a vehicle to be classified as incapacitated
        (stopped/disabled) after a collision.  Default 4 px/s.
    consecutive_frames_to_confirm:
        How many consecutive frames both conditions must hold before emitting.
        Default 8 frames (~0.5s at 15fps) — avoids single-frame noise.
    check_window_frames:
        How many frames after the accident to monitor the pair before giving up.
        Default 60 frames (~4s at 15fps).
    """

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        edge_margin_px: int = 80,
        flee_speed_px_s: float = 12.0,
        incapacitated_speed_px_s: float = 4.0,
        consecutive_frames_to_confirm: int = 8,
        check_window_frames: int = 60,
    ) -> None:
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.edge_margin = edge_margin_px
        self.flee_speed = flee_speed_px_s
        self.incap_speed = incapacitated_speed_px_s
        self.confirm_frames = consecutive_frames_to_confirm
        self.check_window = check_window_frames

        self._watch: Dict[Tuple[int, int], _WatchEntry] = {}

    # ──────────────────────────────────────────────────────────────────────
    def register_accident(self, event: AccidentEvent) -> None:
        """Register a newly confirmed AccidentEvent for monitoring."""
        key = (min(event.track_id_a, event.track_id_b),
               max(event.track_id_a, event.track_id_b))
        if key not in self._watch or self._watch[key].emitted:
            self._watch[key] = _WatchEntry(
                track_a=event.track_id_a,
                track_b=event.track_id_b,
                accident_frame=event.frame_id,
                severity=event.severity_score,
            )

    # ──────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Clear post-collision watches after a scene transition."""
        self._watch.clear()

    def process_frame(
        self,
        tracked_objects: List[TrackedObject],
        frame_id: int,
    ) -> List[HitAndRunEvent]:
        """
        Call once per processed frame.  Returns list of newly confirmed
        HitAndRunEvent objects (typically 0 or 1 per frame).
        """
        if not self._watch:
            return []

        # Build fast lookup: track_id → TrackedObject
        obj_map: Dict[int, TrackedObject] = {o.track_id: o for o in tracked_objects}

        results: List[HitAndRunEvent] = []
        expired_keys = []

        for key, entry in self._watch.items():
            if entry.emitted:
                expired_keys.append(key)
                continue

            # Expire watch after check_window frames
            if frame_id - entry.accident_frame > self.check_window:
                expired_keys.append(key)
                continue

            obj_a = obj_map.get(entry.track_a)
            obj_b = obj_map.get(entry.track_b)

            # If both tracks gone from scene → expire
            if obj_a is None and obj_b is None:
                expired_keys.append(key)
                continue

            # Evaluate each track's kinematic state
            spd_a = obj_a.speed_px_s if obj_a else 0.0
            spd_b = obj_b.speed_px_s if obj_b else 0.0

            # Update consecutive-frame counters
            if spd_a < self.incap_speed:
                entry.frames_a_stationary += 1
                entry.frames_a_moving = 0
            else:
                entry.frames_a_moving += 1
                entry.frames_a_stationary = 0

            if spd_b < self.incap_speed:
                entry.frames_b_stationary += 1
                entry.frames_b_moving = 0
            else:
                entry.frames_b_moving += 1
                entry.frames_b_stationary = 0

            # Determine if one track is "victim" (stationary) and
            # the other is "suspect" (moving + near/toward edge)
            hit_run = None

            # Case A: track_a is victim, track_b is suspect
            if (entry.frames_a_stationary >= self.confirm_frames and
                    entry.frames_b_moving >= self.confirm_frames and
                    obj_b is not None and spd_b >= self.flee_speed):
                if self._near_or_toward_edge(obj_b):
                    hit_run = self._make_event(
                        suspect=obj_b, victim_id=entry.track_a,
                        entry=entry, frame_id=frame_id
                    )

            # Case B: track_b is victim, track_a is suspect
            elif (entry.frames_b_stationary >= self.confirm_frames and
                    entry.frames_a_moving >= self.confirm_frames and
                    obj_a is not None and spd_a >= self.flee_speed):
                if self._near_or_toward_edge(obj_a):
                    hit_run = self._make_event(
                        suspect=obj_a, victim_id=entry.track_b,
                        entry=entry, frame_id=frame_id
                    )

            if hit_run is not None:
                entry.emitted = True
                results.append(hit_run)
                logger.warning(
                    "HIT AND RUN DETECTED",
                    suspect_track=hit_run.suspect_track_id,
                    victim_track=hit_run.victim_track_id,
                    speed_px_s=round(hit_run.flee_speed_px_s, 1),
                    plate=hit_run.suspect_plate or "UNKNOWN",
                    accident_frame=hit_run.accident_frame_id,
                    detected_frame=hit_run.detected_frame_id,
                )

        for k in expired_keys:
            self._watch.pop(k, None)

        return results

    # ──────────────────────────────────────────────────────────────────────
    def _near_or_toward_edge(self, obj: TrackedObject) -> bool:
        """
        True if the vehicle is within edge_margin_px of any frame boundary,
        OR if its velocity vector is pointing toward the nearest edge.

        This two-condition check prevents false alerts for fast vehicles that
        happen to be in the middle of the frame (they're not fleeing).
        """
        cx, cy = float(obj.centroid[0]), float(obj.centroid[1])

        # Condition 1: already near an edge
        near_edge = (
            cx < self.edge_margin or
            cx > self.frame_w - self.edge_margin or
            cy < self.edge_margin or
            cy > self.frame_h - self.edge_margin
        )
        if near_edge:
            return True

        # Condition 2: velocity vector points toward nearest frame edge
        vx, vy = float(obj.velocity_vector[0]), float(obj.velocity_vector[1])
        if abs(vx) < 1.0 and abs(vy) < 1.0:
            return False  # essentially stationary, no direction to evaluate

        # Vector from centroid to nearest edge midpoint
        nearest_x = 0 if cx < self.frame_w / 2 else self.frame_w
        nearest_y = 0 if cy < self.frame_h / 2 else self.frame_h
        to_edge_x = nearest_x - cx
        to_edge_y = nearest_y - cy
        norm = np.sqrt(to_edge_x**2 + to_edge_y**2)
        if norm < 1e-3:
            return True
        # Dot product of velocity direction with edge direction
        # Positive = moving toward that edge corner
        dot = (vx * to_edge_x + vy * to_edge_y) / (norm * max(np.linalg.norm(obj.velocity_vector), 1e-3))
        return dot > 0.5  # moving at least 60° toward the frame corner

    def _make_event(
        self,
        suspect: TrackedObject,
        victim_id: int,
        entry: _WatchEntry,
        frame_id: int,
    ) -> HitAndRunEvent:
        cx, cy = float(suspect.centroid[0]), float(suspect.centroid[1])
        return HitAndRunEvent(
            suspect_track_id=suspect.track_id,
            victim_track_id=victim_id,
            accident_frame_id=entry.accident_frame,
            detected_frame_id=frame_id,
            timestamp=time.time(),
            suspect_plate=suspect.plate_text or "",
            suspect_last_centroid=(cx, cy),
            flee_speed_px_s=suspect.speed_px_s,
            severity_original=entry.severity,
        )
