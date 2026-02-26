"""accident/fusion.py — Multi-signal accident detection and severity scoring.

Signals fused:
  1. IoU spike         — bounding-box overlap exceeds threshold
  2. Trajectory intersection — path crossing in recent history
  3. Sudden deceleration    — speed drop > ratio in short window
  4. Optical flow anomaly   — dense flow magnitude z-score spike
  5. Proximity violation    — centroid distance below threshold

Confirmation: all 5 signals evaluated every frame; accident confirmed only when
  ≥ min_signals_required persist for confirmation_seconds (configurable).
  N-frame confirmation window suppresses false positives.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy import stats

from trackers.tracker_manager import TrackedObject
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AccidentEvent:
    track_id_a: int
    track_id_b: int
    timestamp: float                  # Unix timestamp
    frame_id: int
    severity_score: float             # 0.0 – 1.0
    iou_at_collision: float
    motion_anomaly_score: float
    signals_triggered: List[str] = field(default_factory=list)
    confirmed: bool = False


@dataclass
class _PairState:
    track_id_a: int
    track_id_b: int
    signal_history: deque = field(default_factory=lambda: deque(maxlen=300))  # booleans per frame
    last_confirmed_time: float = 0.0
    cooldown_seconds: float = 30.0   # prevent repeat alerts — 30s stops spam on dense traffic


class AccidentFusion:
    """
    Frame-level fusion engine. Call process_frame() each frame.
    Returns list of newly confirmed AccidentEvent objects.
    """

    def __init__(
        self,
        fps: float = 25.0,
        confirmation_seconds: float = 2.0,
        iou_spike_threshold: float = 0.15,
        speed_drop_ratio: float = 0.5,
        optical_flow_zscore: float = 2.5,
        proximity_px: float = 60.0,
        min_signals: int = 3,
        severity_weights: Optional[Dict[str, float]] = None,
        fp_suppression_frames: int = 5,
        min_track_age_frames: int = 10,
        min_approach_speed_px: float = 5.0,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self.fps = fps
        self.confirm_frames = int(confirmation_seconds * fps)
        self.iou_thresh = iou_spike_threshold
        self.speed_drop = speed_drop_ratio
        self.flow_zscore = optical_flow_zscore
        self.proximity_px = proximity_px
        self.min_signals = min_signals
        self.fp_frames = fp_suppression_frames
        self.min_track_age = min_track_age_frames
        self.min_approach_speed = min_approach_speed_px
        self.cooldown_seconds = cooldown_seconds
        self.min_severity = 0.15
        self.weights = severity_weights or {
            "iou": 0.30, "trajectory": 0.25,
            "deceleration": 0.20, "optical_flow": 0.15, "proximity": 0.10,
        }

        self._pair_states: Dict[Tuple[int, int], _PairState] = {}
        self._prev_frame: Optional[np.ndarray] = None
        self._flow_history: deque = deque(maxlen=90)  # ~3s at 30fps
        self._speed_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

    # ──────────────────────────────────────────────────────────
    def process_frame(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        frame_id: int,
        debug: bool = False,
    ) -> List[AccidentEvent]:
        # 1. Update speed histories
        for obj in tracked_objects:
            self._speed_history[obj.track_id].append(obj.speed_px_s)

        # 2. Compute optical flow anomaly score (frame-level, not per-pair)
        flow_anomaly = self._compute_flow_anomaly(frame)

        # 3. Evaluate all pairs
        confirmed_events: List[AccidentEvent] = []
        vehicle_objs = [o for o in tracked_objects if o.class_id in {2, 3, 5, 7, 1}]

        for obj_a, obj_b in combinations(vehicle_objs, 2):
            pair_key = (min(obj_a.track_id, obj_b.track_id), max(obj_a.track_id, obj_b.track_id))

            # ── Gate 1: both tracks must have sufficient history ──────────────
            age_a, age_b = len(obj_a.trajectory), len(obj_b.trajectory)
            if age_a < self.min_track_age or age_b < self.min_track_age:
                if debug:
                    print(f"[F{frame_id}] GATE1_AGE  pair={pair_key} age_a={age_a} age_b={age_b} need={self.min_track_age}", flush=True)
                continue

            # ── Gate 2: pre-screen — must be spatially close enough ───────────
            pre_dist = float(np.linalg.norm(obj_a.centroid - obj_b.centroid))
            if pre_dist > self.proximity_px * 4:
                if debug:
                    print(f"[F{frame_id}] GATE2_DIST pair={pair_key} dist={pre_dist:.1f} limit={self.proximity_px*4:.0f}", flush=True)
                continue

            # ── Gate 3: vehicles must be approaching (or already colliding) ───
            approaching, approach_speed = self._is_approaching(obj_a, obj_b)
            if not approaching:
                if debug:
                    print(f"[F{frame_id}] GATE3_VEL  pair={pair_key} approach={approach_speed:.2f}px/4f need={self.min_approach_speed}", flush=True)
                continue

            if pair_key not in self._pair_states:
                self._pair_states[pair_key] = _PairState(
                    track_id_a=pair_key[0],
                    track_id_b=pair_key[1],
                    cooldown_seconds=self.cooldown_seconds,
                )
            state = self._pair_states[pair_key]

            # Skip if in cooldown
            if time.time() - state.last_confirmed_time < state.cooldown_seconds:
                if debug:
                    print(f"[F{frame_id}] COOLDOWN   pair={pair_key}", flush=True)
                continue

            signals, scores = self._evaluate_signals(obj_a, obj_b, flow_anomaly)
            state.signal_history.append(signals)

            # ── Fast-path: high IoU + any 1 other signal = immediate confirm ──
            # Boxes overlapping at > 15% IoU with proximity confirmed is a real collision.
            # Don't wait for the full confirmation window — vehicles only overlap briefly.
            current_iou = scores.get("iou", 0.0)
            current_signal_count = sum(signals.values())
            fast_confirm = (current_iou >= self.iou_thresh and current_signal_count >= 2)

            # ── Gate 4: current frame must itself meet min_signals ────────────
            if debug:
                active = [k for k, v in signals.items() if v]
                approach_spd = approach_speed
                print(f"[F{frame_id}] SIGNALS    pair={pair_key} active={active}({current_signal_count}) "
                      f"iou={scores['iou']:.3f} prox_gap={scores.get('proximity',0):.3f} "
                      f"approach={approach_spd:.1f}px/4f need_signals={self.min_signals}"
                      f"{' [FAST_PATH]' if fast_confirm else ''}", flush=True)
            if current_signal_count < self.min_signals and not fast_confirm:
                continue

            # ── Standard confirmation window ──────────────────────────────────
            if not fast_confirm:
                recent = list(state.signal_history)[-self.confirm_frames:]
                if len(recent) < self.confirm_frames:
                    if debug:
                        print(f"[F{frame_id}] CONFIRM_WAIT pair={pair_key} have={len(recent)}/{self.confirm_frames}", flush=True)
                    continue
                triggered_count = sum(
                    1 for s in recent if sum(s.values()) >= self.min_signals
                )
            if fast_confirm or triggered_count >= max(1, self.confirm_frames - self.fp_frames):
                severity = self._severity_score(scores)

                # ── Gate 5: require minimum severity ─────────────────────────
                if severity < self.min_severity:
                    if debug:
                        print(f"[F{frame_id}] GATE5_SEV  pair={pair_key} sev={severity:.3f} need={self.min_severity}", flush=True)
                    continue

                event = AccidentEvent(
                    track_id_a=pair_key[0],
                    track_id_b=pair_key[1],
                    timestamp=time.time(),
                    frame_id=frame_id,
                    severity_score=severity,
                    iou_at_collision=scores.get("iou", 0.0),
                    motion_anomaly_score=scores.get("optical_flow", 0.0),
                    signals_triggered=[k for k, v in signals.items() if v],
                    confirmed=True,
                )
                state.last_confirmed_time = time.time()
                state.signal_history.clear()
                confirmed_events.append(event)
                logger.warning(
                    "ACCIDENT CONFIRMED",
                    track_a=pair_key[0],
                    track_b=pair_key[1],
                    severity=round(severity, 3),
                    signals=event.signals_triggered,
                    frame_id=frame_id,
                )

        self._prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return confirmed_events

    # ──────────────────────────────────────────────────────────
    def _evaluate_signals(
        self,
        a: TrackedObject,
        b: TrackedObject,
        flow_anomaly: float,
    ) -> Tuple[Dict[str, bool], Dict[str, float]]:
        signals: Dict[str, bool] = {}
        scores: Dict[str, float] = {}

        # Signal 1: IoU spike
        iou = self._box_iou(a.bbox, b.bbox)
        scores["iou"] = float(iou)
        signals["iou"] = iou >= self.iou_thresh

        # Signal 2: Trajectory intersection
        traj_signal, traj_score = self._trajectory_intersection(a, b)
        signals["trajectory"] = traj_signal
        scores["trajectory"] = traj_score

        # Signal 3: Sudden deceleration
        decel_a = self._sudden_deceleration(a.track_id)
        decel_b = self._sudden_deceleration(b.track_id)
        signals["deceleration"] = decel_a or decel_b
        scores["deceleration"] = max(
            self._decel_score(a.track_id), self._decel_score(b.track_id)
        )

        # Signal 4: Optical flow anomaly (shared frame-level)
        signals["optical_flow"] = flow_anomaly >= self.flow_zscore
        scores["optical_flow"] = float(flow_anomaly)

        # Signal 5: Proximity — use bbox edge gap, not centroid distance
        # Centroid distance in dense traffic is almost always < 60px for adjacent vehicles.
        # Edge gap < 0 means actual bbox overlap (stronger signal).
        edge_gap = self._bbox_edge_gap(a.bbox, b.bbox)
        # Trigger if boxes are actually overlapping OR edge gap < 20px (touching/near-touching)
        signals["proximity"] = edge_gap < 20.0
        # Score: 1.0 at overlap, 0 at proximity_px gap
        scores["proximity"] = max(0.0, 1.0 - max(edge_gap, 0) / max(self.proximity_px, 1))

        return signals, scores

    def _box_iou(self, a: np.ndarray, b: np.ndarray) -> float:
        xa1, ya1, xa2, ya2 = a
        xb1, yb1, xb2, yb2 = b
        ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
        ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        return float(inter / (area_a + area_b - inter + 1e-6))

    def _trajectory_intersection(
        self, a: TrackedObject, b: TrackedObject
    ) -> Tuple[bool, float]:
        ta = list(a.trajectory)
        tb = list(b.trajectory)
        if len(ta) < 4 or len(tb) < 4:
            return False, 0.0
        # Only check the LAST 15 frames — full-history check fires for parallel-lane vehicles
        # since their long trajectories eventually cross spatially even without a real collision.
        recent_ta = ta[-15:]
        recent_tb = tb[-15:]
        for i in range(len(recent_ta) - 1):
            for j in range(len(recent_tb) - 1):
                if self._segments_intersect(recent_ta[i], recent_ta[i+1],
                                            recent_tb[j], recent_tb[j+1]):
                    return True, 1.0
        # Minimum distance between recent trajectory points
        min_dist = min(
            np.linalg.norm(np.array(p1) - np.array(p2))
            for p1 in recent_ta[-8:]
            for p2 in recent_tb[-8:]
        )
        score = max(0.0, 1.0 - min_dist / 150)   # raised divisor — needs to be really close
        return score > 0.85, float(score)          # raised threshold from 0.7 → 0.85

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4) -> bool:
        """2D line segment intersection test."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def _sudden_deceleration(self, track_id: int) -> bool:
        hist = list(self._speed_history[track_id])
        if len(hist) < 6:
            return False
        recent_avg = np.mean(hist[-3:])
        past_avg = np.mean(hist[-6:-3])
        if past_avg < 1e-3:
            return False
        return (past_avg - recent_avg) / past_avg >= self.speed_drop

    def _decel_score(self, track_id: int) -> float:
        hist = list(self._speed_history[track_id])
        if len(hist) < 6:
            return 0.0
        recent_avg = np.mean(hist[-3:])
        past_avg = np.mean(hist[-6:-3])
        if past_avg < 1e-3:
            return 0.0
        return float(min(1.0, max(0.0, (past_avg - recent_avg) / past_avg)))

    def _compute_flow_anomaly(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_frame is None or self._prev_frame.shape != gray.shape:
            return 0.0
        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_frame, gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            mean_mag = float(np.mean(magnitude))
            self._flow_history.append(mean_mag)
            if len(self._flow_history) < 10:
                return 0.0
            hist = list(self._flow_history)
            z = (mean_mag - np.mean(hist)) / (np.std(hist) + 1e-6)
            return float(z)
        except cv2.error:
            return 0.0

    def _severity_score(self, scores: Dict[str, float]) -> float:
        # Clamp each score to [0, 1] — optical_flow Z-score can be negative, which
        # previously allowed negative total severity scores.
        total = sum(
            self.weights.get(k, 0.0) * max(0.0, min(1.0, v))
            for k, v in scores.items()
        )
        return min(1.0, total)

    def _bbox_edge_gap(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Minimum gap in pixels between bbox edges.
        Negative value = actual overlap (boxes intersect).
        Much more precise than centroid distance for proximity detection.
        """
        xa1, ya1, xa2, ya2 = a
        xb1, yb1, xb2, yb2 = b
        # Horizontal and vertical gaps between the nearest edges
        h_gap = max(xb1 - xa2, xa1 - xb2, 0.0)  # 0 if overlapping horizontally
        v_gap = max(yb1 - ya2, ya1 - yb2, 0.0)  # 0 if overlapping vertically
        if h_gap == 0 and v_gap == 0:
            # Boxes overlap — return negative to signal actual intersection
            return -1.0
        return float(h_gap + v_gap)

    def _is_approaching(self, a: TrackedObject, b: TrackedObject) -> Tuple[bool, float]:
        """
        Return (is_approaching, approach_speed_px_per_4_frames).
        Bypasses speed check when boxes already overlap significantly (mid-collision state).
        Rejects pure parallel-lane traffic with near-zero relative velocity.
        """
        # Bypass: if boxes already overlap, they're in collision regardless of current velocity
        # (post-impact divergence would otherwise reject the event)
        iou = self._box_iou(a.bbox, b.bbox)
        if iou >= self.iou_thresh:
            return True, 999.0   # overlapping → definitely approaching/colliding

        ta = list(a.trajectory)
        tb = list(b.trajectory)
        if len(ta) < 4 or len(tb) < 4:
            return False, 0.0

        # Velocity vectors over last 4 processed frames
        vel_a = np.array(ta[-1]) - np.array(ta[-4])
        vel_b = np.array(tb[-1]) - np.array(tb[-4])

        # Project relative velocity onto unit vector from b→a
        # Positive approach_speed = vehicles closing in on each other
        delta_pos = np.array(a.centroid) - np.array(b.centroid)
        dist = float(np.linalg.norm(delta_pos))
        if dist < 1e-3:
            return True, 999.0

        unit = delta_pos / dist
        rel_vel = vel_a - vel_b
        approach_speed = -float(np.dot(rel_vel, unit))

        return approach_speed >= self.min_approach_speed, approach_speed