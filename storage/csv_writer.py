"""Thread-safe CSV/JSONL writers for surveillance outputs."""
from __future__ import annotations

import csv
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

DETECTION_FIELDNAMES = [
    "timestamp",
    "frame_number",
    "camera_id",
    "track_id",
    "vehicle_identifier",
    "class_name",
    "detection_confidence",
    "plate_text",
    "ocr_confidence",
    "fallback_used",
    "speed_px_per_s",
    "heading_x",
    "heading_y",
    "accident_flag",
    "event_id",
    "collision_partner_id",
    "iou_at_collision",
    "proximity_score",
    "trajectory_conflict_score",
    "velocity_drop_score",
    "optical_flow_anomaly_score",
    "severity_score",
    "snapshot_path",
    "clip_path",
]

ACCIDENT_FIELDNAMES = [
    "timestamp",
    "event_id",
    "camera_id",
    "frame_number",
    "track_id_a",
    "track_id_b",
    "vehicle_identifier_a",
    "vehicle_identifier_b",
    "severity_score",
    "signals",
    "iou_at_collision",
    "proximity_score",
    "trajectory_conflict_score",
    "velocity_drop_score",
    "optical_flow_anomaly_score",
    "snapshot_path",
    "clip_path",
]

COUNT_FIELDNAMES = [
    "timestamp",
    "camera_id",
    "window_start",
    "window_end",
    "total",
    "per_class_counts",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CSVWriter:
    def __init__(
        self,
        path: str = "output/detections.csv",
        accidents_path: str = "output/accidents.csv",
        counts_path: str = "output/counts.csv",
        debug_jsonl_path: str = "output/debug_events.jsonl",
    ) -> None:
        self._path = Path(path)
        self._accidents_path = Path(accidents_path)
        self._counts_path = Path(counts_path)
        self._debug_path = Path(debug_jsonl_path)
        self._lock = threading.Lock()
        self._init_csv(self._path, DETECTION_FIELDNAMES)
        self._init_csv(self._accidents_path, ACCIDENT_FIELDNAMES)
        self._init_csv(self._counts_path, COUNT_FIELDNAMES)
        self._debug_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _init_csv(path: Path, fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            try:
                with open(path, "r", newline="", encoding="utf-8") as fh:
                    reader = csv.reader(fh)
                    header = next(reader, [])
                if header == fieldnames:
                    return
                legacy = path.with_suffix(path.suffix + f".legacy_{int(time.time())}")
                path.replace(legacy)
                logger.warning("Rotated CSV with old schema", old=str(legacy), new=str(path))
            except OSError:
                pass
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    @staticmethod
    def detection_fieldnames() -> list[str]:
        return list(DETECTION_FIELDNAMES)

    def write_detection(
        self,
        camera_id: str,
        track_id: int,
        class_name: str,
        plate_text: str = "",
        ocr_confidence: float = 0.0,
        detection_confidence: float = 0.0,
        speed_px_per_s: float = 0.0,
        accident_flag: bool = False,
        iou_at_collision: float = 0.0,
        motion_anomaly_score: float = 0.0,
        snapshot_path: str = "",
        clip_path: str = "",
        frame_number: int = 0,
        vehicle_identifier: str = "",
        fallback_used: bool = False,
        heading_x: float = 0.0,
        heading_y: float = 0.0,
        event_id: str = "",
        collision_partner_id: str | int = "",
        proximity_score: float = 0.0,
        trajectory_conflict_score: float = 0.0,
        velocity_drop_score: float = 0.0,
        optical_flow_anomaly_score: Optional[float] = None,
        severity_score: float = 0.0,
        timestamp: Optional[str] = None,
    ) -> None:
        if optical_flow_anomaly_score is None:
            optical_flow_anomaly_score = motion_anomaly_score
        if not vehicle_identifier:
            vehicle_identifier = plate_text or ""
        row = {
            "timestamp": timestamp or _utc_now(),
            "frame_number": int(frame_number),
            "camera_id": camera_id,
            "track_id": int(track_id),
            "vehicle_identifier": vehicle_identifier,
            "class_name": class_name,
            "detection_confidence": round(float(detection_confidence), 4),
            "plate_text": plate_text,
            "ocr_confidence": round(float(ocr_confidence), 4),
            "fallback_used": int(bool(fallback_used)),
            "speed_px_per_s": round(float(speed_px_per_s), 2),
            "heading_x": round(float(heading_x), 4),
            "heading_y": round(float(heading_y), 4),
            "accident_flag": int(bool(accident_flag)),
            "event_id": event_id,
            "collision_partner_id": collision_partner_id,
            "iou_at_collision": round(float(iou_at_collision), 4),
            "proximity_score": round(float(proximity_score), 4),
            "trajectory_conflict_score": round(float(trajectory_conflict_score), 4),
            "velocity_drop_score": round(float(velocity_drop_score), 4),
            "optical_flow_anomaly_score": round(float(optical_flow_anomaly_score), 4),
            "severity_score": round(float(severity_score), 4),
            "snapshot_path": snapshot_path,
            "clip_path": clip_path,
        }
        self._append_csv(self._path, DETECTION_FIELDNAMES, row)

    def write_accident(
        self,
        camera_id: str,
        track_id_a: int,
        track_id_b: int,
        plate_a: str,
        plate_b: str,
        iou: float,
        motion_score: float,
        snapshot_path: str,
        clip_path: str,
        event_id: str = "",
        frame_number: int = 0,
        severity_score: float = 0.0,
        signal_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        signal_scores = signal_scores or {}
        for tid, partner, ident in [(track_id_a, track_id_b, plate_a), (track_id_b, track_id_a, plate_b)]:
            self.write_detection(
                camera_id=camera_id,
                track_id=tid,
                class_name="accident_participant",
                plate_text=ident if not ident.startswith("VEHICLE-ID-") else "",
                vehicle_identifier=ident,
                fallback_used=ident.startswith("VEHICLE-ID-"),
                accident_flag=True,
                iou_at_collision=iou,
                motion_anomaly_score=motion_score,
                snapshot_path=snapshot_path,
                clip_path=clip_path,
                event_id=event_id,
                frame_number=frame_number,
                collision_partner_id=partner,
                proximity_score=signal_scores.get("proximity_score", 0.0),
                trajectory_conflict_score=signal_scores.get("trajectory_conflict_score", 0.0),
                velocity_drop_score=signal_scores.get("velocity_drop_score", 0.0),
                optical_flow_anomaly_score=signal_scores.get("optical_flow_anomaly_score", motion_score),
                severity_score=severity_score,
            )

    def write_accident_event(
        self,
        camera_id: str,
        event_id: str,
        frame_number: int,
        track_id_a: int,
        track_id_b: int,
        vehicle_identifier_a: str,
        vehicle_identifier_b: str,
        severity_score: float,
        signals: list[str],
        iou_at_collision: float,
        signal_scores: Dict[str, float],
        snapshot_path: str,
        clip_path: str,
        timestamp: Optional[str] = None,
    ) -> None:
        row = {
            "timestamp": timestamp or _utc_now(),
            "event_id": event_id,
            "camera_id": camera_id,
            "frame_number": int(frame_number),
            "track_id_a": int(track_id_a),
            "track_id_b": int(track_id_b),
            "vehicle_identifier_a": vehicle_identifier_a,
            "vehicle_identifier_b": vehicle_identifier_b,
            "severity_score": round(float(severity_score), 4),
            "signals": ";".join(signals),
            "iou_at_collision": round(float(iou_at_collision), 4),
            "proximity_score": round(float(signal_scores.get("proximity_score", 0.0)), 4),
            "trajectory_conflict_score": round(float(signal_scores.get("trajectory_conflict_score", 0.0)), 4),
            "velocity_drop_score": round(float(signal_scores.get("velocity_drop_score", 0.0)), 4),
            "optical_flow_anomaly_score": round(float(signal_scores.get("optical_flow_anomaly_score", 0.0)), 4),
            "snapshot_path": snapshot_path,
            "clip_path": clip_path,
        }
        self._append_csv(self._accidents_path, ACCIDENT_FIELDNAMES, row)

    def write_count_window(self, camera_id: str, window_start: float, window_end: float, counts: Dict[str, int]) -> None:
        row = {
            "timestamp": _utc_now(),
            "camera_id": camera_id,
            "window_start": datetime.fromtimestamp(window_start, timezone.utc).isoformat(),
            "window_end": datetime.fromtimestamp(window_end, timezone.utc).isoformat(),
            "total": int(sum(counts.values())),
            "per_class_counts": json.dumps(counts, sort_keys=True),
        }
        self._append_csv(self._counts_path, COUNT_FIELDNAMES, row)

    def write_debug_event(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            with open(self._debug_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")

    def _append_csv(self, path: Path, fieldnames: list[str], row: Dict[str, Any]) -> None:
        with self._lock:
            with open(path, "a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=fieldnames).writerow(row)
