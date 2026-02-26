"""storage/csv_writer.py — Thread-safe CSV writer for all detection/accident rows."""
from __future__ import annotations

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

CSV_FIELDNAMES = [
    "timestamp", "camera_id", "track_id", "class", "plate_text",
    "ocr_confidence", "detection_confidence", "speed_px_per_s",
    "accident_flag", "IoU_at_collision", "motion_anomaly_score",
    "snapshot_path", "clip_path",
]


class CSVWriter:
    """
    Single cumulative CSV with the mandated schema.
    Thread-safe via internal lock.
    """

    def __init__(self, path: str = "output/detections.csv") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_file()

    def _init_file(self) -> None:
        if not self._path.exists():
            with open(self._path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()

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
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "camera_id": camera_id,
            "track_id": track_id,
            "class": class_name,
            "plate_text": plate_text,
            "ocr_confidence": round(ocr_confidence, 4),
            "detection_confidence": round(detection_confidence, 4),
            "speed_px_per_s": round(speed_px_per_s, 2),
            "accident_flag": int(accident_flag),
            "IoU_at_collision": round(iou_at_collision, 4),
            "motion_anomaly_score": round(motion_anomaly_score, 4),
            "snapshot_path": snapshot_path,
            "clip_path": clip_path,
        }
        with self._lock:
            with open(self._path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
                writer.writerow(row)

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
    ) -> None:
        """Write one row per vehicle involved in confirmed accident."""
        for tid, plate in [(track_id_a, plate_a), (track_id_b, plate_b)]:
            self.write_detection(
                camera_id=camera_id,
                track_id=tid,
                class_name="accident_participant",
                plate_text=plate,
                accident_flag=True,
                iou_at_collision=iou,
                motion_anomaly_score=motion_score,
                snapshot_path=snapshot_path,
                clip_path=clip_path,
            )
