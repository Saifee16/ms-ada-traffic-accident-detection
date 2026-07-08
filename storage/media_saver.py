"""Evidence capture for confirmed accident events."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class MediaSaver:
    def __init__(
        self,
        snapshots_dir: str = "output/snapshots",
        clips_dir: str = "output/clips",
        evidence_dir: str = "output/evidence",
        clip_duration_seconds: float = 5.0,
        fps: float = 25.0,
        pre_event_seconds: float = 3.0,
        post_event_seconds: float = 2.0,
        save_snapshot_enabled: bool = True,
        save_clip_enabled: bool = True,
    ) -> None:
        self.snap_dir = Path(snapshots_dir)
        self.clips_dir = Path(clips_dir)
        self.evidence_dir = Path(evidence_dir)
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.fps = max(float(fps), 1.0)
        self.clip_duration = float(clip_duration_seconds)
        self.pre_event_seconds = float(pre_event_seconds)
        self.post_event_seconds = float(post_event_seconds)
        self.save_snapshot_enabled = bool(save_snapshot_enabled)
        self.save_clip_enabled = bool(save_clip_enabled)
        buffer_seconds = max(self.clip_duration, self.pre_event_seconds + self.post_event_seconds + 1.0)
        self._frame_buffer: Deque[Tuple[int, np.ndarray]] = deque(maxlen=max(1, int(buffer_seconds * self.fps)))
        self._lock = threading.Lock()

    def push_frame(self, frame_id: int, frame: np.ndarray) -> None:
        with self._lock:
            self._frame_buffer.append((int(frame_id), frame.copy()))

    def save_snapshot(self, frame: np.ndarray, tag: str = "") -> str:
        fname = f"snap_{tag}_{int(time.time())}.jpg"
        path = self.snap_dir / fname
        if self.save_snapshot_enabled:
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return str(path)

    def save_clip(self, frame: np.ndarray, tag: str = "") -> str:
        fname = f"clip_{tag}_{int(time.time())}.mp4"
        path = self.clips_dir / fname
        self._write_clip(path, frame)
        return str(path)

    def save_event_evidence(
        self,
        event_id: str,
        frame: np.ndarray,
        frame_id: int,
        metadata: Dict[str, Any],
    ) -> Tuple[str, str, str]:
        event_dir = self.evidence_dir / event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = event_dir / "snapshot.jpg"
        clip_path = event_dir / "clip.mp4"
        metadata_path = event_dir / "metadata.json"

        if self.save_snapshot_enabled:
            cv2.imwrite(str(snapshot_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if self.save_clip_enabled:
            self._write_clip(clip_path, frame, current_frame_id=frame_id)

        frame_start, frame_end = self._frame_range_for_clip(frame_id)
        full_metadata = {
            **metadata,
            "event_id": event_id,
            "snapshot_path": str(snapshot_path),
            "clip_path": str(clip_path),
            "frame_start": frame_start,
            "frame_end": frame_end,
        }
        metadata_path.write_text(json.dumps(full_metadata, indent=2, default=str), encoding="utf-8")
        logger.info("Evidence saved", event_id=event_id, dir=str(event_dir))
        return str(snapshot_path), str(clip_path), str(metadata_path)

    def update_metadata(self, event_id: str, updates: Dict[str, Any]) -> None:
        metadata_path = self.evidence_dir / event_id / "metadata.json"
        if not metadata_path.exists():
            return
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            data.update(updates)
            metadata_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to update evidence metadata", event_id=event_id, exc=str(exc))

    def _write_clip(self, path: Path, frame: np.ndarray, current_frame_id: int | None = None) -> None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
        with self._lock:
            frames = list(self._frame_buffer)
        if current_frame_id is not None:
            min_frame = current_frame_id - int(self.pre_event_seconds * self.fps)
            max_frame = current_frame_id + int(self.post_event_seconds * self.fps)
            frames = [(fid, img) for fid, img in frames if min_frame <= fid <= max_frame]
        for _, img in frames:
            if img.shape[:2] == (h, w):
                writer.write(img)
        writer.write(frame)
        writer.release()

    def _frame_range_for_clip(self, frame_id: int) -> Tuple[int, int]:
        min_frame = int(frame_id - self.pre_event_seconds * self.fps)
        max_frame = int(frame_id + self.post_event_seconds * self.fps)
        with self._lock:
            available = [fid for fid, _ in self._frame_buffer if min_frame <= fid <= max_frame]
        if not available:
            return int(frame_id), int(frame_id)
        return int(min(available)), int(max(max(available), frame_id))
