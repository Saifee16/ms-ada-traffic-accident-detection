"""storage/media_saver.py — Snapshot and clip saver for accident evidence."""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class MediaSaver:
    """
    Saves frame snapshots and short video clips as accident evidence.
    Maintains a rolling frame buffer for pre-event clip capture.
    """

    def __init__(
        self,
        snapshots_dir: str = "output/snapshots",
        clips_dir: str = "output/clips",
        clip_duration_seconds: float = 5.0,
        fps: float = 25.0,
    ) -> None:
        self.snap_dir = Path(snapshots_dir)
        self.clips_dir = Path(clips_dir)
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.clip_duration = clip_duration_seconds
        self.fps = fps
        self._buffer_size = int(clip_duration_seconds * fps)
        self._frame_buffer: Deque[Tuple[int, np.ndarray]] = deque(maxlen=self._buffer_size)
        self._lock = threading.Lock()

    def push_frame(self, frame_id: int, frame: np.ndarray) -> None:
        with self._lock:
            self._frame_buffer.append((frame_id, frame.copy()))

    def save_snapshot(self, frame: np.ndarray, tag: str = "") -> str:
        fname = f"snap_{tag}_{int(time.time())}.jpg"
        path = self.snap_dir / fname
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.debug("Snapshot saved", path=str(path))
        return str(path)

    def save_clip(self, frame: np.ndarray, tag: str = "") -> str:
        """
        Save buffered frames + current frame as an MP4 clip.
        Runs synchronously (call from alert thread to avoid blocking main pipeline).
        """
        fname = f"clip_{tag}_{int(time.time())}.mp4"
        path = self.clips_dir / fname
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))

        with self._lock:
            frames = list(self._frame_buffer)

        for _, f in frames:
            if f.shape[:2] == (h, w):
                out.write(f)
        out.write(frame)
        out.release()
        logger.debug("Clip saved", path=str(path), frames=len(frames))
        return str(path)
