"""counting/counter.py — Sliding-window per-class vehicle counter."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CountWindow:
    window_start: float
    window_end: float
    camera_id: str
    counts: Dict[str, int] = field(default_factory=dict)
    total: int = 0


class SlidingWindowCounter:
    """
    Maintains per-class counts within a configurable rolling window.
    Emits a CountWindow record at each window boundary.
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        camera_id: str = "cam_01",
        on_window_close: Optional[Callable[[CountWindow], None]] = None,
    ) -> None:
        self.window_seconds = window_seconds
        self.camera_id = camera_id
        self._on_window_close = on_window_close
        self._window_start: float = time.time()
        self._counts: Dict[str, int] = defaultdict(int)
        self._cumulative: Dict[str, int] = defaultdict(int)
        self._seen_tracks: set = set()

    def update(self, track_id: int, class_name: str) -> Optional[CountWindow]:
        """
        Call every frame for each active track.
        Returns a CountWindow if the current window just closed.
        """
        now = time.time()
        closed_window: Optional[CountWindow] = None

        if now - self._window_start >= self.window_seconds:
            closed_window = self._close_window(now)

        key = (track_id, class_name)
        if key not in self._seen_tracks:
            self._seen_tracks.add(key)
            self._counts[class_name] += 1
            self._cumulative[class_name] += 1

        return closed_window

    def _close_window(self, now: float) -> CountWindow:
        cw = CountWindow(
            window_start=self._window_start,
            window_end=now,
            camera_id=self.camera_id,
            counts=dict(self._counts),
            total=sum(self._counts.values()),
        )
        logger.info("Window closed", camera=self.camera_id, counts=dict(self._counts), total=cw.total)
        if self._on_window_close:
            self._on_window_close(cw)
        # Reset for next window
        self._window_start = now
        self._counts = defaultdict(int)
        self._seen_tracks = set()
        return cw

    @property
    def cumulative(self) -> Dict[str, int]:
        return dict(self._cumulative)

    def force_close(self) -> CountWindow:
        return self._close_window(time.time())
