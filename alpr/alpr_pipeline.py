"""alpr/alpr_pipeline.py — ALPR pipeline: plate detection + EasyOCR + normalization.

Choice rationale:
  EasyOCR chosen over Tesseract for superior accuracy on non-ideal plate angles/lighting.
  Pakistani plate format regex covers standard provincial codes (ABC 1234),
  rivet/special series, and numeric-only formats.
  Retry-until-exit ensures we accumulate OCR readings across frames and pick best.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# Pakistan plate regex patterns (ordered by specificity)
_PK_PATTERNS: List[re.Pattern] = [
    re.compile(r'^[A-Z]{3}\s?\d{3,4}$'),      # ABC 1234 (standard)
    re.compile(r'^[A-Z]{2}\s?\d{3,4}$'),       # AB 1234
    re.compile(r'^[A-Z]{2}\s?\d{2,3}$'),       # AB 12 (older)
    re.compile(r'^[A-Z]{1}\s?\d{4}$'),         # A 1234
    re.compile(r'^RIV\s?\d{2,4}$'),            # Rivet series
    re.compile(r'^\d{4,5}$'),                  # numeric-only (some areas)
]

_FALLBACK_COUNTER_LOCK = threading.Lock()
_FALLBACK_COUNTER: Dict[str, int] = {}  # keyed by run_id


def normalize_plate(raw: str) -> str:
    """Strip noise, uppercase, collapse spaces, validate Pakistan format."""
    cleaned = re.sub(r'[^A-Z0-9 ]', '', raw.upper().strip())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    for pattern in _PK_PATTERNS:
        if pattern.match(cleaned):
            return cleaned
    return cleaned  # return cleaned even if format unrecognised


def fallback_plate_id(run_id: str = "default", prefix: str = "VEHICLE-ID") -> str:
    with _FALLBACK_COUNTER_LOCK:
        count = _FALLBACK_COUNTER.get(run_id, 0) + 1
        _FALLBACK_COUNTER[run_id] = count
    return f"{prefix}-{count:04d}"


@dataclass
class ALPRResult:
    plate_text: str
    confidence: float
    is_fallback: bool = False
    raw_readings: List[Tuple[str, float]] = field(default_factory=list)


class ALPRPipeline:
    """
    Per-vehicle ALPR state machine.
    Accumulates OCR readings while vehicle is visible; picks best on exit.
    """

    def __init__(
        self,
        min_confidence: float = 0.45,
        run_id: str = "default",
        fallback_prefix: str = "VEHICLE-ID",
        ocr_engine: str = "easyocr",
        languages: Optional[List[str]] = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.run_id = run_id
        self.fallback_prefix = fallback_prefix
        self._readings: Dict[int, List[Tuple[str, float]]] = {}  # track_id → list
        self._finalized: Dict[int, ALPRResult] = {}

        # Lazy-load OCR engine (heavy import)
        self._reader = None
        self._ocr_engine = ocr_engine
        self._languages = languages or ["en"]

    def _get_reader(self):
        if self._reader is None:
            if self._ocr_engine == "easyocr":
                import easyocr
                self._reader = easyocr.Reader(
                    self._languages,
                    gpu=self._is_gpu_available(),
                    verbose=False,
                )
                logger.info("EasyOCR reader initialized")
            elif self._ocr_engine == "paddleocr":
                from paddleocr import PaddleOCR
                self._reader = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=self._is_gpu_available())
        return self._reader

    @staticmethod
    def _is_gpu_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ──────────────────────────────────────────────────────────
    def process_vehicle_crop(
        self,
        track_id: int,
        vehicle_crop: np.ndarray,
        plate_bboxes: Optional[List[np.ndarray]] = None,
    ) -> Optional[Tuple[str, float]]:
        """
        Run OCR on detected plate region(s) within a vehicle crop.
        Returns (plate_text, confidence) or None if nothing detected.
        """
        if track_id in self._finalized:
            return self._finalized[track_id].plate_text, self._finalized[track_id].confidence

        if track_id not in self._readings:
            self._readings[track_id] = []

        reader = self._get_reader()

        regions: List[np.ndarray] = []
        if plate_bboxes:
            for pb in plate_bboxes:
                x1, y1, x2, y2 = map(int, pb)
                crop = vehicle_crop[max(0, y1):y2, max(0, x1):x2]
                if crop.size > 0:
                    regions.append(crop)
        else:
            # Fall back to lower-half heuristic
            h = vehicle_crop.shape[0]
            regions.append(vehicle_crop[h // 2:, :])

        for region in regions:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            try:
                if self._ocr_engine == "easyocr":
                    results = reader.readtext(thresh, detail=1)
                    for (_, text, prob) in results:
                        text_norm = normalize_plate(text)
                        if prob >= self.min_confidence and text_norm:
                            self._readings[track_id].append((text_norm, float(prob)))
                            return text_norm, float(prob)
                elif self._ocr_engine == "paddleocr":
                    results = reader.ocr(thresh, cls=True)
                    if results and results[0]:
                        for line in results[0]:
                            text, prob = line[1][0], line[1][1]
                            text_norm = normalize_plate(text)
                            if prob >= self.min_confidence and text_norm:
                                self._readings[track_id].append((text_norm, float(prob)))
                                return text_norm, float(prob)
            except Exception as exc:
                logger.warning("OCR error", exc=str(exc), track_id=track_id)

        return None

    def finalize(self, track_id: int) -> ALPRResult:
        """Called when vehicle exits frame. Returns best reading or fallback."""
        if track_id in self._finalized:
            return self._finalized[track_id]

        readings = self._readings.pop(track_id, [])
        if readings:
            best_text, best_conf = max(readings, key=lambda x: x[1])
            result = ALPRResult(
                plate_text=best_text,
                confidence=best_conf,
                is_fallback=False,
                raw_readings=readings,
            )
        else:
            result = ALPRResult(
                plate_text=fallback_plate_id(self.run_id, self.fallback_prefix),
                confidence=0.0,
                is_fallback=True,
            )

        self._finalized[track_id] = result
        logger.debug("ALPR finalized", track_id=track_id, plate=result.plate_text, conf=result.confidence)
        return result

    def get_best_current(self, track_id: int) -> Optional[Tuple[str, float]]:
        """Return best reading so far without finalizing."""
        readings = self._readings.get(track_id, [])
        if not readings:
            return None
        return max(readings, key=lambda x: x[1])
