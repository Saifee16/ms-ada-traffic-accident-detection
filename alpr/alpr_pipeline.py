"""ALPR state management, OCR normalization, and fallback vehicle IDs."""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

_PK_PATTERNS: List[re.Pattern] = [
    re.compile(r"^[A-Z]{3}\s\d{3,4}$"),
    re.compile(r"^[A-Z]{2}\s\d{3,4}$"),
    re.compile(r"^[A-Z]{1,2}\s\d{4}$"),
    re.compile(r"^RIV\s\d{2,4}$"),
    re.compile(r"^\d{4,5}$"),
]

_FALLBACK_COUNTER_LOCK = threading.Lock()
_FALLBACK_COUNTER: Dict[str, int] = {}


def _fix_plate_confusions(text: str) -> str:
    """Fix common OCR character confusions based on alpha/digit context."""
    compact = re.sub(r"\s+", "", text.upper())
    if not compact:
        return ""
    chars = list(compact)
    first_digit = next((i for i, ch in enumerate(chars) if ch.isdigit()), len(chars))
    if first_digit > 3 and chars[first_digit - 1] in {"I", "L"}:
        chars[first_digit - 1] = "1"
        first_digit -= 1
    alpha_map = {"0": "O", "1": "I", "5": "S", "8": "B"}
    digit_map = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8"}
    for idx, ch in enumerate(chars):
        if idx < first_digit:
            chars[idx] = alpha_map.get(ch, ch)
        else:
            chars[idx] = digit_map.get(ch, ch)
    return "".join(chars)


def normalize_plate(raw: str) -> str:
    if not raw:
        return ""
    cleaned = raw.upper().strip()
    cleaned = re.sub(r"[-_.:/|]", " ", cleaned)
    cleaned = re.sub(r"[^A-Z0-9 ]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    compact = _fix_plate_confusions(cleaned)
    compact = re.sub(r"([A-Z]+)(\d+)", r"\1 \2", compact)
    compact = re.sub(r"(\d+)([A-Z]+)", r"\1 \2", compact)
    normalized = re.sub(r"\s+", " ", compact).strip()
    return normalized


def is_valid_plate(text: str) -> bool:
    return any(pattern.match(text) for pattern in _PK_PATTERNS)


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
    """Accumulates OCR readings per track and finalizes stable identifiers."""

    def __init__(
        self,
        min_confidence: float = 0.45,
        run_id: str = "default",
        fallback_prefix: str = "VEHICLE-ID",
        ocr_engine: str = "easyocr",
        languages: Optional[List[str]] = None,
        autoload_reader: bool = True,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.run_id = run_id
        self.fallback_prefix = fallback_prefix
        self._readings: Dict[int, List[Tuple[str, float]]] = {}
        self._finalized: Dict[int, ALPRResult] = {}
        self._best: Dict[int, Tuple[str, float]] = {}
        self._ocr_engine = ocr_engine
        self._languages = languages or ["en"]
        self._reader = None
        self._reader_lock = threading.Lock()
        self._reader_ready = threading.Event()
        if autoload_reader:
            threading.Thread(target=self._prewarm_reader, daemon=True, name="ocr-prewarm").start()
        else:
            self._reader_ready.set()

    def _prewarm_reader(self) -> None:
        try:
            if self._ocr_engine == "easyocr":
                import easyocr

                reader = easyocr.Reader(self._languages, gpu=self._is_gpu_available(), verbose=False)
            elif self._ocr_engine == "paddleocr":
                from paddleocr import PaddleOCR

                reader = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=self._is_gpu_available())
            else:
                reader = None
            with self._reader_lock:
                self._reader = reader
            if reader is not None:
                logger.info("OCR reader initialized", engine=self._ocr_engine)
        except Exception as exc:
            logger.warning("OCR pre-warm failed", exc=str(exc))
        finally:
            self._reader_ready.set()

    def _get_reader(self):
        with self._reader_lock:
            if self._reader is not None:
                return self._reader
        self._reader_ready.wait(timeout=0.25)
        with self._reader_lock:
            return self._reader

    @staticmethod
    def _is_gpu_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def submit_reading(self, track_id: int, raw_text: str, confidence: float) -> Optional[Tuple[str, float]]:
        text = normalize_plate(raw_text)
        confidence = float(confidence)
        if not text or confidence < self.min_confidence:
            return None
        validity_bonus = 0.08 if is_valid_plate(text) else 0.0
        score = min(1.0, confidence + validity_bonus)
        self._readings.setdefault(track_id, []).append((text, score))
        current = self._best.get(track_id)
        if current is None or score > current[1]:
            self._best[track_id] = (text, score)
        return self._best[track_id]

    def process_vehicle_crop(
        self,
        track_id: int,
        vehicle_crop: np.ndarray,
        plate_bboxes: Optional[List[np.ndarray]] = None,
    ) -> Optional[Tuple[str, float]]:
        if track_id in self._finalized:
            result = self._finalized[track_id]
            return result.plate_text, result.confidence

        reader = self._get_reader()
        if reader is None:
            return self.get_best_current(track_id)

        regions: List[np.ndarray] = []
        if plate_bboxes:
            for pb in plate_bboxes:
                x1, y1, x2, y2 = map(int, pb)
                crop = vehicle_crop[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size:
                    regions.append(crop)
        else:
            h = vehicle_crop.shape[0]
            regions.append(vehicle_crop[h // 2 :, :])

        for region in regions:
            if region.size == 0:
                continue
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            try:
                if self._ocr_engine == "easyocr":
                    for _, text, prob in reader.readtext(thresh, detail=1):
                        best = self.submit_reading(track_id, text, float(prob))
                        if best:
                            return best
                elif self._ocr_engine == "paddleocr":
                    results = reader.ocr(thresh, cls=True)
                    for line in results[0] if results and results[0] else []:
                        text, prob = line[1][0], line[1][1]
                        best = self.submit_reading(track_id, text, float(prob))
                        if best:
                            return best
            except Exception as exc:
                logger.warning("OCR error", exc=str(exc), track_id=track_id)

        return self.get_best_current(track_id)

    def finalize(self, track_id: int) -> ALPRResult:
        if track_id in self._finalized:
            return self._finalized[track_id]

        readings = self._readings.pop(track_id, [])
        best = self._best.pop(track_id, None)
        if best is None and readings:
            best = max(readings, key=lambda item: item[1])
        if best:
            result = ALPRResult(
                plate_text=best[0],
                confidence=best[1],
                is_fallback=False,
                raw_readings=readings,
            )
        else:
            result = ALPRResult(
                plate_text=fallback_plate_id(self.run_id, self.fallback_prefix),
                confidence=0.0,
                is_fallback=True,
                raw_readings=readings,
            )
        self._finalized[track_id] = result
        logger.debug("ALPR finalized", track_id=track_id, plate=result.plate_text, conf=result.confidence)
        return result

    def get_best_current(self, track_id: int) -> Optional[Tuple[str, float]]:
        return self._best.get(track_id)

    def get_identifier(self, track_id: int, finalize: bool = False) -> ALPRResult:
        if track_id in self._finalized:
            return self._finalized[track_id]
        best = self.get_best_current(track_id)
        if best and not finalize:
            return ALPRResult(plate_text=best[0], confidence=best[1], is_fallback=False)
        return self.finalize(track_id)
