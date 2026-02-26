"""detectors/yolo_wrapper.py — YOLOv11 detector wrapper with CPU/GPU auto-switch.

Choice rationale:
  YOLOv11n (nano) selected for CPU-first deployment — ~6ms inference at 640px.
  Ultralytics 8.3.x provides a unified API for inference, ONNX export, and fine-tuning.
  TorchScript/ONNX export path is baked in for future GPU acceleration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils.device import resolve_device
from utils.logger import get_logger

logger = get_logger(__name__)

COCO_VEHICLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    bbox: np.ndarray          # [x1, y1, x2, y2] float32
    confidence: float
    class_id: int
    class_name: str
    frame_id: int = 0


@dataclass
class InferenceResult:
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    frame_id: int = 0


class YOLODetector:
    """Thread-safe YOLO inference wrapper."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        device_preference: str = "auto",
        half: bool = False,
    ) -> None:
        self.device = resolve_device(device_preference)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.classes = classes or list(COCO_VEHICLE_CLASSES.keys())
        self.half = half and self.device.type == "cuda"

        logger.info("Loading YOLO model", model=model_path, device=str(self.device))
        self.model = YOLO(model_path)
        self.model.to(self.device)
        # Warm-up pass
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        logger.info("YOLO model ready", classes=self.classes)

    # ──────────────────────────────────────────────────────────
    def infer(self, frame: np.ndarray, frame_id: int = 0, min_area_ratio: float = 0.0005) -> InferenceResult:
        t0 = time.perf_counter()
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            half=self.half,
            verbose=False,
            stream=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        frame_area = frame.shape[0] * frame.shape[1]
        min_box_area = frame_area * min_area_ratio

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = box
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < min_box_area:
                    continue  # reject tiny detections (shadows, reflections, noise)
                detections.append(
                    Detection(
                        bbox=box,
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=COCO_VEHICLE_CLASSES.get(int(cls_id), str(cls_id)),
                        frame_id=frame_id,
                    )
                )
        return InferenceResult(detections=detections, inference_ms=elapsed_ms, frame_id=frame_id)

    # ──────────────────────────────────────────────────────────
    def export_onnx(self, output_path: str = "models/yolo11n.onnx", dynamic: bool = True) -> str:
        """Export to ONNX for TensorRT / GPU deployment."""
        path = self.model.export(format="onnx", dynamic=dynamic, simplify=True)
        logger.info("ONNX export complete", path=path)
        return str(path)

    def export_torchscript(self, output_path: str = "models/yolo11n.torchscript") -> str:
        path = self.model.export(format="torchscript")
        logger.info("TorchScript export complete", path=path)
        return str(path)

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def fine_tune_recipe() -> str:
        """Return the CLI command for fine-tuning on custom dataset."""
        return (
            "yolo detect train "
            "model=yolo11n.pt "
            "data=datasets/traffic/data.yaml "
            "epochs=100 "
            "imgsz=640 "
            "batch=16 "
            "lr0=0.01 "
            "augment=True "
            "project=models/finetune "
            "name=traffic_v1"
        )


# ── Plate-specific detector (same class, different weights) ──
class PlateDetector(YOLODetector):
    """YOLO fine-tuned on license plate bounding boxes."""

    def __init__(self, model_path: str = "models/plate_detector.pt", **kwargs) -> None:
        # classes=None → detect everything (single class: plate)
        super().__init__(model_path=model_path, classes=None, **kwargs)

    def detect_plates(self, vehicle_crop: np.ndarray) -> List[Detection]:
        result = self.infer(vehicle_crop)
        return result.detections
