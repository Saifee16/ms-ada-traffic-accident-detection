"""tests/test_detector.py — Detector wrapper unit tests with mocked YOLO model."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_detector():
    """YOLODetector with mocked model to avoid downloading weights."""
    with patch("detectors.yolo_wrapper.YOLO") as MockYOLO:
        # Build a mock result that mimics ultralytics Results
        mock_boxes = MagicMock()
        import torch
        mock_boxes.xyxy = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        mock_boxes.conf = torch.tensor([0.85])
        mock_boxes.cls = torch.tensor([2.0])  # car

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        mock_model_instance = MagicMock()
        mock_model_instance.return_value = [mock_result]
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)

        MockYOLO.return_value = mock_model_instance

        from detectors.yolo_wrapper import YOLODetector
        detector = YOLODetector.__new__(YOLODetector)
        detector.model = mock_model_instance
        detector.conf = 0.35
        detector.iou = 0.45
        detector.classes = [0, 1, 2, 3, 5, 7]
        detector.half = False
        import torch
        detector.device = torch.device("cpu")

        yield detector


def test_infer_returns_detection(mock_detector, dummy_frame):
    result = mock_detector.infer(dummy_frame, frame_id=1)
    assert len(result.detections) == 1
    d = result.detections[0]
    assert d.class_id == 2
    assert d.class_name == "car"
    assert d.confidence == pytest.approx(0.85)
    assert result.frame_id == 1


def test_infer_bbox_shape(mock_detector, dummy_frame):
    result = mock_detector.infer(dummy_frame)
    assert result.detections[0].bbox.shape == (4,)


def test_infer_empty_frame(mock_detector):
    """Detector should handle tiny/empty frames without crashing."""
    mock_detector.model.return_value[0].boxes = None
    tiny = np.zeros((1, 1, 3), dtype=np.uint8)
    result = mock_detector.infer(tiny)
    assert result.detections == []
