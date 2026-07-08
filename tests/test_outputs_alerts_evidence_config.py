from __future__ import annotations

import csv
import json
import time

import numpy as np

from accident.fusion import AccidentEvent
from alerts.whatsapp import WhatsAppAlert, WhatsAppAlerter
from pipelines.video_pipeline import VideoPipeline
from storage.csv_writer import CSVWriter, DETECTION_FIELDNAMES
from storage.media_saver import MediaSaver
from trackers.tracker_manager import TrackedObject
from utils.config import Config


def test_csv_writer_includes_required_columns(tmp_path):
    writer = CSVWriter(
        path=str(tmp_path / "detections.csv"),
        accidents_path=str(tmp_path / "accidents.csv"),
        counts_path=str(tmp_path / "counts.csv"),
        debug_jsonl_path=str(tmp_path / "debug_events.jsonl"),
    )
    writer.write_detection(camera_id="CAM01", track_id=1, class_name="car", frame_number=12)
    with open(tmp_path / "detections.csv", newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == DETECTION_FIELDNAMES
    for field in [
        "vehicle_identifier",
        "fallback_used",
        "collision_partner_id",
        "trajectory_conflict_score",
        "velocity_drop_score",
        "optical_flow_anomaly_score",
    ]:
        assert field in header


def test_alert_mock_receives_correct_payload():
    alerter = WhatsAppAlerter(mock_mode=True, recipient_number="TEST_RECIPIENT")
    alert = WhatsAppAlert(
        event_id="ACC-000001-1-2-001",
        vehicle_id_a="1",
        vehicle_id_b="2",
        vehicle_identifier_a="ABC 1234",
        vehicle_identifier_b="VEHICLE-ID-0001",
        plate_a="ABC 1234",
        plate_b="",
        timestamp="2026-01-01T00:00:00Z",
        camera_id="CAM01",
        severity_score=0.72,
        snapshot_path="output/evidence/ACC/snapshot.jpg",
        clip_path="output/evidence/ACC/clip.mp4",
        summary="unit test",
    )
    alerter.send(alert)
    deadline = time.time() + 2
    while not alerter.mock_payloads and time.time() < deadline:
        time.sleep(0.01)
    assert alerter.mock_payloads
    payload = alerter.mock_payloads[0]["payload"]
    assert payload["to"] == "TEST_RECIPIENT"
    assert "ACC-000001-1-2-001" in payload["text"]["body"]
    assert "ABC 1234" in payload["text"]["body"]


def test_evidence_metadata_created(tmp_path):
    saver = MediaSaver(evidence_dir=str(tmp_path / "evidence"), fps=5.0, pre_event_seconds=1.0, post_event_seconds=1.0)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    for fid in range(5):
        saver.push_frame(fid, frame)
    snap, clip, metadata = saver.save_event_evidence(
        event_id="ACC-TEST",
        frame=frame,
        frame_id=5,
        metadata={
            "camera_id": "CAM01",
            "timestamp": "2026-01-01T00:00:00Z",
            "involved_vehicles": [{"track_id": 1}, {"track_id": 2}],
            "signal_scores": {"proximity_score": 0.8},
            "severity": 0.7,
            "alert_status": {"queued": False},
        },
    )
    assert snap.endswith("snapshot.jpg")
    assert clip.endswith("clip.mp4")
    data = json.loads(open(metadata, encoding="utf-8").read())
    assert data["event_id"] == "ACC-TEST"
    assert data["camera_id"] == "CAM01"
    assert data["frame_start"] <= data["frame_end"]


def test_default_config_loads_new_parameters():
    cfg = Config.load("configs/default.yaml")
    assert cfg.validate() == []
    assert cfg.get("system", "device") == "auto"
    assert cfg.get("accident", "confirmation_seconds") == 1.2
    assert cfg.get("accident", "candidate_threshold") == 0.45
    assert cfg.get("accident", "confirmed_threshold") == 0.60
    assert cfg.get("video", "frame_skip") == 1
    assert cfg.get("video", "drop_frames_when_full") is False
    assert cfg.get("accident", "async_enabled") is False
    assert cfg.get("accident", "scene_cut_reset_enabled") is True
    assert cfg.get("accident", "min_track_history_frames") == 5
    assert cfg.get("accident", "confirmation_score_threshold") == 0.60
    assert cfg.get("accident", "normalized_speed_enabled") is True
    assert cfg.get("accident", "accident_scene_detection_enabled") is False
    assert cfg.get("output", "save_debug_jsonl") is True


def test_candidate_yellow_confirmed_green_overlay_state():
    class FakeFusion:
        def __init__(self, state):
            self.state = state

        def get_active_candidates(self):
            return {
                (1, 2): {
                    "state": self.state,
                    "event_id": "ACC-TEST",
                    "confirming_frames": 5,
                    "last": {"severity": 0.7},
                }
            }

    class FakeTracker:
        def get_track(self, track_id):
            return None

    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    obj = TrackedObject(
        track_id=1,
        class_id=2,
        class_name="car",
        bbox=np.array([20, 20, 80, 70], dtype=float),
        centroid=np.array([50, 45], dtype=float),
    )
    pipe = object.__new__(VideoPipeline)
    pipe._vehicle_ids = {}
    pipe._last_fps = 0.0
    pipe.tracker = FakeTracker()

    pipe.fusion = FakeFusion("CANDIDATE")
    candidate = VideoPipeline._annotate(pipe, frame, [obj], [])
    assert tuple(candidate[20, 20].tolist()) == (0, 255, 255)

    pipe.fusion = FakeFusion("SUSPECT")
    suspect = VideoPipeline._annotate(pipe, frame, [obj], [])
    assert tuple(suspect[20, 20].tolist()) != (0, 255, 255)

    pipe.fusion = FakeFusion("CONFIRMED")
    confirmed = VideoPipeline._annotate(pipe, frame, [obj], [])
    assert tuple(confirmed[20, 20].tolist()) == (0, 255, 0)

    accident = AccidentEvent(
        track_id_a=1,
        track_id_b=2,
        timestamp=0.0,
        frame_id=10,
        severity_score=0.8,
        iou_at_collision=0.1,
        motion_anomaly_score=0.5,
        confirmed=True,
        event_id="ACC-TEST",
    )
    confirmed_now = VideoPipeline._annotate(pipe, frame, [obj], [accident])
    assert tuple(confirmed_now[20, 20].tolist()) == (0, 255, 0)
