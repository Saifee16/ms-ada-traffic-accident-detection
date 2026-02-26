"""pipelines/video_pipeline.py — Async producer/consumer video ingestion pipeline.

Architecture:
  Producer thread  → reads frames from VideoCapture, pushes to frame_queue
  Worker thread    → pops frames, runs detect+track+ALPR+accident, pushes results
  Renderer thread  → draws overlays, writes output video, displays (optional)
  Alert dispatcher → non-blocking, spawned per confirmed accident

Bounded queues prevent memory blow-up on slow consumers.
"""
from __future__ import annotations

import datetime
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from accident.fusion import AccidentEvent, AccidentFusion
from alpr.alpr_pipeline import ALPRPipeline
from alerts.smtp import SMTPAlerter
from alerts.whatsapp import WhatsAppAlert, WhatsAppAlerter
from counting.counter import SlidingWindowCounter
from detectors.yolo_wrapper import YOLODetector, PlateDetector
from storage.csv_writer import CSVWriter
from storage.media_saver import MediaSaver
from trackers.tracker_manager import TrackerManager
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

_STOP_SENTINEL = None  # pushed to queues on shutdown


@dataclass
class FramePacket:
    frame_id: int
    frame: np.ndarray
    timestamp: float


@dataclass
class ResultPacket:
    frame_id: int
    annotated_frame: np.ndarray
    accident_events: List[AccidentEvent] = field(default_factory=list)
    timestamp: float = 0.0


class VideoPipeline:
    """
    Full processing pipeline for a single video source.
    """

    def __init__(
        self,
        source: str,
        cfg: Config,
        camera_id: str = "cam_01",
        output_path: Optional[str] = None,
        display: bool = False,
    ) -> None:
        self.source = source
        self.cfg = cfg
        self.camera_id = camera_id
        self.display = display

        # Queues
        buf = cfg.get("video", "buffer_size") or 64
        self._frame_q: queue.Queue = queue.Queue(maxsize=buf)
        self._result_q: queue.Queue = queue.Queue(maxsize=buf)

        # Components
        dev = cfg.get("system", "device") or "auto"
        det_cfg = cfg["detector"]
        self.detector = YOLODetector(
            model_path=det_cfg.get("model", "yolo11n.pt"),
            conf_threshold=det_cfg.get("conf_threshold", 0.35),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            classes=det_cfg.get("classes"),
            device_preference=dev,
        )
        # Use ultralytics .track() with bytetrack config
        self._yolo_model = self.detector.model

        plate_cfg = cfg["plate_detector"]
        self.plate_detector = PlateDetector(
            model_path=plate_cfg.get("model", "models/plate_detector.pt"),
            device_preference=dev,
        ) if Path(plate_cfg.get("model", "")).exists() else None

        trk_cfg = cfg["tracker"]
        self.tracker = TrackerManager(
            tracker_type=trk_cfg.get("type", "bytetrack"),
            reid_gap_seconds=trk_cfg.get("reid_gap_seconds", 10.0),
            fps=cfg.get("video", "output_fps") or 25.0,
            pixels_per_meter=cfg.get("cameras", "default", "pixels_per_meter") or 8.0,
        )

        alpr_cfg = cfg["alpr"]
        self.alpr = ALPRPipeline(
            min_confidence=alpr_cfg.get("min_confidence", 0.45),
            run_id=camera_id,
            fallback_prefix=cfg.get("plate_detector", "fallback_id_prefix", default="VEHICLE-ID"),
            ocr_engine=alpr_cfg.get("ocr_engine", "easyocr"),
            languages=alpr_cfg.get("languages", ["en"]),
        )

        # Probe source FPS before creating fusion so confirm_frames is calculated correctly.
        # Processed FPS = source_fps / frame_skip (we only evaluate every Nth frame).
        _cap_probe = cv2.VideoCapture(source)
        _src_fps = _cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
        _cap_probe.release()
        _frame_skip = cfg.get("video", "frame_skip") or 2
        _processed_fps = _src_fps / max(1, _frame_skip)

        acc_cfg = cfg["accident"]
        fusion = AccidentFusion(
            fps=_processed_fps,
            confirmation_seconds=acc_cfg.get("confirmation_seconds", 0.8),
            iou_spike_threshold=acc_cfg.get("iou_spike_threshold", 0.20),
            speed_drop_ratio=acc_cfg.get("speed_drop_ratio", 0.5),
            optical_flow_zscore=acc_cfg.get("optical_flow_anomaly_zscore", 2.5),
            proximity_px=acc_cfg.get("proximity_px", 80.0),
            min_signals=acc_cfg.get("min_signals_required", 2),
            severity_weights=acc_cfg.get("severity_weights"),
            fp_suppression_frames=acc_cfg.get("false_positive_suppression_frames", 3),
            min_track_age_frames=acc_cfg.get("min_track_age_frames", 8),
            min_approach_speed_px=acc_cfg.get("min_approach_speed_px", 2.0),
            cooldown_seconds=acc_cfg.get("cooldown_seconds", 30.0),
        )
        fusion.min_severity = float(acc_cfg.get("min_severity") or 0.15)
        self.fusion = fusion

        cnt_cfg = cfg["counting"]
        self.counter = SlidingWindowCounter(
            window_seconds=cnt_cfg.get("window_seconds", 30.0),
            camera_id=camera_id,
        )

        store_cfg = cfg["storage"]
        self.csv_writer = CSVWriter(path=store_cfg.get("csv_path", "output/detections.csv"))
        self.media = MediaSaver(
            snapshots_dir=store_cfg.get("snapshots_dir", "output/snapshots"),
            clips_dir=store_cfg.get("clips_dir", "output/clips"),
            clip_duration_seconds=store_cfg.get("clip_duration_seconds", 5.0),
        )

        alerts_cfg = cfg["alerts"]
        self._debug_fusion = False   # set via pipeline._debug_fusion = True before run()
        self.wa = WhatsAppAlerter.from_config(cfg) if alerts_cfg.get("enabled") else None
        self.smtp = SMTPAlerter.from_config(cfg) if alerts_cfg.get("enabled") else None

        # Output video writer (initialized in producer once resolution known)
        self._out_writer: Optional[cv2.VideoWriter] = None
        self._output_path = output_path
        self._frame_skip = cfg.get("video", "frame_skip") or 1
        self._resize_w = cfg.get("video", "resize_width") or 640
        self._stop_event = threading.Event()
        # ALPR throttle: only attempt OCR every N *processed* frames per track
        # EasyOCR on CPU = ~400ms/call; 15 frames ≈ every 1s of real video at 30fps+skip2
        self._alpr_interval: int = int(cfg.get("alpr", "frame_interval") or 15)
        self._alpr_last_frame: dict = {}   # track_id → last frame_id OCR was run
        # Display queue: renderer pushes frames, main thread calls imshow (Windows-safe)
        self._display_q: queue.Queue = queue.Queue(maxsize=4)

    # ──────────────────────────────────────────────────────────
    def run(self) -> None:
        prod = threading.Thread(target=self._producer, name="producer", daemon=True)
        work = threading.Thread(target=self._worker,   name="worker",   daemon=True)
        rend = threading.Thread(target=self._renderer, name="renderer", daemon=True)

        for t in (prod, work, rend):
            t.start()

        win_name = f"Camera {self.camera_id}"
        if self.display:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                # Main-thread display loop (Windows requires imshow on main thread)
                if self.display:
                    try:
                        disp_frame = self._display_q.get_nowait()
                        if disp_frame is _STOP_SENTINEL:
                            break
                        cv2.imshow(win_name, disp_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            self._stop_event.set()
                            break
                    except queue.Empty:
                        cv2.waitKey(1)
                        if not prod.is_alive() and not work.is_alive() and not rend.is_alive():
                            break
                else:
                    prod.join()
                    work.join()
                    rend.join()
                    break
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping pipeline")
            self._stop_event.set()
        finally:
            if self.display:
                cv2.destroyAllWindows()
            prod.join(timeout=5)
            work.join(timeout=5)
            rend.join(timeout=5)
            if self._out_writer:
                self._out_writer.release()
            self.counter.force_close()
            logger.info("Pipeline shut down", camera=self.camera_id)

    # ──────────────────────────────────────────────────────────
    def _producer(self) -> None:
        """Read frames from source and push to frame queue."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error("Cannot open video source", source=self.source)
            self._frame_q.put(_STOP_SENTINEL)
            return

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Video opened", source=self.source, fps=fps_src, frames=total_frames)

        frame_id = 0
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if frame_id % self._frame_skip != 0:
                continue

            # Resize for inference
            h, w = frame.shape[:2]
            if w != self._resize_w:
                scale = self._resize_w / w
                frame = cv2.resize(frame, (self._resize_w, int(h * scale)))

            # Init output writer on first frame
            if self._out_writer is None and self._output_path:
                h2, w2 = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_fps = self.cfg.get("video", "output_fps") or 25.0
                self._out_writer = cv2.VideoWriter(self._output_path, fourcc, out_fps, (w2, h2))

            self.media.push_frame(frame_id, frame)
            try:
                self._frame_q.put(FramePacket(frame_id=frame_id, frame=frame, timestamp=time.time()), timeout=2)
            except queue.Full:
                logger.warning("Frame queue full — dropping frame", frame_id=frame_id)

        cap.release()
        self._frame_q.put(_STOP_SENTINEL)
        logger.info("Producer finished", camera=self.camera_id)

    # ──────────────────────────────────────────────────────────
    def _worker(self) -> None:
        """Pop frames, run detect → track → ALPR → accident fusion."""
        tracker_cfg = TrackerManager.ultralytics_tracker_cfg(
            self.cfg.get("tracker", "type") or "bytetrack"
        )

        while not self._stop_event.is_set():
            try:
                packet = self._frame_q.get(timeout=1)
            except queue.Empty:
                continue
            if packet is _STOP_SENTINEL:
                break

            frame = packet.frame
            frame_id = packet.frame_id

            # Track via Ultralytics (returns Results with track IDs)
            results = self._yolo_model.track(
                frame,
                conf=self.cfg.get("detector", "conf_threshold") or 0.35,
                classes=self.cfg.get("detector", "classes"),
                tracker=tracker_cfg,
                persist=True,
                verbose=False,
            )

            # Parse tracking results
            tracked_objects = []
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                tids = boxes.id.cpu().numpy().astype(int)
                bboxes = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                cls_names = [self.detector.model.names[c] for c in cls_ids]

                tracked_objects = self.tracker.update(tids, bboxes, cls_ids, cls_names)

                # ALPR per vehicle — throttled to once every alpr_interval frames
                # Prevents EasyOCR (400ms/call on CPU) from blocking the worker thread
                for obj in tracked_objects:
                    # Always update counter (cheap)
                    self.counter.update(obj.track_id, obj.class_name)

                    # Reuse cached plate text if already identified with high confidence
                    cached = self.alpr.get_best_current(obj.track_id)
                    if cached and cached[1] >= 0.75:
                        obj.plate_text, obj.ocr_confidence = cached
                        continue  # good enough — skip OCR this frame

                    # Rate-gate: only run OCR every alpr_interval processed frames
                    last = self._alpr_last_frame.get(obj.track_id, -self._alpr_interval)
                    if frame_id - last < self._alpr_interval:
                        if cached:
                            obj.plate_text, obj.ocr_confidence = cached
                        continue

                    self._alpr_last_frame[obj.track_id] = frame_id
                    x1, y1, x2, y2 = map(int, obj.bbox)
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size == 0:
                        continue
                    plate_bboxes = None
                    if self.plate_detector:
                        plates = self.plate_detector.detect_plates(crop)
                        plate_bboxes = [p.bbox for p in plates]
                    ocr_result = self.alpr.process_vehicle_crop(obj.track_id, crop, plate_bboxes)
                    if ocr_result:
                        obj.plate_text, obj.ocr_confidence = ocr_result

                    # CSV row per active track (sampled — could be every N frames)
                    if frame_id % 25 == 0:
                        best = self.alpr.get_best_current(obj.track_id)
                        plate_t, plate_c = best if best else ("", 0.0)
                        self.csv_writer.write_detection(
                            camera_id=self.camera_id,
                            track_id=obj.track_id,
                            class_name=obj.class_name,
                            plate_text=plate_t,
                            ocr_confidence=plate_c,
                            speed_px_per_s=obj.speed_px_s,
                        )

            # Accident fusion
            if self._debug_fusion and frame_id % 50 == 0:
                print(f"[WORKER] frame={frame_id} tracked={len(tracked_objects)} vehicles={[o.track_id for o in tracked_objects]}", flush=True)
            accident_events = self.fusion.process_frame(frame, tracked_objects, frame_id,
                                                         debug=self._debug_fusion)
            for event in accident_events:
                self._handle_accident(frame, event, tracked_objects)

            # Annotate frame
            annotated = self._annotate(frame, results, tracked_objects, accident_events)
            try:
                self._result_q.put(ResultPacket(
                    frame_id=frame_id,
                    annotated_frame=annotated,
                    accident_events=accident_events,
                    timestamp=packet.timestamp,
                ), timeout=2)
            except queue.Full:
                pass

        self._result_q.put(_STOP_SENTINEL)
        logger.info("Worker finished", camera=self.camera_id)

    # ──────────────────────────────────────────────────────────
    def _handle_accident(
        self,
        frame: np.ndarray,
        event: AccidentEvent,
        tracked_objects,
    ) -> None:
        """Save evidence and dispatch alerts for a confirmed accident."""
        tag = f"{self.camera_id}_acc_{event.track_id_a}_{event.track_id_b}"
        snap_path = self.media.save_snapshot(frame, tag=tag)
        clip_path = self.media.save_clip(frame, tag=tag)

        alpr_a = self.alpr.get_best_current(event.track_id_a)
        alpr_b = self.alpr.get_best_current(event.track_id_b)
        plate_a = alpr_a[0] if alpr_a else self.alpr.finalize(event.track_id_a).plate_text
        plate_b = alpr_b[0] if alpr_b else self.alpr.finalize(event.track_id_b).plate_text

        self.csv_writer.write_accident(
            camera_id=self.camera_id,
            track_id_a=event.track_id_a,
            track_id_b=event.track_id_b,
            plate_a=plate_a,
            plate_b=plate_b,
            iou=event.iou_at_collision,
            motion_score=event.motion_anomaly_score,
            snapshot_path=snap_path,
            clip_path=clip_path,
        )

        alert = WhatsAppAlert(
            vehicle_id_a=str(event.track_id_a),
            vehicle_id_b=str(event.track_id_b),
            plate_a=plate_a,
            plate_b=plate_b,
            timestamp=datetime.datetime.utcnow().isoformat(),
            camera_id=self.camera_id,
            severity_score=event.severity_score,
            snapshot_path=snap_path,
            clip_path=clip_path,
            summary=f"Vehicles {event.track_id_a} & {event.track_id_b} collided. "
                    f"Signals: {', '.join(event.signals_triggered)}.",
        )
        if self.wa:
            self.wa.send(alert)
        if self.smtp:
            self.smtp.send(alert)

    # ──────────────────────────────────────────────────────────
    def _annotate(self, frame, yolo_results, tracked_objects, accidents):
        annotated = frame.copy()
        # Draw YOLO boxes + track IDs
        if yolo_results:
            annotated = yolo_results[0].plot(img=annotated)
        # Overlay speed + plate
        for obj in tracked_objects:
            x1, y1 = int(obj.bbox[0]), int(obj.bbox[1])
            plate_label = obj.plate_text if obj.plate_text else "..."
            spd_label = f"{obj.speed_px_s:.0f}px/s"
            cv2.putText(annotated, plate_label, (x1, max(0, y1-20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(annotated, spd_label, (x1, max(0, y1-40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        # Mark accident pairs
        for evt in accidents:
            a = self.tracker.get_track(evt.track_id_a)
            b = self.tracker.get_track(evt.track_id_b)
            if a and b:
                ca = tuple(a.centroid.astype(int))
                cb = tuple(b.centroid.astype(int))
                cv2.line(annotated, ca, cb, (0, 0, 255), 3)
                cv2.putText(annotated, f"ACCIDENT sev={evt.severity_score:.2f}",
                            ca, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return annotated

    # ──────────────────────────────────────────────────────────
    def _renderer(self) -> None:
        """Write annotated frames to output video; push to display queue for main thread."""
        while not self._stop_event.is_set():
            try:
                packet = self._result_q.get(timeout=1)
            except queue.Empty:
                continue
            if packet is _STOP_SENTINEL:
                break
            if self._out_writer:
                self._out_writer.write(packet.annotated_frame)
            if self.display:
                try:
                    self._display_q.put_nowait(packet.annotated_frame)
                except queue.Full:
                    pass  # drop display frame — never block the renderer
        # Signal main thread display loop to exit
        if self.display:
            try:
                self._display_q.put(_STOP_SENTINEL, timeout=2)
            except queue.Full:
                pass
        logger.info("Renderer finished", camera=self.camera_id)