"""Threaded video processing pipeline for one camera."""
from __future__ import annotations

import datetime as _dt
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from accident.fusion import AccidentEvent, AccidentFusion
from accident.hit_and_run import HitAndRunMonitor
from accident.scene import AccidentSceneDetector, AccidentSceneEvent, SceneChangeDetector
from alpr.alpr_pipeline import ALPRPipeline
from alerts.smtp import SMTPAlerter
from alerts.whatsapp import WhatsAppAlert, WhatsAppAlerter
from counting.counter import CountWindow, SlidingWindowCounter
from detectors.yolo_wrapper import PlateDetector, YOLODetector
from storage.csv_writer import CSVWriter
from storage.media_saver import MediaSaver
from trackers.tracker_manager import TrackedObject, TrackerManager
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

_STOP_SENTINEL = None


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
    scene_events: List[AccidentSceneEvent] = field(default_factory=list)
    timestamp: float = 0.0


class VideoPipeline:
    def __init__(
        self,
        source: str,
        cfg: Config,
        camera_id: str = "cam_01",
        output_path: Optional[str] = None,
        display: bool = False,
        max_frames: Optional[int] = None,
    ) -> None:
        self.source = source
        self.cfg = cfg
        self.camera_id = camera_id
        self.display = display
        self.max_frames = max_frames
        self._debug_fusion = bool(cfg.get("accident", "debug_mode") or False)
        self._source_is_file = self._is_file_source(source)
        self._drop_frames_when_full = bool(
            cfg.get("video", "drop_frames_when_full", default=not self._source_is_file)
        )
        self._fusion_async_enabled = bool(cfg.get("accident", "async_enabled", default=False))
        self._frame_skip = max(1, int(cfg.get("video", "frame_skip") or 1))
        self._resize_w = int(cfg.get("video", "resize_width") or cfg.get("video", "input_size") or 640)
        self._output_path = output_path if cfg.get("output", "save_video", default=True) else None
        self._out_writer: Optional[cv2.VideoWriter] = None
        self._stop_event = threading.Event()
        self._frame_q: queue.Queue = queue.Queue(maxsize=int(cfg.get("video", "buffer_size") or 64))
        self._result_q: queue.Queue = queue.Queue(maxsize=int(cfg.get("video", "buffer_size") or 64))
        self._display_q: queue.Queue = queue.Queue(maxsize=4)
        self._last_fps = 0.0
        self._vehicle_ids: Dict[int, str] = {}

        self._processed_fps = self._probe_processed_fps()
        self.detector = self._make_detector()
        self._yolo_model = self.detector.model
        self.plate_detector = self._make_plate_detector()
        self.tracker = self._make_tracker()
        self.alpr = self._make_alpr()
        self.csv_writer = self._make_csv_writer()
        self.media = self._make_media_saver()
        self.fusion = self._make_accident_fusion()
        self.hit_run_monitor = self._make_hit_run_monitor()
        self.scene_cut = self._make_scene_change_detector()
        self.scene_detector = self._make_accident_scene_detector()
        self._active_scene_events: List[AccidentSceneEvent] = []
        self.counter = SlidingWindowCounter(
            window_seconds=cfg.get("counting", "window_seconds") or 30.0,
            camera_id=camera_id,
            on_window_close=self._write_count_window,
        )
        alerts_enabled = bool(cfg.get("alerts", "enabled"))
        self.wa = WhatsAppAlerter.from_config(cfg) if alerts_enabled else None
        self.smtp = SMTPAlerter.from_config(cfg) if alerts_enabled else None
        self._alpr_interval = int(cfg.get("alpr", "frame_interval") or 15)
        self._alpr_last_frame: Dict[int, int] = {}

    @staticmethod
    def _is_file_source(source: str) -> bool:
        try:
            return Path(str(source)).exists()
        except OSError:
            return False

    def _probe_processed_fps(self) -> float:
        cap = cv2.VideoCapture(self.source)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        return max(1.0, float(src_fps) / max(1, self._frame_skip))

    def _make_detector(self) -> YOLODetector:
        det_cfg = self.cfg["detector"]
        return YOLODetector(
            model_path=det_cfg.get("model", "yolo11n.pt"),
            conf_threshold=det_cfg.get("conf_threshold", det_cfg.get("confidence_threshold", 0.4)),
            iou_threshold=det_cfg.get("iou_threshold", 0.5),
            classes=det_cfg.get("classes"),
            device_preference=self.cfg.get("system", "device") or "auto",
            half=det_cfg.get("half", False),
        )

    def _make_plate_detector(self):
        plate_cfg = self.cfg["plate_detector"]
        model_path = plate_cfg.get("model", "")
        if not plate_cfg.get("enabled", True) or not Path(model_path).exists():
            return None
        return PlateDetector(
            model_path=model_path,
            conf_threshold=plate_cfg.get("conf_threshold", 0.4),
            device_preference=self.cfg.get("system", "device") or "auto",
        )

    def _make_tracker(self) -> TrackerManager:
        trk_cfg = self.cfg["tracker"]
        return TrackerManager(
            tracker_type=trk_cfg.get("type", "bytetrack"),
            reid_gap_seconds=trk_cfg.get("reid_gap_seconds", 10.0),
            fps=self._processed_fps,
            pixels_per_meter=self.cfg.get("cameras", "default", "pixels_per_meter") or 8.0,
            speed_window=self.cfg.get("speed", "smoothing_window") or 5,
            cluster_radius_px=trk_cfg.get("cluster_radius_px", 50.0),
        )

    def _make_alpr(self) -> ALPRPipeline:
        alpr_cfg = self.cfg["alpr"]
        return ALPRPipeline(
            min_confidence=alpr_cfg.get("min_ocr_confidence", alpr_cfg.get("min_confidence", 0.45)),
            run_id=self.camera_id,
            fallback_prefix=self.cfg.get("plate_detector", "fallback_id_prefix", default="VEHICLE-ID"),
            ocr_engine=alpr_cfg.get("ocr_engine", "easyocr"),
            languages=alpr_cfg.get("languages", ["en"]),
            autoload_reader=bool(alpr_cfg.get("enabled", True)),
        )

    def _make_csv_writer(self) -> CSVWriter:
        store_cfg = self.cfg["storage"]
        return CSVWriter(
            path=store_cfg.get("csv_path", "output/detections.csv"),
            accidents_path=store_cfg.get("accidents_csv_path", "output/accidents.csv"),
            counts_path=store_cfg.get("counts_csv_path", "output/counts.csv"),
            debug_jsonl_path=store_cfg.get("debug_jsonl_path", "output/debug_events.jsonl"),
        )

    def _make_media_saver(self) -> MediaSaver:
        store_cfg = self.cfg["storage"]
        ev_cfg = self.cfg["evidence"]
        return MediaSaver(
            snapshots_dir=store_cfg.get("snapshots_dir", "output/snapshots"),
            clips_dir=store_cfg.get("clips_dir", "output/clips"),
            evidence_dir=store_cfg.get("evidence_dir", "output/evidence"),
            clip_duration_seconds=store_cfg.get("clip_duration_seconds", 5.0),
            fps=self._processed_fps,
            pre_event_seconds=ev_cfg.get("pre_event_seconds", 3.0),
            post_event_seconds=ev_cfg.get("post_event_seconds", 2.0),
            save_snapshot_enabled=ev_cfg.get("save_snapshot", True),
            save_clip_enabled=ev_cfg.get("save_clip", True),
        )

    def _make_accident_fusion(self) -> AccidentFusion:
        acc = self.cfg["accident"]
        return AccidentFusion(
            fps=self._processed_fps,
            confirmation_seconds=acc.get("confirmation_seconds", 1.5),
            min_confirming_frames=acc.get("min_confirming_frames"),
            min_signals_required=acc.get("min_signals_required", 3),
            severity_threshold=acc.get("severity_threshold", acc.get("min_severity", 0.45)),
            candidate_threshold=acc.get("candidate_threshold", 0.35),
            confirmed_threshold=acc.get("confirmed_threshold", acc.get("severity_threshold", 0.55)),
            cooldown_seconds=acc.get("cooldown_seconds", 30.0),
            proximity_threshold_px=acc.get("proximity_threshold_px", acc.get("proximity_px", 120)),
            iou_spike_threshold=acc.get("iou_spike_threshold", 0.08),
            min_contact_iou=acc.get("min_contact_iou", 0.05),
            contact_edge_px_abs=acc.get("contact_edge_px_abs", 6),
            velocity_drop_threshold=acc.get("velocity_drop_threshold", 0.30),
            min_relative_speed=acc.get("min_relative_speed_for_impact", 0.16),
            optical_flow_threshold=acc.get("optical_flow_threshold", acc.get("flow_spike_threshold", 0.12)),
            trajectory_prediction_horizon=acc.get("trajectory_prediction_horizon", 8),
            prediction_error_threshold=acc.get("prediction_error_threshold", 0.35),
            post_impact_seconds=acc.get("post_impact_seconds", 0.6),
            pre_window_seconds=acc.get("pre_window_seconds", 1.0),
            moving_threshold=acc.get("moving_threshold", 0.012),
            min_peak_speed_for_collision=acc.get("min_peak_speed_for_collision", 6.0),
            stationary_speed_px_s=acc.get("stationary_speed_px_s", 4.0),
            stationary_window_frames=acc.get("stationary_window_frames", 15),
            parked_min_seconds=acc.get("parked_min_seconds", 2.0),
            require_post_impact_validation=acc.get("require_post_impact_validation", True),
            require_hard_impact_signal=acc.get("require_hard_impact_signal", True),
            min_hard_impact_signals=acc.get("min_hard_impact_signals", 1),
            min_supporting_impact_signals=acc.get("min_supporting_impact_signals", 1),
            suppress_same_direction_traffic=acc.get("suppress_same_direction_traffic", True),
            suppress_static_static=acc.get("suppress_static_static", True),
            suppress_normal_passing_parked=acc.get("suppress_normal_passing_parked", True),
            closing_distance_enabled=acc.get("closing_distance_enabled", True),
            closing_distance_min_drop_px=acc.get("closing_distance_min_drop_px", 12.0),
            closing_distance_min_drop_ratio=acc.get("closing_distance_min_drop_ratio", 0.12),
            max_candidate_gap_frames=acc.get("max_candidate_gap_frames", 12),
            state_grace_frames=acc.get("state_grace_frames"),
            pair_grid_cell_px=acc.get("pair_grid_cell_px"),
            max_pair_distance_px=acc.get("max_pair_distance_px"),
            max_pairs_per_frame=acc.get("max_pairs_per_frame", 80),
            debug_record_normal_interval=acc.get("debug_record_normal_interval", 30),
            min_bbox_height_ratio=acc.get("min_bbox_height_ratio", 0.015),
            normalized_speed_enabled=acc.get("normalized_speed_enabled", True),
            road_roi_enabled=acc.get("road_roi_enabled", False),
            pair_cooldown_frames=acc.get("pair_cooldown_frames"),
            candidate_persistence_frames=acc.get("candidate_persistence_frames"),
            confirmation_window_frames=acc.get("confirmation_window_frames"),
            confirmation_score_threshold=acc.get("confirmation_score_threshold"),
            muted_track_frames=acc.get("muted_track_frames", 20),
            muted_base_radius_px=acc.get("muted_base_radius_px", 60.0),
            muted_velocity_radius_scale=acc.get("muted_velocity_radius_scale", 0.75),
            muted_aspect_ratio_delta=acc.get("muted_aspect_ratio_delta", 0.35),
            muted_area_ratio_delta=acc.get("muted_area_ratio_delta", 0.60),
            async_queue_size=acc.get("async_queue_size", 4),
            confirmed_display_seconds=acc.get("confirmed_display_seconds", 3.0),
            same_direction_cosine=acc.get("same_direction_cosine", 0.88),
            high_precision_mode=acc.get("high_precision_mode", True),
            min_track_age_frames=acc.get("min_track_age_frames", 5),
            min_track_history_frames=acc.get("min_track_history_frames"),
            fp_suppression_frames=acc.get("false_positive_suppression_frames", 2),
            grazing_iou_limit=acc.get("grazing_iou_limit", 0.08),
            approach_angle_deg=acc.get("approach_angle_deg", 20.0),
            debug_events=acc.get("debug_events", False) or self.cfg.get("output", "save_debug_jsonl", default=False),
            debug_events_path=self.cfg.get("storage", "debug_jsonl_path", default="output/debug_events.jsonl"),
            severity_weights=acc.get("severity_weights"),
        )

    def _make_hit_run_monitor(self) -> HitAndRunMonitor:
        acc = self.cfg["accident"]
        return HitAndRunMonitor(
            frame_width=self._resize_w,
            frame_height=480,
            flee_speed_px_s=float(acc.get("flee_speed_px_s") or 12.0),
            incapacitated_speed_px_s=float(acc.get("incapacitated_speed_px_s") or 4.0),
        )

    def _make_scene_change_detector(self) -> SceneChangeDetector:
        acc = self.cfg["accident"]
        return SceneChangeDetector(
            enabled=acc.get("scene_cut_reset_enabled", True),
            threshold=acc.get("scene_cut_threshold", 0.45),
            consecutive_frames=acc.get("scene_cut_consecutive_frames", 1),
        )

    def _make_accident_scene_detector(self) -> AccidentSceneDetector:
        acc = self.cfg["accident"]
        return AccidentSceneDetector(
            enabled=acc.get("accident_scene_detection_enabled", False),
            heuristic_enabled=acc.get("accident_scene_heuristic_enabled", True),
            persistence_frames=acc.get("accident_scene_persistence_frames", 45),
            min_stopped_vehicles=acc.get("accident_scene_min_stopped_vehicles", 2),
            min_bbox_height_ratio=acc.get("min_bbox_height_ratio", 0.015),
            stopped_speed_px_s=acc.get("stationary_speed_px_s", 4.0),
            cluster_radius_px=acc.get("accident_scene_cluster_radius_px", 150),
            score_threshold=acc.get("accident_scene_score_threshold", 0.68),
            region_cooldown_frames=acc.get("region_cooldown_frames", 300),
            display_frames=acc.get("accident_scene_display_frames", 90),
        )

    def run(self) -> None:
        prod = threading.Thread(target=self._producer, name="producer", daemon=True)
        work = threading.Thread(target=self._worker, name="worker", daemon=True)
        rend = threading.Thread(target=self._renderer, name="renderer", daemon=True)
        for thread in (prod, work, rend):
            thread.start()

        win_name = f"Camera {self.camera_id}"
        if self.display:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        try:
            while True:
                if self.display:
                    try:
                        frame = self._display_q.get_nowait()
                        if frame is _STOP_SENTINEL:
                            break
                        cv2.imshow(win_name, frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
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
        finally:
            self._stop_event.set()
            if hasattr(self.fusion, "stop_async"):
                self.fusion.stop_async()
            if self.display:
                cv2.destroyAllWindows()
            prod.join(timeout=5)
            work.join(timeout=5)
            rend.join(timeout=5)
            if self._out_writer:
                self._out_writer.release()
            self.counter.force_close()
            logger.info("Pipeline shut down", camera=self.camera_id)

    def _producer(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error("Cannot open video source", source=self.source)
            self._frame_q.put(_STOP_SENTINEL)
            return
        frame_id = 0
        processed_frames = 0
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id % self._frame_skip != 0:
                continue
            h, w = frame.shape[:2]
            if w != self._resize_w:
                scale = self._resize_w / float(w)
                frame = cv2.resize(frame, (self._resize_w, int(h * scale)))
            if self._out_writer is None and self._output_path:
                Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)
                h2, w2 = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._out_writer = cv2.VideoWriter(self._output_path, fourcc, self._processed_fps, (w2, h2))
            self.media.push_frame(frame_id, frame)
            if self._enqueue_frame(FramePacket(frame_id, frame, time.time())):
                processed_frames += 1
                if self.max_frames is not None and processed_frames >= self.max_frames:
                    break
        cap.release()
        self._enqueue_stop()

    def _enqueue_frame(self, packet: FramePacket) -> bool:
        if not self._drop_frames_when_full:
            while not self._stop_event.is_set():
                try:
                    self._frame_q.put(packet, timeout=0.25)
                    return True
                except queue.Full:
                    continue
            return False

        try:
            self._frame_q.put_nowait(packet)
            return True
        except queue.Full:
            try:
                dropped = self._frame_q.get_nowait()
                dropped_id = getattr(dropped, "frame_id", None)
            except queue.Empty:
                dropped_id = None
            try:
                self._frame_q.put_nowait(packet)
                logger.warning(
                    "Frame queue full; dropped oldest frame",
                    dropped_frame_id=dropped_id,
                    frame_id=packet.frame_id,
                )
                return True
            except queue.Full:
                logger.warning("Frame queue full; dropping current frame", frame_id=packet.frame_id)
                return False

    def _enqueue_stop(self) -> None:
        if not self._drop_frames_when_full:
            while True:
                try:
                    self._frame_q.put(_STOP_SENTINEL, timeout=0.25)
                    return
                except queue.Full:
                    if self._stop_event.is_set():
                        return

        while True:
            try:
                self._frame_q.put_nowait(_STOP_SENTINEL)
                return
            except queue.Full:
                try:
                    self._frame_q.get_nowait()
                except queue.Empty:
                    return

    def _worker(self) -> None:
        tracker_cfg = TrackerManager.ultralytics_tracker_cfg(self.cfg.get("tracker", "type") or "bytetrack")
        while not self._stop_event.is_set():
            try:
                packet = self._frame_q.get(timeout=1)
            except queue.Empty:
                continue
            if packet is _STOP_SENTINEL:
                if self._fusion_async_enabled:
                    self.fusion.stop_async(timeout=5.0)
                    for result in self.fusion.drain_results():
                        for event in result.events:
                            self.hit_run_monitor.register_accident(event)
                            self._handle_accident(result.frame, event, result.tracked_objects)
                break
            started = time.perf_counter()
            frame = packet.frame
            frame_id = packet.frame_id
            scene_cut = self.scene_cut.update(frame)
            if scene_cut.is_cut:
                self._handle_scene_cut(frame_id, scene_cut.score)
            results = self._yolo_model.track(
                frame,
                conf=self.cfg.get("detector", "conf_threshold") or self.cfg.get("detector", "confidence_threshold") or 0.4,
                iou=self.cfg.get("detector", "iou_threshold") or 0.5,
                classes=self.cfg.get("detector", "classes"),
                tracker=tracker_cfg,
                persist=not scene_cut.is_cut,
                verbose=False,
            )
            tracked_objects = self._parse_tracks(results)
            self._finalize_exited_tracks()
            self._process_alpr_and_counts(frame, frame_id, tracked_objects)
            scene_events = self.scene_detector.process_frame(frame, tracked_objects, frame_id)
            active_scene_events = scene_events + self.scene_detector.get_active_events(frame_id)
            self._active_scene_events = active_scene_events

            accident_events: List[AccidentEvent] = []
            if self._fusion_async_enabled:
                self.fusion.submit_frame(
                    frame,
                    tracked_objects,
                    frame_id,
                    debug=self._debug_fusion,
                    drop_oldest=self._drop_frames_when_full,
                )
                for result in self.fusion.drain_results():
                    accident_events.extend(result.events)
                    for event in result.events:
                        self.hit_run_monitor.register_accident(event)
                        self._handle_accident(result.frame, event, result.tracked_objects)
            else:
                accident_events = self.fusion.process_frame(frame, tracked_objects, frame_id, debug=self._debug_fusion)
                for event in accident_events:
                    self.hit_run_monitor.register_accident(event)
                    self._handle_accident(frame, event, tracked_objects)
            self.hit_run_monitor.process_frame(tracked_objects, frame_id)

            elapsed = time.perf_counter() - started
            self._last_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            annotated = self._annotate(frame, tracked_objects, accident_events, active_scene_events)
            self._enqueue_result(ResultPacket(frame_id, annotated, accident_events, active_scene_events, packet.timestamp))
        self._result_q.put(_STOP_SENTINEL)

    def _handle_scene_cut(self, frame_id: int, score: float) -> None:
        logger.info("Scene cut detected; resetting accident state", frame_id=frame_id, score=round(score, 3))
        self.fusion.reset_state()
        self.scene_detector.reset()
        self._active_scene_events = []
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
        if hasattr(self.hit_run_monitor, "reset"):
            self.hit_run_monitor.reset()

    def _enqueue_result(self, packet: ResultPacket) -> bool:
        if not self._drop_frames_when_full:
            while not self._stop_event.is_set():
                try:
                    self._result_q.put(packet, timeout=0.25)
                    return True
                except queue.Full:
                    continue
            return False

        try:
            self._result_q.put_nowait(packet)
            return True
        except queue.Full:
            try:
                self._result_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._result_q.put_nowait(packet)
                logger.warning("Result queue full; dropped oldest rendered frame", frame_id=packet.frame_id)
                return True
            except queue.Full:
                logger.warning("Result queue full; dropping rendered frame", frame_id=packet.frame_id)
                return False

    def _parse_tracks(self, results) -> List[TrackedObject]:
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            self.tracker.update(np.array([]), np.empty((0, 4)), np.array([]), [], np.array([]))
            return []
        boxes = results[0].boxes
        tids = boxes.id.cpu().numpy().astype(int)
        bboxes = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        names = [self.detector.model.names[int(c)] for c in cls_ids]
        return self.tracker.update(tids, bboxes, cls_ids, names, confs)

    def _finalize_exited_tracks(self) -> None:
        if not self.cfg.get("alpr", "retry_until_exit", default=True):
            return
        for obj in self.tracker.pop_exited_tracks():
            result = self.alpr.finalize(obj.track_id)
            self._vehicle_ids[obj.track_id] = result.plate_text

    def _process_alpr_and_counts(self, frame: np.ndarray, frame_id: int, tracked_objects: List[TrackedObject]) -> None:
        for obj in tracked_objects:
            self.counter.update(obj.track_id, obj.class_name)
            best = self.alpr.get_best_current(obj.track_id)
            if best:
                obj.plate_text, obj.ocr_confidence = best
                obj.vehicle_identifier = best[0]
                self._vehicle_ids[obj.track_id] = best[0]
            if self.cfg.get("alpr", "enabled", default=True):
                high_conf = self.cfg.get("alpr", "high_confidence_skip", default=0.75)
                if not best or best[1] < high_conf:
                    last = self._alpr_last_frame.get(obj.track_id, -self._alpr_interval)
                    if frame_id - last >= self._alpr_interval:
                        self._alpr_last_frame[obj.track_id] = frame_id
                        self._run_ocr(frame, obj)
            if self.cfg.get("output", "save_csv", default=True) and frame_id % 25 == 0:
                self._write_detection_row(obj, frame_id)

    def _run_ocr(self, frame: np.ndarray, obj: TrackedObject) -> None:
        x1, y1, x2, y2 = map(int, obj.bbox)
        crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        if crop.size == 0:
            return
        plate_bboxes = None
        if self.plate_detector:
            plate_bboxes = [p.bbox for p in self.plate_detector.detect_plates(crop)]
        result = self.alpr.process_vehicle_crop(obj.track_id, crop, plate_bboxes)
        if result:
            obj.plate_text, obj.ocr_confidence = result
            obj.vehicle_identifier = result[0]
            self._vehicle_ids[obj.track_id] = result[0]

    def _write_detection_row(self, obj: TrackedObject, frame_id: int) -> None:
        heading = obj.velocity_vector / max(float(np.linalg.norm(obj.velocity_vector)), 1e-6)
        ident = obj.vehicle_identifier or obj.plate_text or self._vehicle_ids.get(obj.track_id, "")
        self.csv_writer.write_detection(
            frame_number=frame_id,
            camera_id=self.camera_id,
            track_id=obj.track_id,
            vehicle_identifier=ident,
            class_name=obj.class_name,
            detection_confidence=obj.detection_confidence,
            plate_text=obj.plate_text,
            ocr_confidence=obj.ocr_confidence,
            fallback_used=obj.fallback_used,
            speed_px_per_s=obj.speed_px_s,
            heading_x=float(heading[0]),
            heading_y=float(heading[1]),
        )

    def _handle_accident(self, frame: np.ndarray, event: AccidentEvent, tracked_objects: List[TrackedObject]) -> None:
        result_a = self.alpr.get_identifier(event.track_id_a, finalize=True)
        result_b = self.alpr.get_identifier(event.track_id_b, finalize=True)
        ident_a = result_a.plate_text
        ident_b = result_b.plate_text
        self._vehicle_ids[event.track_id_a] = ident_a
        self._vehicle_ids[event.track_id_b] = ident_b

        timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat()
        metadata = {
            "camera_id": self.camera_id,
            "timestamp": timestamp,
            "involved_vehicles": [
                {"track_id": event.track_id_a, "identifier": ident_a, "fallback_used": result_a.is_fallback},
                {"track_id": event.track_id_b, "identifier": ident_b, "fallback_used": result_b.is_fallback},
            ],
            "signal_scores": event.signal_scores,
            "signals": event.signals_triggered,
            "severity": event.severity_score,
            "alert_status": {"queued": False},
        }
        snap_path, clip_path, metadata_path = self.media.save_event_evidence(
            event_id=event.event_id,
            frame=frame,
            frame_id=event.frame_id,
            metadata=metadata,
        )

        self.csv_writer.write_accident(
            camera_id=self.camera_id,
            track_id_a=event.track_id_a,
            track_id_b=event.track_id_b,
            plate_a=ident_a,
            plate_b=ident_b,
            iou=event.iou_at_collision,
            motion_score=event.motion_anomaly_score,
            snapshot_path=snap_path,
            clip_path=clip_path,
            event_id=event.event_id,
            frame_number=event.frame_id,
            severity_score=event.severity_score,
            signal_scores=event.signal_scores,
        )
        self.csv_writer.write_accident_event(
            camera_id=self.camera_id,
            event_id=event.event_id,
            frame_number=event.frame_id,
            track_id_a=event.track_id_a,
            track_id_b=event.track_id_b,
            vehicle_identifier_a=ident_a,
            vehicle_identifier_b=ident_b,
            severity_score=event.severity_score,
            signals=event.signals_triggered,
            iou_at_collision=event.iou_at_collision,
            signal_scores=event.signal_scores,
            snapshot_path=snap_path,
            clip_path=clip_path,
            timestamp=timestamp,
        )

        alert = WhatsAppAlert(
            event_id=event.event_id,
            vehicle_id_a=str(event.track_id_a),
            vehicle_id_b=str(event.track_id_b),
            vehicle_identifier_a=ident_a,
            vehicle_identifier_b=ident_b,
            plate_a="" if result_a.is_fallback else ident_a,
            plate_b="" if result_b.is_fallback else ident_b,
            timestamp=timestamp,
            camera_id=self.camera_id,
            severity_score=event.severity_score,
            snapshot_path=snap_path,
            clip_path=clip_path,
            summary=f"Signals: {', '.join(event.signals_triggered)}. Metadata: {metadata_path}",
        )
        alert_status = {"queued": True, "whatsapp": bool(self.wa), "smtp": bool(self.smtp)}
        if self.wa:
            self.wa.send(alert)
        if self.smtp:
            self.smtp.send(alert)
        self.media.update_metadata(event.event_id, {"alert_status": alert_status})

    def _write_count_window(self, window: CountWindow) -> None:
        if not self.cfg.get("output", "save_csv", default=True):
            return
        self.csv_writer.write_count_window(window.camera_id, window.window_start, window.window_end, window.counts)

    def _annotate(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        accidents: List[AccidentEvent],
        scene_events: Optional[List[AccidentSceneEvent]] = None,
    ) -> np.ndarray:
        annotated = frame.copy()
        scene_events = scene_events if scene_events is not None else getattr(self, "_active_scene_events", [])
        candidates = self.fusion.get_active_candidates()
        accident_pairs = {
            (min(evt.track_id_a, evt.track_id_b), max(evt.track_id_a, evt.track_id_b)): evt
            for evt in accidents
        }
        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj.bbox)
            color = (255, 200, 80)
            pair_state = ""
            event_label = ""
            candidate_threshold = 0.45
            if hasattr(self, "cfg"):
                candidate_threshold = self.cfg.get("accident", "candidate_threshold", default=0.45) or 0.45
            for key, data in candidates.items():
                if obj.track_id in key:
                    pair_state = data["state"]
                    if pair_state == "CONFIRMED":
                        color = (0, 255, 0)
                        event_label = f"ACCIDENT DETECTED {data.get('event_id', '')}"
                    elif pair_state == "CANDIDATE":
                        severity = float(data.get("last", {}).get("severity", 0.0) or 0.0)
                        if severity >= candidate_threshold:
                            color = (0, 255, 255)
                            event_label = f"CANDIDATE {data.get('event_id', '')}"
                        else:
                            pair_state = ""
                    elif pair_state == "SUSPECT":
                        pair_state = ""
            for key, evt in accident_pairs.items():
                if obj.track_id in key:
                    pair_state = "CONFIRMED"
                    event_label = f"ACCIDENT DETECTED {evt.event_id} sev={evt.severity_score:.2f}"
                    color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            ident = obj.vehicle_identifier or obj.plate_text or self._vehicle_ids.get(obj.track_id) or f"T{obj.track_id}"
            label = f"{ident} | {obj.class_name} | {obj.speed_px_s:.0f}px/s"
            cv2.putText(annotated, label, (x1, max(14, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
            if event_label:
                cv2.putText(annotated, event_label, (x1, min(frame.shape[0] - 8, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
        for evt in accidents:
            obj_a = self.tracker.get_track(evt.track_id_a)
            obj_b = self.tracker.get_track(evt.track_id_b)
            if obj_a is not None and obj_b is not None:
                cv2.line(annotated, tuple(obj_a.centroid.astype(int)), tuple(obj_b.centroid.astype(int)), (0, 255, 0), 3)
                cv2.putText(
                    annotated,
                    f"ACCIDENT DETECTED {evt.event_id} severity={evt.severity_score:.2f}",
                    (20, min(frame.shape[0] - 20, 60)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
        for scene_event in scene_events:
            x1, y1, x2, y2 = scene_event.bbox
            color = (0, 128, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"{scene_event.label} {scene_event.event_id} score={scene_event.score:.2f}"
            cv2.putText(
                annotated,
                label,
                (max(10, x1), max(22, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
            )
        cv2.putText(annotated, f"FPS {self._last_fps:.1f}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return annotated

    def _renderer(self) -> None:
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
                    pass
        if self.display:
            try:
                self._display_q.put(_STOP_SENTINEL, timeout=2)
            except queue.Full:
                pass
