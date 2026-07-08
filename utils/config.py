"""YAML config loader with environment-variable overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env(data: dict) -> None:
    alerts = data.setdefault("alerts", {})
    whatsapp = alerts.setdefault("whatsapp", {})
    smtp = alerts.setdefault("smtp", {})

    env_map = {
        "WA_ACCESS_TOKEN": (whatsapp, "access_token"),
        "WA_PHONE_NUMBER_ID": (whatsapp, "phone_number_id"),
        "WA_RECIPIENT_NUMBER": (whatsapp, "recipient_number"),
        "SMTP_USERNAME": (smtp, "username"),
        "SMTP_PASSWORD": (smtp, "password"),
        "SMTP_RECIPIENT": (smtp, "recipient"),
    }
    for env_name, (section, key) in env_map.items():
        value = os.getenv(env_name)
        if value:
            section[key] = value


class Config:
    """Singleton-like config bag loaded once from YAML."""

    _instance: "Config | None" = None

    def __init__(self, path: str | Path = "configs/default.yaml", overrides: dict | None = None):
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if overrides:
            data = _deep_merge(data, overrides)
        _apply_env(data)
        self._data = data

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self._data
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
            if node is default:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def as_dict(self) -> dict:
        return dict(self._data)

    def validate(self) -> list[str]:
        """Return a list of schema/threshold errors. Empty means valid."""
        errors: list[str] = []
        required_paths = [
            ("system", "device"),
            ("video", "input_size"),
            ("video", "drop_frames_when_full"),
            ("detector", "confidence_threshold"),
            ("detector", "iou_threshold"),
            ("tracker", "max_age"),
            ("tracker", "min_hits"),
            ("tracker", "reid_enabled"),
            ("accident", "high_precision_mode"),
            ("accident", "confirmation_seconds"),
            ("accident", "candidate_threshold"),
            ("accident", "confirmed_threshold"),
            ("accident", "min_track_history_frames"),
            ("accident", "confirmation_score_threshold"),
            ("accident", "min_signals_required"),
            ("accident", "severity_threshold"),
            ("accident", "cooldown_seconds"),
            ("accident", "region_cooldown_frames"),
            ("accident", "max_candidate_gap_frames"),
            ("accident", "state_grace_frames"),
            ("accident", "pair_grid_cell_px"),
            ("accident", "max_pair_distance_px"),
            ("accident", "max_pairs_per_frame"),
            ("accident", "debug_record_normal_interval"),
            ("accident", "async_enabled"),
            ("accident", "async_queue_size"),
            ("accident", "scene_cut_reset_enabled"),
            ("accident", "scene_cut_threshold"),
            ("accident", "scene_cut_consecutive_frames"),
            ("accident", "muted_track_frames"),
            ("accident", "muted_base_radius_px"),
            ("accident", "muted_velocity_radius_scale"),
            ("accident", "muted_aspect_ratio_delta"),
            ("accident", "muted_area_ratio_delta"),
            ("accident", "stationary_speed_px_s"),
            ("accident", "stationary_window_frames"),
            ("accident", "parked_min_seconds"),
            ("accident", "min_bbox_height_ratio"),
            ("accident", "normalized_speed_enabled"),
            ("accident", "road_roi_enabled"),
            ("accident", "debug_candidates"),
            ("accident", "accident_scene_detection_enabled"),
            ("accident", "accident_scene_heuristic_enabled"),
            ("accident", "accident_scene_persistence_frames"),
            ("accident", "accident_scene_min_stopped_vehicles"),
            ("accident", "accident_scene_cluster_radius_px"),
            ("accident", "accident_scene_score_threshold"),
            ("accident", "accident_scene_display_frames"),
            ("accident", "require_post_impact_validation"),
            ("accident", "require_hard_impact_signal"),
            ("accident", "min_hard_impact_signals"),
            ("accident", "min_supporting_impact_signals"),
            ("accident", "suppress_same_direction_traffic"),
            ("accident", "suppress_static_static"),
            ("accident", "suppress_normal_passing_parked"),
            ("accident", "closing_distance_enabled"),
            ("accident", "closing_distance_min_drop_px"),
            ("accident", "closing_distance_min_drop_ratio"),
            ("accident", "candidate_overlay_min_state"),
            ("accident", "confirmed_display_seconds"),
            ("accident", "proximity_threshold_px"),
            ("accident", "iou_spike_threshold"),
            ("accident", "velocity_drop_threshold"),
            ("accident", "optical_flow_threshold"),
            ("accident", "trajectory_prediction_horizon"),
            ("accident", "debug_events"),
            ("alpr", "enabled"),
            ("alpr", "min_ocr_confidence"),
            ("alpr", "retry_until_exit"),
            ("alpr", "fallback_format"),
            ("counting", "window_seconds"),
            ("alerts", "enabled"),
            ("alerts", "mock_mode"),
            ("alerts", "whatsapp"),
            ("alerts", "smtp"),
            ("evidence", "save_snapshot"),
            ("evidence", "save_clip"),
            ("evidence", "pre_event_seconds"),
            ("evidence", "post_event_seconds"),
            ("output", "save_video"),
            ("output", "save_csv"),
            ("output", "save_debug_jsonl"),
        ]
        for path in required_paths:
            if self.get(*path, default=None) is None:
                errors.append("Missing config key: " + ".".join(path))

        numeric_positive = [
            ("video", "input_size"),
            ("accident", "confirmation_seconds"),
            ("accident", "candidate_threshold"),
            ("accident", "confirmed_threshold"),
            ("accident", "min_track_history_frames"),
            ("accident", "confirmation_score_threshold"),
            ("accident", "min_signals_required"),
            ("accident", "cooldown_seconds"),
            ("accident", "region_cooldown_frames"),
            ("accident", "proximity_threshold_px"),
            ("accident", "max_candidate_gap_frames"),
            ("accident", "state_grace_frames"),
            ("accident", "pair_grid_cell_px"),
            ("accident", "max_pairs_per_frame"),
            ("accident", "debug_record_normal_interval"),
            ("accident", "async_queue_size"),
            ("accident", "scene_cut_threshold"),
            ("accident", "scene_cut_consecutive_frames"),
            ("accident", "muted_track_frames"),
            ("accident", "muted_base_radius_px"),
            ("accident", "muted_velocity_radius_scale"),
            ("accident", "muted_aspect_ratio_delta"),
            ("accident", "muted_area_ratio_delta"),
            ("accident", "stationary_speed_px_s"),
            ("accident", "stationary_window_frames"),
            ("accident", "parked_min_seconds"),
            ("accident", "min_bbox_height_ratio"),
            ("accident", "accident_scene_persistence_frames"),
            ("accident", "accident_scene_min_stopped_vehicles"),
            ("accident", "accident_scene_cluster_radius_px"),
            ("accident", "accident_scene_score_threshold"),
            ("accident", "accident_scene_display_frames"),
            ("accident", "confirmed_display_seconds"),
            ("accident", "min_hard_impact_signals"),
            ("accident", "closing_distance_min_drop_px"),
            ("accident", "closing_distance_min_drop_ratio"),
            ("counting", "window_seconds"),
        ]
        for path in numeric_positive:
            value = self.get(*path, default=0)
            try:
                if float(value) <= 0:
                    errors.append("Config key must be positive: " + ".".join(path))
            except (TypeError, ValueError):
                errors.append("Config key must be numeric: " + ".".join(path))
        return errors

    @classmethod
    def load(cls, path: str | Path = "configs/default.yaml", overrides: dict | None = None) -> "Config":
        cls._instance = cls(path, overrides)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "Config":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
