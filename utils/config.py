"""utils/config.py — YAML config loader with env-variable override."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class Config:
    """Singleton-like config bag loaded once from YAML."""

    _instance: "Config | None" = None

    def __init__(self, path: str | Path = "configs/default.yaml", overrides: dict | None = None):
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        if overrides:
            data = _deep_merge(data, overrides)
        # Inject env secrets
        wa = data.get("alerts", {}).get("whatsapp", {})
        if token := os.getenv("WA_ACCESS_TOKEN"):
            wa["access_token"] = token
        smtp = data.get("alerts", {}).get("smtp", {})
        if pwd := os.getenv("SMTP_PASSWORD"):
            smtp["password"] = pwd
        self._data = data

    # ── attribute-style deep access ──────────────────────────
    def get(self, *keys: str, default: Any = None) -> Any:
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
            if node is default:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    @classmethod
    def load(cls, path: str | Path = "configs/default.yaml", overrides: dict | None = None) -> "Config":
        cls._instance = cls(path, overrides)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "Config":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
