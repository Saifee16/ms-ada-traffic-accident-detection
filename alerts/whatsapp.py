"""Meta Cloud WhatsApp alert dispatch."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

WA_BASE_URL = "https://graph.facebook.com/{api_version}/{phone_number_id}/messages"


@dataclass
class WhatsAppAlert:
    vehicle_id_a: str
    vehicle_id_b: str
    plate_a: str
    plate_b: str
    timestamp: str
    camera_id: str
    severity_score: float
    snapshot_path: str
    clip_path: str
    summary: str
    event_id: str = ""
    vehicle_identifier_a: str = ""
    vehicle_identifier_b: str = ""

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class WhatsAppAlerter:
    def __init__(
        self,
        api_version: str = "v19.0",
        phone_number_id: str = "",
        access_token: str = "",
        recipient_number: str = "",
        mock_mode: bool = False,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        dedupe_cooldown_seconds: float = 30.0,
    ) -> None:
        self.url = WA_BASE_URL.format(api_version=api_version, phone_number_id=phone_number_id)
        self.token = access_token
        self.recipient = recipient_number
        self.mock_mode = bool(mock_mode)
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_backoff_seconds = max(0.1, float(retry_backoff_seconds))
        self.dedupe_cooldown_seconds = float(dedupe_cooldown_seconds)
        self._sent: Dict[str, float] = {}
        self.mock_payloads: List[Dict[str, object]] = []
        self._lock = threading.Lock()

    def send(self, alert: WhatsAppAlert) -> None:
        if self._is_duplicate(alert):
            logger.info("WhatsApp alert deduplicated", event_id=alert.event_id)
            return
        threading.Thread(target=self._dispatch, args=(alert,), daemon=True, name=f"wa-alert-{alert.event_id}").start()

    def _is_duplicate(self, alert: WhatsAppAlert) -> bool:
        key = alert.event_id or f"{alert.camera_id}:{alert.vehicle_id_a}:{alert.vehicle_id_b}"
        now = time.time()
        with self._lock:
            last = self._sent.get(key)
            if last is not None and now - last < self.dedupe_cooldown_seconds:
                return True
            self._sent[key] = now
        return False

    def _dispatch(self, alert: WhatsAppAlert) -> None:
        message_body = self._build_message(alert)
        payload = {
            "messaging_product": "whatsapp",
            "to": self.recipient,
            "type": "text",
            "text": {"body": message_body},
        }
        if self.mock_mode:
            record = {"channel": "whatsapp", "payload": payload, "alert": alert.as_dict()}
            self.mock_payloads.append(record)
            logger.info("[MOCK] WhatsApp alert", event_id=alert.event_id, body=message_body)
            return

        if not self.token or not self.recipient or "YOUR_" in self.token:
            logger.error("WhatsApp alert not sent: missing credentials or recipient", event_id=alert.event_id)
            return

        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        for attempt in range(1, self.retry_attempts + 1):
            try:
                resp = requests.post(self.url, headers=headers, json=payload, timeout=15)
                resp.raise_for_status()
                logger.info("WhatsApp alert sent", event_id=alert.event_id, status=resp.status_code)
                self._log_media_upload_status(alert)
                return
            except requests.RequestException as exc:
                logger.warning("WhatsApp alert attempt failed", event_id=alert.event_id, attempt=attempt, exc=str(exc))
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_backoff_seconds * (2 ** (attempt - 1)))
        logger.error("WhatsApp alert failed after retries", event_id=alert.event_id)

    def _log_media_upload_status(self, alert: WhatsAppAlert) -> None:
        if Path(alert.snapshot_path).is_file():
            logger.info(
                "Snapshot is local only; configure a media uploader/CDN before sending WhatsApp image messages",
                event_id=alert.event_id,
                snapshot_path=alert.snapshot_path,
            )

    @staticmethod
    def _build_message(alert: WhatsAppAlert) -> str:
        ident_a = alert.vehicle_identifier_a or alert.plate_a or alert.vehicle_id_a
        ident_b = alert.vehicle_identifier_b or alert.plate_b or alert.vehicle_id_b
        return (
            "ACCIDENT DETECTED\n"
            f"Event: {alert.event_id}\n"
            f"Camera: {alert.camera_id}\n"
            f"Time: {alert.timestamp}\n"
            f"Vehicle A: {ident_a} (track {alert.vehicle_id_a})\n"
            f"Vehicle B: {ident_b} (track {alert.vehicle_id_b})\n"
            f"Severity: {alert.severity_score:.2f}/1.00\n"
            f"Snapshot: {alert.snapshot_path}\n"
            f"Clip: {alert.clip_path}\n"
            f"Summary: {alert.summary}"
        )
    @classmethod
    def from_config(cls, cfg) -> "WhatsAppAlerter":
        wa_cfg = cfg.get("alerts", "whatsapp") or {}
        return cls(
            api_version=wa_cfg.get("api_version", "v19.0"),
            phone_number_id=wa_cfg.get("phone_number_id", ""),
            access_token=wa_cfg.get("access_token", ""),
            recipient_number=wa_cfg.get("recipient_number", ""),
            mock_mode=cfg.get("alerts", "mock_mode") or False,
            retry_attempts=cfg.get("alerts", "retry_attempts", default=3) or 3,
            retry_backoff_seconds=cfg.get("alerts", "retry_backoff_seconds", default=1.0) or 1.0,
            dedupe_cooldown_seconds=cfg.get("alerts", "dedupe_cooldown_seconds", default=30.0) or 30.0,
        )
