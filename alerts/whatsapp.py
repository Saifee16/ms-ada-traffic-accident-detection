"""alerts/whatsapp.py — Meta Cloud WhatsApp Business API alert module.

Setup steps (insert real credentials):
  1. Create a Meta Developer App at https://developers.facebook.com
  2. Enable WhatsApp Business API product on the app.
  3. Under WhatsApp > API Setup: copy Phone Number ID and generate Access Token.
  4. Export env vars:
       set WA_ACCESS_TOKEN=<token>       (Windows CMD)
       $env:WA_ACCESS_TOKEN="<token>"   (PowerShell)
  5. Set phone_number_id and recipient_number in configs/default.yaml.

Do NOT commit real credentials to VCS. Use environment variables only.
"""
from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


class WhatsAppAlerter:
    """Non-blocking WhatsApp alert dispatcher."""

    def __init__(
        self,
        api_version: str = "v19.0",
        phone_number_id: str = "",
        access_token: str = "",
        recipient_number: str = "",
        mock_mode: bool = False,
    ) -> None:
        self.url = WA_BASE_URL.format(
            api_version=api_version,
            phone_number_id=phone_number_id,
        )
        self.token = access_token
        self.recipient = recipient_number
        self.mock_mode = mock_mode

    def send(self, alert: WhatsAppAlert) -> None:
        """Dispatch alert in background thread (non-blocking)."""
        thread = threading.Thread(
            target=self._dispatch,
            args=(alert,),
            daemon=True,
            name=f"wa-alert-{alert.timestamp}",
        )
        thread.start()

    def _dispatch(self, alert: WhatsAppAlert) -> None:
        message_body = self._build_message(alert)
        if self.mock_mode:
            logger.info("[MOCK] WhatsApp alert", body=message_body)
            return

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": self.recipient,
            "type": "text",
            "text": {"body": message_body},
        }
        try:
            resp = requests.post(self.url, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            logger.info("WhatsApp alert sent", status=resp.status_code, recipient=self.recipient)
        except requests.RequestException as exc:
            logger.error("WhatsApp alert failed", exc=str(exc))

        # Send snapshot image if available
        if Path(alert.snapshot_path).is_file():
            self._send_image(alert.snapshot_path, headers)

    def _send_image(self, image_path: str, headers: dict) -> None:
        """Upload image as a link (requires a publicly accessible URL in production)."""
        # In production: upload to S3/CDN and send URL.
        # For local testing: log the path.
        logger.info(
            "Snapshot available for manual review (upload to CDN for WA image message)",
            path=image_path,
        )

    @staticmethod
    def _build_message(alert: WhatsAppAlert) -> str:
        return (
            f"🚨 ACCIDENT DETECTED\n"
            f"Camera: {alert.camera_id}\n"
            f"Time: {alert.timestamp}\n"
            f"Vehicle A: {alert.vehicle_id_a} | Plate: {alert.plate_a}\n"
            f"Vehicle B: {alert.vehicle_id_b} | Plate: {alert.plate_b}\n"
            f"Severity: {alert.severity_score:.2f}/1.00\n"
            f"Summary: {alert.summary}\n"
            f"Clip: {alert.clip_path}"
        )

    @classmethod
    def from_config(cls, cfg) -> "WhatsAppAlerter":
        wa_cfg = cfg.get("alerts", "whatsapp")
        return cls(
            api_version=wa_cfg.get("api_version", "v19.0"),
            phone_number_id=wa_cfg.get("phone_number_id", ""),
            access_token=wa_cfg.get("access_token", ""),
            recipient_number=wa_cfg.get("recipient_number", ""),
            mock_mode=cfg.get("alerts", "mock_mode") or False,
        )
