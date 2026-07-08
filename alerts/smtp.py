"""SMTP accident alert dispatch."""
from __future__ import annotations

import smtplib
import threading
import time
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List

from alerts.whatsapp import WhatsAppAlert
from utils.logger import get_logger

logger = get_logger(__name__)


class SMTPAlerter:
    def __init__(
        self,
        host: str = "smtp.gmail.com",
        port: int = 587,
        username: str = "",
        password: str = "",
        recipient: str = "",
        use_tls: bool = True,
        mock_mode: bool = False,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        dedupe_cooldown_seconds: float = 30.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.username = username
        self.password = password
        self.recipient = recipient
        self.use_tls = bool(use_tls)
        self.mock_mode = bool(mock_mode)
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_backoff_seconds = max(0.1, float(retry_backoff_seconds))
        self.dedupe_cooldown_seconds = float(dedupe_cooldown_seconds)
        self._sent: Dict[str, float] = {}
        self.mock_payloads: List[Dict[str, object]] = []
        self._lock = threading.Lock()

    def send(self, alert: WhatsAppAlert) -> None:
        if self._is_duplicate(alert):
            logger.info("SMTP alert deduplicated", event_id=alert.event_id)
            return
        threading.Thread(target=self._dispatch, args=(alert,), daemon=True, name=f"smtp-alert-{alert.event_id}").start()

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
        msg = self._build_email(alert)
        if self.mock_mode:
            self.mock_payloads.append({"channel": "smtp", "subject": msg["Subject"], "alert": alert.as_dict()})
            logger.info("[MOCK] SMTP alert", event_id=alert.event_id, to=self.recipient, subject=msg["Subject"])
            return

        if not self.username or not self.password or not self.recipient or "YOUR_" in self.password:
            logger.error("SMTP alert not sent: missing credentials or recipient", event_id=alert.event_id)
            return

        for attempt in range(1, self.retry_attempts + 1):
            try:
                with smtplib.SMTP(self.host, self.port, timeout=20) as server:
                    if self.use_tls:
                        server.starttls()
                    server.login(self.username, self.password)
                    server.sendmail(self.username, self.recipient, msg.as_string())
                logger.info("SMTP alert sent", event_id=alert.event_id, recipient=self.recipient)
                return
            except Exception as exc:
                logger.warning("SMTP alert attempt failed", event_id=alert.event_id, attempt=attempt, exc=str(exc))
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_backoff_seconds * (2 ** (attempt - 1)))
        logger.error("SMTP alert failed after retries", event_id=alert.event_id)

    def _build_email(self, alert: WhatsAppAlert) -> MIMEMultipart:
        ident_a = alert.vehicle_identifier_a or alert.plate_a or alert.vehicle_id_a
        ident_b = alert.vehicle_identifier_b or alert.plate_b or alert.vehicle_id_b
        msg = MIMEMultipart()
        msg["From"] = self.username
        msg["To"] = self.recipient
        msg["Subject"] = f"[ALERT] Accident {alert.event_id} | {alert.camera_id} | Severity {alert.severity_score:.2f}"
        body = (
            "ACCIDENT CONFIRMED\n\n"
            f"Event ID: {alert.event_id}\n"
            f"Camera ID: {alert.camera_id}\n"
            f"Timestamp: {alert.timestamp}\n"
            f"Vehicle A: {ident_a} | Track: {alert.vehicle_id_a} | Plate: {alert.plate_a}\n"
            f"Vehicle B: {ident_b} | Track: {alert.vehicle_id_b} | Plate: {alert.plate_b}\n"
            f"Severity: {alert.severity_score:.2f} / 1.00\n"
            f"Snapshot: {alert.snapshot_path}\n"
            f"Clip: {alert.clip_path}\n\n"
            f"Summary:\n{alert.summary}\n"
        )
        msg.attach(MIMEText(body, "plain"))
        snap = Path(alert.snapshot_path)
        if snap.is_file():
            with open(snap, "rb") as fh:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(fh.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={snap.name}")
            msg.attach(part)
        return msg

    @classmethod
    def from_config(cls, cfg) -> "SMTPAlerter":
        smtp_cfg = cfg.get("alerts", "smtp") or {}
        return cls(
            host=smtp_cfg.get("host", "smtp.gmail.com"),
            port=smtp_cfg.get("port", 587),
            username=smtp_cfg.get("username", ""),
            password=smtp_cfg.get("password", ""),
            recipient=smtp_cfg.get("recipient", ""),
            use_tls=smtp_cfg.get("use_tls", True),
            mock_mode=cfg.get("alerts", "mock_mode") or False,
            retry_attempts=cfg.get("alerts", "retry_attempts", default=3) or 3,
            retry_backoff_seconds=cfg.get("alerts", "retry_backoff_seconds", default=1.0) or 1.0,
            dedupe_cooldown_seconds=cfg.get("alerts", "dedupe_cooldown_seconds", default=30.0) or 30.0,
        )
