"""alerts/smtp.py — SMTP email alert module with snapshot attachment."""
from __future__ import annotations

import smtplib
import threading
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from alerts.whatsapp import WhatsAppAlert
from utils.logger import get_logger

logger = get_logger(__name__)


class SMTPAlerter:
    """Non-blocking SMTP alert dispatcher."""

    def __init__(
        self,
        host: str = "smtp.gmail.com",
        port: int = 587,
        username: str = "",
        password: str = "",
        recipient: str = "",
        use_tls: bool = True,
        mock_mode: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.recipient = recipient
        self.use_tls = use_tls
        self.mock_mode = mock_mode

    def send(self, alert: WhatsAppAlert) -> None:
        thread = threading.Thread(
            target=self._dispatch,
            args=(alert,),
            daemon=True,
            name=f"smtp-alert-{alert.timestamp}",
        )
        thread.start()

    def _dispatch(self, alert: WhatsAppAlert) -> None:
        msg = MIMEMultipart()
        msg["From"] = self.username
        msg["To"] = self.recipient
        msg["Subject"] = f"[ALERT] Accident Detected | Camera {alert.camera_id} | Severity {alert.severity_score:.2f}"

        body = (
            f"ACCIDENT CONFIRMED\n\n"
            f"Camera ID:   {alert.camera_id}\n"
            f"Timestamp:   {alert.timestamp}\n"
            f"Vehicle A:   {alert.vehicle_id_a}  |  Plate: {alert.plate_a}\n"
            f"Vehicle B:   {alert.vehicle_id_b}  |  Plate: {alert.plate_b}\n"
            f"Severity:    {alert.severity_score:.2f} / 1.00\n\n"
            f"Summary:\n{alert.summary}\n\n"
            f"Clip saved at: {alert.clip_path}\n"
        )
        msg.attach(MIMEText(body, "plain"))

        # Attach snapshot
        snap = Path(alert.snapshot_path)
        if snap.is_file():
            with open(snap, "rb") as fh:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(fh.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={snap.name}")
            msg.attach(part)

        if self.mock_mode:
            logger.info("[MOCK] SMTP alert", to=self.recipient, subject=msg["Subject"])
            return

        try:
            with smtplib.SMTP(self.host, self.port, timeout=20) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.username, self.recipient, msg.as_string())
            logger.info("SMTP alert sent", recipient=self.recipient)
        except Exception as exc:
            logger.error("SMTP alert failed", exc=str(exc))

    @classmethod
    def from_config(cls, cfg) -> "SMTPAlerter":
        smtp_cfg = cfg.get("alerts", "smtp")
        return cls(
            host=smtp_cfg.get("host", "smtp.gmail.com"),
            port=smtp_cfg.get("port", 587),
            username=smtp_cfg.get("username", ""),
            password=smtp_cfg.get("password", ""),
            recipient=smtp_cfg.get("recipient", ""),
            use_tls=smtp_cfg.get("use_tls", True),
            mock_mode=cfg.get("alerts", "mock_mode") or False,
        )
