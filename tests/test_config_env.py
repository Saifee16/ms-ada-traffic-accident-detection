from __future__ import annotations

from utils.config import Config


def test_alert_settings_can_be_overridden_from_environment(monkeypatch):
    monkeypatch.setenv("WA_ACCESS_TOKEN", "wa-token")
    monkeypatch.setenv("WA_PHONE_NUMBER_ID", "phone-id")
    monkeypatch.setenv("WA_RECIPIENT_NUMBER", "923001234567")
    monkeypatch.setenv("SMTP_USERNAME", "sender@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "smtp-secret")
    monkeypatch.setenv("SMTP_RECIPIENT", "receiver@example.com")

    cfg = Config.load("configs/default.yaml")

    assert cfg.get("alerts", "whatsapp", "access_token") == "wa-token"
    assert cfg.get("alerts", "whatsapp", "phone_number_id") == "phone-id"
    assert cfg.get("alerts", "whatsapp", "recipient_number") == "923001234567"
    assert cfg.get("alerts", "smtp", "username") == "sender@example.com"
    assert cfg.get("alerts", "smtp", "password") == "smtp-secret"
    assert cfg.get("alerts", "smtp", "recipient") == "receiver@example.com"
