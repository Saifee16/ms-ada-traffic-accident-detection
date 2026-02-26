"""main.py — Traffic Surveillance System entry point."""
from __future__ import annotations

from utils.config import Config
from utils.logger import setup_logging


def main(config_path: str = "configs/default.yaml", **overrides) -> None:
    cfg = Config.load(config_path, overrides or None)
    setup_logging(
        level=cfg.get("system", "log_level") or "INFO",
        json_output=cfg.get("system", "log_json") or False,
    )
    from utils.logger import get_logger
    logger = get_logger("main")
    from utils.device import device_info
    logger.info("Device info", **device_info())
    logger.info("Config loaded", path=config_path)


if __name__ == "__main__":
    main()
