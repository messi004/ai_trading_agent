"""Structured JSON logging with correlation IDs.

Usage:
    from core.logger import get_logger
    log = get_logger(__name__)
    log.info("tick_processed", extra={"spot": 24005.5, "correlation_id": cid})
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from typing import Any

_JSON_FORMAT = "%(message)s"
_TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"

_RESERVED = {
    "message",
    "asctime",
    "created",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "module",
    "msecs",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Emit a single-line JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """Attach a process-wide correlation id to every log record."""

    def __init__(self) -> None:
        super().__init__()
        self.correlation_id: str = ""

    def set_correlation_id(self, cid: str) -> None:
        self.correlation_id = cid

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = getattr(record, "correlation_id", None) or self.correlation_id
        return True


def new_correlation_id() -> str:
    return uuid.uuid4().hex


_shared_filter = ContextFilter()


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure the root logger once. Idempotent."""
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_shared_filter)
    handler.setFormatter(JsonFormatter() if json_output else logging.Formatter(_TEXT_FORMAT))
    root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def set_correlation_id(cid: str) -> None:
    _shared_filter.set_correlation_id(cid)


def get_logger(name: str) -> logging.Logger:
    """Get a logger (assumes setup_logging already called)."""
    return logging.getLogger(name)
