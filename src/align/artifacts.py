"""Durable diagnostic JSON and local logging; no training checkpoint semantics."""

import json
import logging
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def create_run_directory(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    identifier = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    run_dir = output_dir / identifier
    run_dir.mkdir()
    return run_dir


def write_json_atomic(path: Path, value: dict) -> None:
    """Replace one JSON file after serialization and fsync; preserve old data on failure.

    Uses a temporary file on the same filesystem. This is not a complete training
    checkpoint protocol or a guarantee against disk/controller failure.
    """
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class EventFormatter(logging.Formatter):
    """JSONL events from the same records used by human-readable logging."""

    def format(self, record: logging.LogRecord) -> str:
        event = {
            "timestamp_utc": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "run_id": record.name.removeprefix("align.doctor."),
            "event": getattr(record, "event", "message"),
            "message": record.getMessage(),
        }
        for key in ("check_name", "check_status"):
            if hasattr(record, key):
                event[key] = getattr(record, key)
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, allow_nan=False)


@contextmanager
def diagnostic_logger(run_dir: Path, level: str) -> Iterator[logging.Logger]:
    """Own and close handlers without changing the application's root logger."""
    logger = logging.getLogger(f"align.doctor.{run_dir.name}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    readable = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    readable.converter = time.gmtime
    try:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(readable)
        logger.addHandler(console)
        for filename, formatter in (("doctor.log", readable), ("events.jsonl", EventFormatter())):
            handler = logging.FileHandler(run_dir / filename, mode="x", encoding="utf-8")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        yield logger
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
