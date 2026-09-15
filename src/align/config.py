"""Validated configuration for the diagnostic command."""

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DoctorConfig:
    """Paths are resolved by the CLI; timeout is seconds per external command."""

    output_dir: Path
    timeout_seconds: float = 5.0
    require_nvidia: bool = False
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 30:
            raise ValueError("timeout must be finite and greater than 0, up to 30 seconds")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("log level must be DEBUG, INFO, WARNING, or ERROR")

    def to_dict(self) -> dict:
        """Return resolved, JSON-serializable command settings."""
        return {
            "output_dir": str(self.output_dir),
            "timeout_seconds": self.timeout_seconds,
            "require_nvidia": self.require_nvidia,
            "log_level": self.log_level,
        }
