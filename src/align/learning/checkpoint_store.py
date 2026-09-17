"""Host-safe, append-only checkpoint publication and integrity selection.

Checkpoint payloads are opaque bytes here. Verify the manifest and checksum
before passing a trusted payload to a model deserializer.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from align.artifacts import as_ist, utc_now, write_json_atomic

SCHEMA_VERSION = 1
RECOVERY_MODE = "training_resume_with_environment_reset"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_name(value: str) -> bool:
    return bool(value) and value == Path(value).name and "/" not in value and "\\" not in value


class CheckpointStore:
    """Publish immutable payload/manifest pairs and select the newest valid pair."""

    def __init__(
        self,
        directory: Path,
        *,
        run_id: str,
        config_sha256: str,
        file_mode: int = 0o600,
    ):
        if not _safe_name(run_id):
            raise ValueError("run_id must be a single path-safe name")
        if len(config_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in config_sha256
        ):
            raise ValueError("config_sha256 must be a lowercase SHA-256 hex digest")
        if file_mode not in {0o600, 0o640, 0o644}:
            raise ValueError("file_mode must be 0600, 0640, or 0644")
        self.directory = directory
        self.run_id = run_id
        self.config_sha256 = config_sha256
        self.file_mode = file_mode
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        writer: Callable,
        *,
        validator: Callable[[Path], None] | None = None,
        attempt_id: str,
        completed_updates: int,
        environment_transitions: int,
        agent_transitions: int,
        active_training_seconds: float,
        source_identity: str,
        runtime_identity: str,
        parent_checkpoint_sha256: str | None = None,
    ) -> dict:
        """Commit one coherent post-update state; failed writes leave old checkpoints."""
        if not _safe_name(attempt_id):
            raise ValueError("attempt_id must be a single path-safe name")
        counts = (completed_updates, environment_transitions, agent_transitions)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("progress counts must be nonnegative integers")
        if not isinstance(active_training_seconds, (float, int)) or not (
            0 <= active_training_seconds < float("inf")
        ):
            raise ValueError("active_training_seconds must be finite and nonnegative")
        if not source_identity or not runtime_identity:
            raise ValueError("source and runtime identities are required")
        if parent_checkpoint_sha256 is not None and (
            len(parent_checkpoint_sha256) != 64
            or any(char not in "0123456789abcdef" for char in parent_checkpoint_sha256)
        ):
            raise ValueError("parent checkpoint hash must be SHA-256 hex")

        checkpoint_id = f"{completed_updates:012d}-{uuid4().hex}"
        payload_name = f"checkpoint-{checkpoint_id}.pt"
        manifest_name = f"checkpoint-{checkpoint_id}.json"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.directory, prefix=".checkpoint-", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                os.fchmod(stream.fileno(), self.file_mode)
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
            if temporary.stat().st_size == 0:
                raise ValueError("checkpoint writer produced an empty payload")
            if validator is not None:
                validator(temporary)
            payload_sha256 = _sha256(temporary)
            payload_bytes = temporary.stat().st_size
            payload_path = self.directory / payload_name
            os.replace(temporary, payload_path)
            temporary = None
            _sync_directory(self.directory)
            saved_at_utc = utc_now()
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "checkpoint_id": checkpoint_id,
                "run_id": self.run_id,
                "attempt_id": attempt_id,
                "recovery_mode": RECOVERY_MODE,
                "config_sha256": self.config_sha256,
                "source_identity": source_identity,
                "runtime_identity": runtime_identity,
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "completed_updates": completed_updates,
                "environment_transitions": environment_transitions,
                "agent_transitions": agent_transitions,
                "active_training_seconds": float(active_training_seconds),
                "saved_at_utc": saved_at_utc,
                "saved_at_ist": as_ist(saved_at_utc),
                "payload": {
                    "file": payload_name,
                    "bytes": payload_bytes,
                    "sha256": payload_sha256,
                },
            }
            write_json_atomic(self.directory / manifest_name, manifest, mode=self.file_mode)
            _sync_directory(self.directory)
            write_json_atomic(
                self.directory / "latest.json",
                {"schema_version": SCHEMA_VERSION, "manifest": manifest_name},
                mode=self.file_mode,
            )
            _sync_directory(self.directory)
            return manifest
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def verify(self, manifest_path: Path) -> tuple[dict, Path]:
        """Validate metadata and payload before any model deserialization."""
        if manifest_path.parent.resolve() != self.directory.resolve():
            raise ValueError("manifest must be inside the checkpoint directory")
        if manifest_path.is_symlink():
            raise ValueError("checkpoint manifest cannot be a link")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema")
        if (
            manifest.get("run_id") != self.run_id
            or manifest.get("config_sha256") != self.config_sha256
        ):
            raise ValueError("checkpoint run or configuration differs")
        if manifest.get("recovery_mode") != RECOVERY_MODE:
            raise ValueError("unsupported checkpoint recovery mode")
        checkpoint_id = manifest.get("checkpoint_id")
        if (
            not isinstance(checkpoint_id, str)
            or manifest_path.name != f"checkpoint-{checkpoint_id}.json"
        ):
            raise ValueError("checkpoint manifest name or identity differs")
        payload = manifest.get("payload")
        if not isinstance(payload, dict) or payload.get("file") != f"checkpoint-{checkpoint_id}.pt":
            raise ValueError("checkpoint payload name or identity differs")
        if not _safe_name(payload["file"]):
            raise ValueError("checkpoint payload path is unsafe")
        path = self.directory / payload["file"]
        if not path.is_file() or path.is_symlink():
            raise ValueError("checkpoint payload is missing or is a link")
        if path.stat().st_size != payload.get("bytes") or _sha256(path) != payload.get("sha256"):
            raise ValueError("checkpoint payload checksum or size differs")
        return manifest, path

    def for_update(self, completed_updates: int) -> tuple[dict, Path]:
        """Select one exact valid update without falling forward or backward."""
        if type(completed_updates) is not int or completed_updates < 0:
            raise ValueError("completed_updates must be a nonnegative integer")
        matches = []
        for candidate in self.directory.glob("checkpoint-*.json"):
            try:
                manifest, payload = self.verify(candidate)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if manifest["completed_updates"] == completed_updates:
                matches.append((manifest, payload))
        if len(matches) > 1:
            raise ValueError(f"multiple valid checkpoints exist for update {completed_updates}")
        if not matches:
            raise FileNotFoundError(f"no valid checkpoint exists for update {completed_updates}")
        return matches[0]

    def latest_valid(self) -> tuple[dict, Path]:
        """Prefer highest committed update; fall back past incomplete/corrupt saves."""
        manifests = sorted(
            self.directory.glob("checkpoint-*.json"),
            key=lambda path: path.name,
            reverse=True,
        )
        for candidate in manifests:
            try:
                return self.verify(candidate)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
        raise FileNotFoundError("no valid committed checkpoint for this run/configuration")
