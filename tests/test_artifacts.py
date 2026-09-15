"""Artifact integrity matters even when a diagnostic is interrupted."""

import json
import logging
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from align.artifacts import (
    EventFormatter,
    ISTFormatter,
    as_ist,
    create_run_directory,
    diagnostic_logger,
    write_json_atomic,
)


class ArtifactTests(unittest.TestCase):
    def test_serialization_failure_preserves_previous_report(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_json_atomic(path, {"status": "running"})
            with self.assertRaises(ValueError):
                write_json_atomic(path, {"duration": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"status": "running"})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_publication_preserves_previous_report(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_json_atomic(path, {"status": "running"})
            with (
                patch("align.artifacts.os.replace", side_effect=OSError("Disk unavailable")),
                self.assertRaises(OSError),
            ):
                write_json_atomic(path, {"status": "completed"})
            self.assertEqual(json.loads(path.read_text()), {"status": "running"})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_repeated_invocations_have_unique_directories(self):
        with TemporaryDirectory() as directory:
            first = create_run_directory(Path(directory))
            second = create_run_directory(Path(directory))
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir() and second.is_dir())

    def test_event_identity_and_handler_cleanup(self):
        with TemporaryDirectory() as directory:
            run_dir = create_run_directory(Path(directory))
            with diagnostic_logger(run_dir, "ERROR") as logger:
                logger.info("Probe complete", extra={"event": "check_completed"})
            self.assertEqual(logger.handlers, [])
            events = [
                json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["run_id"], run_dir.name)
            self.assertEqual(events[0]["event"], "check_completed")
            self.assertIn("Probe complete", (run_dir / "doctor.log").read_text())

    def test_ist_rollover_preserves_instant(self):
        utc = "2026-09-15T19:20:28.862792+00:00"
        converted = as_ist(utc)
        self.assertEqual(converted, "2026-09-16T00:50:28.862792+05:30")
        self.assertEqual(datetime.fromisoformat(converted), datetime.fromisoformat(utc))
        record = logging.LogRecord("align.doctor.test", logging.INFO, "", 0, "ready", (), None)
        record.created = datetime.fromisoformat(utc).timestamp()
        self.assertEqual(ISTFormatter().formatTime(record), "2026-09-16T00:50:28.862+05:30")
        event = json.loads(EventFormatter().format(record))
        self.assertEqual(event["timestamp_ist"], converted)
        self.assertEqual(event["timestamp_utc"], utc)
        with self.assertRaises(ValueError):
            as_ist("2026-09-15T19:20:28")

    def test_run_directory_uses_explicit_ist_label(self):
        with TemporaryDirectory() as directory:
            run_dir = create_run_directory(Path(directory))
            self.assertRegex(run_dir.name, r"^\d{8}T\d{6}\.\d{6}IST-[a-f0-9]{8}$")
