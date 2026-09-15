"""Artifact integrity matters even when a diagnostic is interrupted."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from align.artifacts import create_run_directory, diagnostic_logger, write_json_atomic


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
