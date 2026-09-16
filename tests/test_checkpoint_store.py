"""Host-safe checkpoint integrity, fallback, and publication checks."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from align.learning.checkpoint_store import CheckpointStore

CONFIG_HASH = hashlib.sha256(b"frozen-config").hexdigest()


class CheckpointStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.store = CheckpointStore(
            self.directory, run_id="logical-run", config_sha256=CONFIG_HASH
        )

    def save(self, step, payload=b"state", writer=None):
        return self.store.save(
            writer or (lambda stream: stream.write(payload)),
            attempt_id="attempt-1",
            completed_updates=step,
            environment_transitions=step * 4,
            agent_transitions=step * 16,
            active_training_seconds=float(step),
            source_identity="git:abc",
            runtime_identity="image:sha256:abc",
        )

    def test_immutable_checkpoint_and_latest_selection(self):
        first = self.save(1, b"first")
        self.save(2, b"second")
        self.save(3, b"third")
        newest = self.save(4, b"fourth")
        selected, path = self.store.latest_valid()
        self.assertEqual(selected["checkpoint_id"], newest["checkpoint_id"])
        self.assertEqual(path.read_bytes(), b"fourth")
        self.assertGreaterEqual(len(list(self.directory.glob("checkpoint-*.pt"))), 3)
        self.assertEqual(first["recovery_mode"], "training_resume_with_environment_reset")
        self.assertEqual(
            json.loads((self.directory / "latest.json").read_text())["manifest"],
            f"checkpoint-{newest['checkpoint_id']}.json",
        )

    def test_corrupt_newest_and_broken_pointer_fall_back_to_previous(self):
        first = self.save(1, b"first")
        second = self.save(2, b"second")
        (self.directory / second["payload"]["file"]).write_bytes(b"corrupt")
        (self.directory / "latest.json").write_text("{")
        selected, path = self.store.latest_valid()
        self.assertEqual(selected["checkpoint_id"], first["checkpoint_id"])
        self.assertEqual(path.read_bytes(), b"first")

    def test_failed_writer_and_incomplete_temporary_preserve_old_state(self):
        first = self.save(1, b"first")

        def interrupted(stream):
            stream.write(b"partial")
            raise OSError("disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            self.save(2, writer=interrupted)
        (self.directory / ".checkpoint-abandoned.tmp").write_bytes(b"half-written")
        selected, _ = self.store.latest_valid()
        self.assertEqual(selected["checkpoint_id"], first["checkpoint_id"])
        self.assertEqual(len(list(self.directory.glob("checkpoint-*.json"))), 1)

    def test_abrupt_subprocess_exit_leaves_committed_checkpoint_selectable(self):
        first = self.save(1, b"first")
        code = """
import os
import sys
from pathlib import Path
from align.learning.checkpoint_store import CheckpointStore

store = CheckpointStore(Path(sys.argv[1]), run_id="logical-run", config_sha256=sys.argv[2])
def terminate(stream):
    stream.write(b"partial")
    stream.flush()
    os._exit(17)
store.save(
    terminate,
    attempt_id="attempt-2",
    completed_updates=2,
    environment_transitions=8,
    agent_transitions=32,
    active_training_seconds=2.0,
    source_identity="git:abc",
    runtime_identity="image:abc",
)
"""
        completed = subprocess.run(
            [sys.executable, "-c", code, str(self.directory), CONFIG_HASH], check=False
        )
        self.assertEqual(completed.returncode, 17)
        selected, path = self.store.latest_valid()
        self.assertEqual(selected["checkpoint_id"], first["checkpoint_id"])
        self.assertEqual(path.read_bytes(), b"first")

    def test_failed_prepublication_validation_preserves_previous_checkpoint(self):
        first = self.save(1, b"first")

        def reject(_path):
            raise ValueError("payload cannot be deserialized")

        with self.assertRaisesRegex(ValueError, "cannot be deserialized"):
            self.store.save(
                lambda stream: stream.write(b"malformed"),
                validator=reject,
                attempt_id="attempt-2",
                completed_updates=2,
                environment_transitions=8,
                agent_transitions=32,
                active_training_seconds=2.0,
                source_identity="git:abc",
                runtime_identity="image:abc",
            )
        selected, path = self.store.latest_valid()
        self.assertEqual(selected["checkpoint_id"], first["checkpoint_id"])
        self.assertEqual(path.read_bytes(), b"first")

    def test_manifest_without_payload_and_wrong_config_are_rejected(self):
        manifest = self.save(1)
        payload = self.directory / manifest["payload"]["file"]
        payload.unlink()
        with self.assertRaises(FileNotFoundError):
            self.store.latest_valid()
        self.save(2)
        other = CheckpointStore(self.directory, run_id="logical-run", config_sha256="0" * 64)
        with self.assertRaises(FileNotFoundError):
            other.latest_valid()

    def test_rejects_unsafe_names_and_invalid_progress(self):
        with self.assertRaises(ValueError):
            CheckpointStore(self.directory, run_id="../escape", config_sha256=CONFIG_HASH)
        with self.assertRaises(ValueError):
            self.store.save(
                lambda stream: stream.write(b"x"),
                attempt_id="../escape",
                completed_updates=1,
                environment_transitions=0,
                agent_transitions=0,
                active_training_seconds=0.0,
                source_identity="git:abc",
                runtime_identity="image:abc",
            )
