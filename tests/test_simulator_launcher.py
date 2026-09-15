"""CPU checks of launcher failure handling; these never execute Docker."""

import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_isaac_sim_smoke.py"
SPEC = importlib.util.spec_from_file_location("smoke_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class SimulatorLauncherTests(unittest.TestCase):
    def run_fake(self, behavior):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            with (
                patch.object(launcher, "create_run_directory", return_value=run_dir),
                patch.object(launcher.subprocess, "run", side_effect=behavior) as process,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = launcher.main(["--accept-eula", "--gpu", "2"])
            report = json.loads((run_dir / "report.json").read_text())
            return code, report, process.call_args_list

    def test_zero_exit_without_probe_result_is_failure(self):
        code, report, _ = self.run_fake(lambda *a, **kw: subprocess.CompletedProcess(a[0], 0))
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")

    def test_explicit_completed_probe_is_required(self):
        def completed(command, **kwargs):
            result = {
                "status": "passed",
                "updates_completed": 20,
                "cuda_sum": 1024.0,
                "probe_sha256": hashlib.sha256(
                    SCRIPT.with_name("isaac_sim_smoke.py").read_bytes()
                ).hexdigest(),
            }
            kwargs["stdout"].write("ALIGN_SMOKE_RESULT=" + json.dumps(result) + "\n")
            return subprocess.CompletedProcess(command, 0)

        code, report, calls = self.run_fake(completed)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed")
        self.assertIn("device=2", calls[0].args[0])
        self.assertIn(launcher.IMAGE, calls[0].args[0])

    def test_timeout_removes_only_its_named_container(self):
        def timeout(command, **kwargs):
            if "run" in command:
                raise subprocess.TimeoutExpired(command, 1200)
            return subprocess.CompletedProcess(command, 0, "removed", "")

        code, report, calls = self.run_fake(timeout)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "timed_out")
        command = calls[0].args[0]
        name = command[command.index("--name") + 1]
        self.assertEqual(calls[1].args[0], ["sudo", "-n", "docker", "rm", "--force", name])

    def test_no_license_acceptance_does_not_launch(self):
        with (
            patch.object(launcher.subprocess, "run") as process,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            launcher.main([])
        self.assertEqual(error.exception.code, 2)
        process.assert_not_called()
