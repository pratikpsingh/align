"""Exercise report semantics and the installed entrypoint."""

import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from align.cli import main
from align.runtime.diagnostics import Check


class CliTests(unittest.TestCase):
    def invoke(self, output, checks):
        with (
            patch("align.cli.collect_checks", return_value=iter(checks)),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            code = main(["doctor", "--output-dir", str(output)])
        reports = list(output.glob("*/report.json"))
        self.assertEqual(len(reports), 1)
        return code, json.loads(reports[0].read_text()), reports[0].parent

    def test_optional_missing_gpu_completes_and_records_configuration(self):
        with TemporaryDirectory() as directory:
            code, report, run_dir = self.invoke(
                Path(directory), [Check("nvidia", "warning", "Not available", {})]
            )
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["simulation_validation"], "not_performed")
            self.assertEqual(report["configuration"]["output_dir"], str(Path(directory).resolve()))
            self.assertGreaterEqual(report["duration_seconds"], 0)
            self.assertTrue(report["finished_at_utc"])
            events = [
                json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[0]["event"], "started")
            self.assertEqual(events[-1]["event"], "finished")
            self.assertTrue(all(event["run_id"] == report["run_id"] for event in events))

    def test_failed_requirement_still_saves_report(self):
        with TemporaryDirectory() as directory:
            code, report, _ = self.invoke(
                Path(directory), [Check("nvidia", "error", "Required device absent", {})]
            )
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "requirements_unmet")
        self.assertEqual(report["exit_code"], 1)

    def test_interruption_and_unexpected_failure_keep_completed_probes(self):
        for error, status, expected_code in (
            (KeyboardInterrupt(), "interrupted", 130),
            (RuntimeError("Probe failed"), "failed", 1),
        ):
            with self.subTest(status=status), TemporaryDirectory() as directory:

                def checks(config, run_dir, error=error):
                    yield Check("python", "ok", "Available", {})
                    raise error

                with (
                    patch("align.cli.collect_checks", side_effect=checks),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    code = main(["doctor", "--output-dir", directory])
                report = json.loads(next(Path(directory).glob("*/report.json")).read_text())
                self.assertEqual(code, expected_code)
                self.assertEqual(report["status"], status)
                self.assertEqual([check["name"] for check in report["checks"]], ["python"])

    def test_invalid_options_create_no_artifacts(self):
        for arguments in (["--timeout", "nan"], ["--unknown-option"]):
            with self.subTest(arguments=arguments), TemporaryDirectory() as directory:
                output = Path(directory) / "reports"
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    main(["doctor", "--output-dir", str(output), *arguments])
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())

    def test_unwritable_destination_is_an_explicit_failure(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_text("Existing file")
            errors = io.StringIO()
            with redirect_stderr(errors):
                code = main(["doctor", "--output-dir", str(path)])
            self.assertEqual(code, 1)
            self.assertIn("could not read/write", errors.getvalue())
            self.assertEqual(path.read_text(), "Existing file")

    def test_installed_package_runs_outside_checkout_without_simulator_imports(self):
        program = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'omni', 'omni_drones', 'isaacsim'}:
        raise AssertionError('Unexpected simulator/learning import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from unittest.mock import patch
from align.cli import main
from align.runtime.diagnostics import CommandResult
with patch('align.runtime.diagnostics.run_command',
           return_value=CommandResult('missing', error='No external tools in this test')):
    raise SystemExit(main(['doctor', '--output-dir', 'reports', '--log-level', 'ERROR']))
"""
        with TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", program],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Report:", result.stdout)
        self.assertIn("Simulation: not tested", result.stdout)
