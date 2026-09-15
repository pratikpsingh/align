"""Probe failure modes without requiring an NVIDIA GPU or simulator."""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from align.config import DoctorConfig
from align.runtime.diagnostics import (
    CommandResult,
    parse_gpu_csv,
    probe_nvidia,
    probe_packages,
    run_command,
)


class ConfigurationTests(unittest.TestCase):
    def test_nonfinite_and_out_of_range_timeouts_are_rejected(self):
        for timeout in (0, -1, 31, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                DoctorConfig(Path("output"), timeout_seconds=timeout)


class NvidiaTests(unittest.TestCase):
    def test_multiple_devices_preserve_individual_memory(self):
        devices = parse_gpu_csv(
            "0, NVIDIA RTX A4000, 16376, 550.54.15\n1, NVIDIA RTX A4000, 16376, 550.54.15\n"
        )
        self.assertEqual([device["memory_total_mib"] for device in devices], [16376, 16376])
        self.assertEqual([device["index"] for device in devices], [0, 1])

    def test_unknown_memory_is_null_not_zero(self):
        device = parse_gpu_csv("0, NVIDIA device, [N/A], 550.54.15")[0]
        self.assertIsNone(device["memory_total_mib"])

    def test_malformed_output_is_rejected(self):
        examples = (
            "0, GPU, 16",
            "0, GPU, -16, driver",
            "0, GPU, 16, driver\n0, GPU, 16, driver",
            "index, name, memory, driver",
        )
        for example in examples:
            with self.subTest(example=example), self.assertRaises(ValueError):
                parse_gpu_csv(example)

    def test_missing_nvidia_is_optional_unless_requested(self):
        with patch(
            "align.runtime.diagnostics.run_command",
            return_value=CommandResult("missing", error="Not installed"),
        ):
            self.assertEqual(probe_nvidia(5, False).status, "warning")
            self.assertEqual(probe_nvidia(5, True).status, "error")

    def test_failed_timed_out_empty_or_malformed_query_never_passes(self):
        cases = (
            CommandResult("failed", error="Driver failed"),
            CommandResult("timeout", error="Timed out"),
            CommandResult("ok", stdout=""),
            CommandResult("ok", stdout="unexpected output"),
        )
        for result in cases:
            with (
                self.subTest(result=result),
                patch("align.runtime.diagnostics.run_command", return_value=result),
            ):
                check = probe_nvidia(5, True)
                self.assertEqual(check.status, "error")
                self.assertEqual(check.details["devices"], [])
                self.assertIsNotNone(check.details["reason"])

    def test_visible_gpu_does_not_claim_cuda_execution(self):
        with patch(
            "align.runtime.diagnostics.run_command",
            return_value=CommandResult("ok", stdout="0, GPU, 8192, driver"),
        ):
            check = probe_nvidia(5, True)
        self.assertEqual(check.status, "ok")
        self.assertIn("untested", check.message)


class CommandTests(unittest.TestCase):
    def test_failures_are_distinguishable_and_timeout_is_passed(self):
        cases = (
            (FileNotFoundError(), "missing"),
            (PermissionError("Denied"), "failed"),
            (subprocess.TimeoutExpired("tool", 2), "timeout"),
        )
        for exception, expected in cases:
            with (
                self.subTest(expected=expected),
                patch("align.runtime.diagnostics.subprocess.run", side_effect=exception) as run,
            ):
                self.assertEqual(run_command(["tool", "argument with spaces"], 2).status, expected)
                self.assertEqual(run.call_args.kwargs["timeout"], 2)
                self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_driver_error_output_is_retained(self):
        result = subprocess.CompletedProcess(["tool"], 9, "", "Driver/library mismatch")
        with patch("align.runtime.diagnostics.subprocess.run", return_value=result):
            check = run_command(["tool"], 5)
        self.assertEqual(check.status, "failed")
        self.assertIn("Driver/library mismatch", check.error)


class PackageTests(unittest.TestCase):
    def test_even_present_packages_do_not_imply_simulation_validation(self):
        with patch("align.runtime.diagnostics.metadata.version", return_value="test-version"):
            check = probe_packages()
        self.assertEqual(check.status, "not_checked")
        self.assertFalse(check.details["simulation_tested"])
        self.assertFalse(check.details["simulator_imported"])
