"""Host/container artifact boundaries and device-allocation checks."""

import stat
import tempfile
import unittest
from pathlib import Path

from align.artifacts import write_json_atomic
from align.runtime.drone_runtime import check_gpu_available


class RuntimeBoundaryTests(unittest.TestCase):
    def test_export_json_is_readable_without_changing_private_default(self):
        with tempfile.TemporaryDirectory() as folder:
            public = Path(folder) / "public.json"
            private = Path(folder) / "private.json"
            write_json_atomic(public, {"status": "passed"}, mode=0o644)
            write_json_atomic(private, {"status": "passed"})
            self.assertEqual(stat.S_IMODE(public.stat().st_mode), 0o644)
            self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)

    def test_busy_selected_gpu_is_rejected(self):
        inventory = {
            "exit_code": 0,
            "stdout": "index, uuid, name, driver, total, used, utilization\n"
            "0, GPU-zero, A4000, 580, 16376 MiB, 16 MiB, 0 %\n",
        }
        processes = {"exit_code": 0, "stdout": "gpu_uuid, pid, process_name, used_memory\n"}
        self.assertEqual(check_gpu_available(inventory, processes, 0)["uuid"], "GPU-zero")
        processes["stdout"] += "GPU-zero, 1234, python, 100 MiB\n"
        with self.assertRaises(RuntimeError):
            check_gpu_available(inventory, processes, 0)

    def test_missing_device_and_failed_inventory_are_rejected(self):
        with self.assertRaises(RuntimeError):
            check_gpu_available({"exit_code": 1}, {"exit_code": 0}, 0)
        with self.assertRaises(RuntimeError):
            check_gpu_available(
                {"exit_code": 0, "stdout": "header\n"}, {"exit_code": 0, "stdout": "header\n"}, 4
            )
