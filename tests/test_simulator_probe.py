"""Probe lifecycle checks with fake modules, without Isaac Sim or CUDA."""

import contextlib
import importlib.util
import io
import json
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/isaac_sim_smoke.py"
SPEC = importlib.util.spec_from_file_location("smoke_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class SimulatorProbeTests(unittest.TestCase):
    def exercise(self, *, cuda=True, close_error=None, mode="fast"):
        app = Mock()
        app.close.side_effect = close_error
        isaacsim = ModuleType("isaacsim")
        isaacsim.SimulationApp = Mock(return_value=app)
        omni = ModuleType("omni")
        usd = ModuleType("omni.usd")
        usd.get_context = Mock()
        omni.usd = usd
        torch = Mock()
        torch.__version__ = "fake"
        torch.version.cuda = "fake"
        torch.cuda.is_available.return_value = cuda
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_name.return_value = "fake GPU"
        torch.ones.return_value.sum.return_value.item.return_value = 1024.0
        output = io.StringIO()
        with (
            patch.dict(
                "sys.modules", {"isaacsim": isaacsim, "omni": omni, "omni.usd": usd, "torch": torch}
            ),
            patch.dict("os.environ", {"ALIGN_SHUTDOWN_MODE": mode}),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            try:
                code = probe.main()
            except SystemExit as exc:
                code = exc.code
        records = [
            (line.partition("=")[0], json.loads(line.partition("=")[2]))
            for line in output.getvalue().splitlines()
        ]
        return code, records, app

    def test_fast_exit_preserves_checks_before_close(self):
        code, records, app = self.exercise(close_error=SystemExit(0))
        self.assertEqual(code, 0)
        marker, result = records[-1]
        self.assertEqual(marker, "ALIGN_SMOKE_PROGRESS")
        self.assertEqual(result["phase"], "before_close")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["updates_completed"], 20)
        self.assertEqual(app.update.call_count, 20)

    def test_failed_check_survives_zero_fast_exit(self):
        _, records, app = self.exercise(cuda=False, close_error=SystemExit(0))
        self.assertEqual(records[-1][1]["status"], "failed")
        self.assertIn("CUDA", records[-1][1]["error"])
        app.update.assert_not_called()
        app.close.assert_called_once()

    def test_full_close_exception_cannot_emit_passed_final_result(self):
        code, records, _ = self.exercise(close_error=RuntimeError("close failed"), mode="full")
        self.assertEqual(code, 1)
        marker, result = records[-1]
        self.assertEqual(marker, "ALIGN_SMOKE_RESULT")
        self.assertEqual(result["status"], "failed")
        self.assertIn("close failed", result["shutdown_error"])
