"""Standalone Python 3.10 probe, executed by Isaac Sim's bundled python.sh.

This deliberately does not import ALiGn: its development environment is separate
from the simulator runtime whose compatibility we are establishing.
"""

import hashlib
import json
import platform
import time
import traceback
from pathlib import Path


def main():
    started = time.perf_counter()
    result = {
        "schema_version": 1,
        "python": platform.python_version(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "running",
        "updates_completed": 0,
        "drone_physics_tested": False,
    }
    app = None
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True, "multi_gpu": False, "fast_shutdown": False})
        result["startup_seconds"] = time.perf_counter() - started

        # Simulator extensions must be imported after application startup.
        import omni.usd
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot execute CUDA in the simulator runtime")
        result["torch_version"] = torch.__version__
        result["torch_cuda_version"] = torch.version.cuda
        result["visible_cuda_devices"] = torch.cuda.device_count()
        result["cuda_device_name"] = torch.cuda.get_device_name(0)
        if result["visible_cuda_devices"] != 1:
            raise RuntimeError("Expected exactly one exposed CUDA device")
        value = torch.ones(1024, device="cuda:0").sum().item()
        torch.cuda.synchronize()
        if value != 1024.0:
            raise RuntimeError("CUDA arithmetic check failed")
        result["cuda_sum"] = value
        if omni.usd.get_context().get_stage() is None:
            raise RuntimeError("Isaac Sim did not create a USD stage")
        for _ in range(20):
            app.update()
            result["updates_completed"] += 1
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if app is not None:
            try:
                app.close()
            except Exception as exc:
                result["status"] = "failed"
                result["shutdown_error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
        result["total_seconds"] = time.perf_counter() - started
        print("ALIGN_SMOKE_RESULT=" + json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
