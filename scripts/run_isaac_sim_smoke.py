"""Host launcher: use uv; run the probe in the pinned vendor container."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from align.artifacts import create_run_directory, utc_now, write_json_atomic

IMAGE = (
    "nvcr.io/nvidia/isaac-sim@sha256:"
    "5bd94fce4318ca2f8bf887c4ce3220bfc1cedd303008bf6d6976b0e9e556d173"
)


def read_probe_result(log_path):
    """Require an explicit post-shutdown result, not just Docker exit code zero."""
    result = None
    with log_path.open(errors="replace") as stream:
        for line in stream:
            if line.startswith("ALIGN_SMOKE_RESULT="):
                result = json.loads(line.partition("=")[2])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="Allocated host NVIDIA device index")
    parser.add_argument(
        "--timeout", type=int, default=1200, help="Whole container timeout in seconds"
    )
    parser.add_argument("--accept-eula", action="store_true", help="Accept the NVIDIA image EULA")
    args = parser.parse_args(argv)
    if not args.accept_eula:
        parser.error("Read the image EULA in docs/04-isaac-sim-smoke.md, then use --accept-eula")
    if args.gpu < 0 or not 1 <= args.timeout <= 3600:
        parser.error("GPU index must be nonnegative; timeout must be 1–3600 seconds")

    root = Path(__file__).resolve().parents[1]
    probe = Path(__file__).with_name("isaac_sim_smoke.py").resolve()
    run_dir = create_run_directory(root / "runs" / "isaac-sim-smoke")
    container_name = "align-smoke-" + run_dir.name
    (run_dir / "kit-logs").mkdir()
    command = [
        "sudo",
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        container_name,
        "--runtime=nvidia",
        "--gpus",
        f"device={args.gpu}",
        "-e",
        "ACCEPT_EULA=Y",
        "--mount",
        f"type=bind,src={probe},dst=/workspace/probe.py,readonly",
        "--mount",
        f"type=bind,src={run_dir / 'kit-logs'},dst=/root/.nvidia-omniverse/logs",
        "--entrypoint",
        "/isaac-sim/python.sh",
        IMAGE,
        "/workspace/probe.py",
        "--allow-root",
    ]
    report = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": "running",
        "started_at_utc": utc_now(),
        "image": IMAGE,
        "host_gpu_index": args.gpu,
        "timeout_seconds": args.timeout,
        "accepted_eula": True,
        "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
        "command": command,
        "container_exit_code": None,
        "probe_result": None,
    }
    report_path = run_dir / "report.json"
    write_json_atomic(report_path, report)
    print(f"Run directory: {run_dir}", flush=True)
    print(
        "First startup can take several minutes. Follow console.log from another terminal.",
        flush=True,
    )
    started = time.perf_counter()
    cleanup_needed = False
    exit_code = 1
    try:
        with (run_dir / "console.log").open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False
            )
        report["container_exit_code"] = result.returncode
        report["probe_result"] = read_probe_result(run_dir / "console.log")
        probe_result = report["probe_result"]
        passed = (
            result.returncode == 0
            and isinstance(probe_result, dict)
            and probe_result.get("status") == "passed"
            and probe_result.get("updates_completed") == 20
            and probe_result.get("cuda_sum") == 1024.0
            and probe_result.get("probe_sha256") == report["probe_sha256"]
        )
        report["status"] = "passed" if passed else "failed"
        exit_code = 0 if passed else 1
    except subprocess.TimeoutExpired:
        report["status"] = "timed_out"
        cleanup_needed = True
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        cleanup_needed = True
        exit_code = 130
    except (OSError, ValueError) as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        cleanup_needed = True
    finally:
        if cleanup_needed:
            try:
                cleanup = subprocess.run(
                    ["sudo", "-n", "docker", "rm", "--force", container_name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                report["cleanup"] = {
                    "exit_code": cleanup.returncode,
                    "output": cleanup.stdout + cleanup.stderr,
                }
            except (OSError, subprocess.TimeoutExpired) as exc:
                report["cleanup"] = {"error": str(exc)}
            print(
                f"If cleanup failed, remove only this test container: {container_name}", flush=True
            )
        report["finished_at_utc"] = utc_now()
        report["duration_seconds"] = time.perf_counter() - started
        write_json_atomic(report_path, report)
    print(f"Result: {report['status']}. Report: {report_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
