"""Host-only preparation and launch of the pinned single-drone runtime."""

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from align.artifacts import as_ist, create_run_directory, utc_now, write_json_atomic
from align.runtime.diagnostics import probe_source, probe_system
from align.simulation.contract import DroneCheckConfig
from align.simulation.report import assess_saved_run


def project_root():
    root = Path(__file__).resolve().parents[3]
    if not (root / "runtime/stack.json").is_file():
        raise RuntimeError("Run from an editable ALiGn checkout with runtime/stack.json")
    return root


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(command, timeout=30, cwd=None):
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, cwd=cwd, check=False
        )
        return {
            "argv": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": command, "exit_code": -1, "stdout": "", "stderr": str(exc)}


def check_gpu_available(inventory, processes, selected):
    """Reject a missing/busy selected device; being idle is not a scheduler reservation."""
    if inventory["exit_code"] or processes["exit_code"]:
        raise RuntimeError("Could not verify current GPU availability")
    rows = list(csv.reader(io.StringIO(inventory["stdout"]), skipinitialspace=True))[1:]
    row = next((row for row in rows if int(row[0]) == selected), None)
    if row is None:
        raise RuntimeError(f"GPU {selected} does not exist")
    active = list(csv.reader(io.StringIO(processes["stdout"]), skipinitialspace=True))[1:]
    if any(item and item[0].strip() == row[1].strip() for item in active):
        raise RuntimeError(
            f"GPU {selected} has active compute processes; select an idle allocated device"
        )
    if int(row[5].split()[0]) > 512 or int(row[6].split()[0]) > 5:
        raise RuntimeError(f"GPU {selected} appears busy; inspect the saved preflight")
    return {"index": selected, "uuid": row[1].strip(), "name": row[2].strip()}


def execute(command, log, timeout, cwd=None):
    with log.open("w") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, cwd=cwd, check=False
        )
    return result.returncode


def finish(run, report, started):
    report["finished_at_utc"] = utc_now()
    report["finished_at_ist"] = as_ist(report["finished_at_utc"])
    report["duration_seconds"] = time.perf_counter() - started
    write_json_atomic(run / "report.json", report)
    print(f"{report['finished_at_ist']} IST {report['status']}: {run}", flush=True)


def new_report():
    value = {"status": "running", "started_at_utc": utc_now(), "schema_version": 1}
    value["started_at_ist"] = as_ist(value["started_at_utc"])
    return value


def docker_prefix(mode):
    return ["sudo", "-n", "docker"] if mode == "sudo" else ["docker"]


def prepare(root, run):
    stack = json.loads((root / "runtime/stack.json").read_text())
    source = root / ".runtime/sources/OmniDrones"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", stack["omnidrones"]["repository"], str(source)],
            check=True,
        )
    revision = stack["omnidrones"]["revision"]
    subprocess.run(["git", "-C", str(source), "cat-file", "-e", revision + "^{commit}"], check=True)
    context = run / "context"
    context.mkdir()
    (context / "OmniDrones").mkdir()
    archive = run / "omnidrones.tar"
    subprocess.run(["git", "-C", str(source), "archive", revision, "-o", str(archive)], check=True)
    with tarfile.open(archive) as stream:
        stream.extractall(context / "OmniDrones", filter="data")
    for patch in sorted((root / "runtime/patches").glob("*.patch")):
        subprocess.run(
            [
                "git",
                "apply",
                "--unsafe-paths",
                "--directory",
                str(context / "OmniDrones"),
                str(patch),
            ],
            cwd=root,
            check=True,
        )
    shutil.copytree(root / "runtime", context / "runtime")
    wheels = context / "wheels"
    wheels.mkdir()
    subprocess.run(
        [
            "uvx",
            "--from",
            "pip==25.3",
            "pip",
            "download",
            "--no-deps",
            "--require-hashes",
            "--only-binary=:all:",
            "--platform",
            "manylinux2014_x86_64",
            "--platform",
            "manylinux_2_17_x86_64",
            "--python-version",
            "310",
            "--implementation",
            "cp",
            "--abi",
            "cp310",
            "-r",
            str(root / "runtime/additions.lock"),
            "--dest",
            str(wheels),
        ],
        check=True,
    )
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(wheels)], cwd=root, check=True)
    manifest = {
        str(p.relative_to(context)): digest(p) for p in sorted(context.rglob("*")) if p.is_file()
    }
    write_json_atomic(run / "context-manifest.json", manifest)
    return context, stack


def build_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the pinned OmniDrones image without changing its base."
    )
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared-run", type=Path)
    args = parser.parse_args(argv)
    root = project_root()
    run = (
        args.prepared_run.resolve()
        if args.prepared_run
        else create_run_directory(root / "runs/runtime-build")
    )
    started = time.perf_counter()
    if args.prepared_run:
        report = json.loads((run / "report.json").read_text())
        if report["status"] != "prepared":
            raise ValueError(
                "Only an unbuilt prepared context can be used; create a new build attempt"
            )
    else:
        report = new_report()
    print(f"Build artifacts: {run}", flush=True)
    write_json_atomic(run / "report.json", report)
    try:
        if not args.prepared_run:
            context, stack = prepare(root, run)
            report.update(stack=stack, source=probe_source(30).details)
        else:
            context = run / "context"
            manifest = json.loads((run / "context-manifest.json").read_text())
            if any(
                not (context / p).is_file() or digest(context / p) != sha
                for p, sha in manifest.items()
            ):
                raise RuntimeError("Prepared context was changed")
        tag = "align-omnidrones:" + run.name.lower()
        docker = docker_prefix(args.docker)
        command = [
            *docker,
            "build",
            "--network=none",
            "--pull=false",
            "-t",
            tag,
            "-f",
            str(context / "runtime/Dockerfile"),
            str(context),
        ]
        report.update(tag=tag, command=command)
        if args.prepare_only:
            report["status"] = "prepared"
        else:
            report["docker_version"] = capture([*docker, "version", "--format", "{{json .}}"])
            if report["docker_version"]["exit_code"]:
                raise RuntimeError(
                    "Docker access failed; authenticate in this terminal with sudo -v"
                )
            report["build_exit_code"] = execute(command, run / "console.log", 1800)
            if report["build_exit_code"]:
                raise RuntimeError("Docker build failed; inspect console.log")
            inspection = capture([*docker, "image", "inspect", tag])
            if inspection["exit_code"]:
                raise RuntimeError("Cannot inspect built image")
            report["image_inspect"] = json.loads(inspection["stdout"])
            report["image_id"] = report["image_inspect"][0]["Id"]
            report["status"] = "built"
            (root / ".runtime").mkdir(exist_ok=True)
            write_json_atomic(
                root / ".runtime/latest-build.json", {"report": str(run / "report.json")}
            )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        report.update(status="failed", error=str(exc))
    finally:
        finish(run, report, started)
    return 0 if report["status"] in {"built", "prepared"} else 1


def valid_result(exit_code, probe, metrics):
    return (
        exit_code == 0
        and isinstance(probe, dict)
        and probe.get("status") == "passed"
        and probe.get("phase") == "before_close"
        and probe.get("drone_physics_tested") is True
        and isinstance(metrics, dict)
        and metrics.get("status") == "passed"
        and bool(metrics.get("checks"))
        and all(v is True for v in metrics["checks"].values())
    )


def run_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate one drone on one allocated GPU; no learning."
    )
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept_eula or args.gpu < 0 or not 60 <= args.timeout <= 3600:
        parser.error("Require --accept-eula, a nonnegative allocated GPU, and timeout 60–3600")
    root = project_root()
    config_path = args.config or root / "configs/single-drone.json"
    config = DroneCheckConfig.from_dict(json.loads(config_path.read_text()))
    build_path = args.build_report or Path(
        json.loads((root / ".runtime/latest-build.json").read_text())["report"]
    )
    build = json.loads(build_path.read_text())
    if build["status"] != "built":
        raise RuntimeError("A successfully built runtime image is required")
    run = create_run_directory(root / "runs/single-drone")
    (run / "kit-logs").mkdir()
    write_json_atomic(run / "config.json", config.to_dict())
    shutil.copy2(build_path, run / "build-report.json")
    shutil.copy2(build_path.parent / "context-manifest.json", run / "context-manifest.json")
    started = time.perf_counter()
    report = new_report()
    report.update(
        run_id=run.name,
        host_gpu_index=args.gpu,
        image_id=build["image_id"],
        source=probe_source(30).details,
        system=probe_system().details,
        config_sha256=digest(run / "config.json"),
        rendering_validated=False,
    )
    report["gpu_before"] = capture(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv",
        ]
    )
    report["gpu_processes_before"] = capture(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"]
    )
    docker = docker_prefix(args.docker)
    name = "align-drone-" + run.name.lower()
    command = [
        *docker,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--name",
        name,
        "--runtime=nvidia",
        "--gpus",
        f"device={args.gpu}",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "TZ=Asia/Kolkata",
        "--mount",
        f"type=bind,src={run},dst=/output",
        "--mount",
        f"type=bind,src={run / 'kit-logs'},dst=/isaac-sim/kit/logs",
        build["image_id"],
        "-m",
        "align.simulation.single_drone",
        "--config",
        "/output/config.json",
        "--output",
        "/output",
        "--allow-root",
    ]
    report["command"] = command
    write_json_atomic(run / "report.json", report)
    print(f"{report['started_at_ist']} IST Run: {run}\nFollow {run / 'console.log'}", flush=True)
    cleanup = False
    try:
        report["selected_gpu"] = check_gpu_available(
            report["gpu_before"], report["gpu_processes_before"], args.gpu
        )
        report["container_exit_code"] = execute(command, run / "console.log", args.timeout)
        probe_path, metrics_path = run / "probe-result.json", run / "metrics.json"
        probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        report["probe_result"] = probe
        if metrics is not None:
            assessed, _ = assess_saved_run(run)
            if assessed != metrics:
                raise ValueError("Container metrics disagree with host evaluation of raw data")
            report["host_recomputed_metrics"] = True
        report["status"] = (
            "passed" if valid_result(report["container_exit_code"], probe, metrics) else "failed"
        )
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        cleanup = True
    except subprocess.TimeoutExpired:
        report["status"] = "timed_out"
        cleanup = True
    except (OSError, ValueError, RuntimeError) as exc:
        report.update(status="failed", error=str(exc))
        cleanup = True
    finally:
        if cleanup:
            report["cleanup"] = capture([*docker, "rm", "--force", name])
        report["gpu_after"] = capture(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv"]
        )
        finish(run, report, started)
    return 0 if report["status"] == "passed" else 1
