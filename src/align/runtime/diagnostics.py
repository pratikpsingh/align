"""Read-only host and package probes; these do not validate simulator execution."""

import csv
import hashlib
import io
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal

from align.config import DoctorConfig

Status = Literal["ok", "warning", "error", "not_checked"]
SIMULATION_PACKAGES = ("omni-drones", "isaacsim", "torch", "torchrl", "tensordict")


@dataclass(frozen=True)
class Check:
    """One observation, its interpretation, and structured supporting evidence."""

    name: str
    status: Status
    message: str
    details: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CommandResult:
    status: Literal["ok", "missing", "failed", "timeout"]
    stdout: str = ""
    error: str | None = None


def run_command(args: Sequence[str], timeout: float, *, cwd: Path | None = None) -> CommandResult:
    """Run a fixed argument list without a shell or unbounded waiting."""
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult("missing", error=f"{args[0]} is not available on PATH")
    except subprocess.TimeoutExpired:
        return CommandResult("timeout", error=f"{args[0]} exceeded {timeout:g} seconds")
    except OSError as exc:
        return CommandResult("failed", error=str(exc))
    if result.returncode:
        error = result.stderr.strip() or result.stdout.strip() or "No diagnostic output"
        return CommandResult("failed", error=f"exit {result.returncode}: {error[:1000]}")
    return CommandResult("ok", stdout=result.stdout.strip())


def parse_gpu_csv(text: str) -> list[dict]:
    """Parse NVIDIA index/name/VRAM MiB/driver rows; never sum per-GPU VRAM."""
    devices = []
    indices = set()
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) != 4:
            raise ValueError("Expected four NVIDIA fields: index, name, memory, driver")
        index_text, name, memory, driver = (field.strip() for field in row)
        index = int(index_text)
        if index < 0 or index in indices or not name or not driver:
            raise ValueError("Invalid or repeated NVIDIA device identity")
        indices.add(index)
        memory_mib = None if memory in {"N/A", "[N/A]", "[Not Supported]"} else int(memory)
        if memory_mib is not None and memory_mib <= 0:
            raise ValueError("GPU memory must be positive when reported")
        devices.append(
            {
                "index": index,
                "name": name,
                "memory_total_mib": memory_mib,
                "driver_version": driver,
            }
        )
    return devices


def probe_nvidia(timeout: float, require_nvidia: bool) -> Check:
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout,
    )
    devices = []
    reason = result.error
    if result.status == "ok":
        try:
            devices = parse_gpu_csv(result.stdout)
        except ValueError as exc:
            reason = f"Could not interpret NVIDIA output: {exc}"
        if not devices and reason is None:
            reason = "NVIDIA query returned no devices"
    details = {"query_status": result.status, "devices": devices, "reason": reason}
    if reason is not None:
        return Check(
            "nvidia",
            "error" if require_nvidia else "warning",
            "NVIDIA inventory unavailable; CPU diagnostics remain usable.",
            details,
        )
    return Check(
        "nvidia",
        "ok",
        f"NVIDIA driver reports {len(devices)} device(s); CUDA execution is untested.",
        details,
    )


def probe_system() -> Check:
    details = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_model": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "ram_total_bytes": None,
        "ram_available_bytes": None,
        "memory_scope": "Host-reported RAM; container/job limits are not measured.",
    }
    notes = []
    if platform.system() == "Linux":
        try:
            details["distribution"] = platform.freedesktop_os_release().get("PRETTY_NAME")
        except OSError as exc:
            notes.append(f"OS distribution unavailable: {exc}")
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    details["cpu_model"] = line.split(":", 1)[1].strip()
                    break
            memory = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable"}:
                    amount, unit = value.split()
                    if unit != "kB":
                        raise ValueError("Unknown /proc/meminfo unit")
                    memory[key] = int(amount) * 1024
            details["ram_total_bytes"] = memory.get("MemTotal")
            details["ram_available_bytes"] = memory.get("MemAvailable")
        except (OSError, ValueError) as exc:
            notes.append(f"CPU/RAM detail unavailable: {exc}")
    else:
        notes.append("Detailed RAM inventory is currently implemented for Linux only.")
    details["notes"] = notes
    return Check("system", "ok", "Recorded host OS, CPU, and available RAM information.", details)


def probe_packages() -> Check:
    versions = {}
    for name in SIMULATION_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return Check(
        "simulation_packages",
        "not_checked",
        "Recorded package metadata only; simulator compatibility and execution are untested.",
        {
            "versions": versions,
            "scope": "Active Python distributions only; external Isaac runtimes may not appear.",
            "simulator_imported": False,
            "simulation_tested": False,
        },
    )


def probe_source(timeout: float) -> Check:
    """Identify actual imported source, including edits when there is no Git commit."""
    package_dir = Path(__file__).resolve().parents[1]
    files = sorted(package_dir.rglob("*.py"))
    entries = []
    digest = hashlib.sha256()
    for path in files:
        content = path.read_bytes()
        relative = path.relative_to(package_dir).as_posix()
        checksum = hashlib.sha256(content).hexdigest()
        entries.append({"path": relative, "bytes": len(content), "sha256": checksum})
        digest.update(f"{relative}\0{checksum}\n".encode())
    details = {
        "package_sha256": digest.hexdigest(),
        "python_source_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
        "git_revision": None,
        "git_dirty": None,
        "project_files": {},
    }
    # Editable src-layout installs can identify their project. Wheels need no checkout.
    project = package_dir.parent.parent
    if package_dir.parent.name == "src" and (project / "pyproject.toml").is_file():
        for name in ("pyproject.toml", "uv.lock", ".python-version"):
            path = project / name
            if path.is_file():
                details["project_files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        head = run_command(["git", "rev-parse", "--verify", "HEAD"], timeout, cwd=project)
        dirty = run_command(
            ["git", "status", "--porcelain", "--untracked-files=normal"], timeout, cwd=project
        )
        details["git_revision"] = head.stdout if head.status == "ok" else None
        details["git_dirty"] = bool(dirty.stdout) if dirty.status == "ok" else None
        details["git_note"] = head.error or dirty.error
    return Check("source", "ok", "Fingerprint recorded for the imported Python source.", details)


def collect_checks(config: DoctorConfig, run_dir: Path) -> Iterator[Check]:
    """Yield each completed probe so the caller can retain partial progress."""
    yield Check(
        "python",
        "ok",
        f"ALiGn is running in Python {platform.python_version()}.",
        {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "in_virtual_environment": sys.prefix != sys.base_prefix,
            "align_version": metadata.version("align"),
            "simulation_python_compatibility": "not_validated",
        },
    )
    yield probe_system()
    uv = run_command(["uv", "--version"], config.timeout_seconds)
    yield Check(
        "uv",
        "ok" if uv.status == "ok" else "warning",
        uv.stdout or "uv is unavailable to this process; the installed CLI still runs.",
        {"version_output": uv.stdout or None, "reason": uv.error},
    )
    disk = shutil.disk_usage(run_dir)
    yield Check(
        "storage",
        "ok",
        "Created the output directory and initial report on the selected filesystem.",
        {"path": str(run_dir), "total_bytes": disk.total, "free_bytes": disk.free},
    )
    yield probe_nvidia(config.timeout_seconds, config.require_nvidia)
    yield probe_packages()
    yield probe_source(config.timeout_seconds)
