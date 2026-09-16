"""Record vendor packages and Docker metadata without starting simulation."""

import argparse
import json
import subprocess

from align.artifacts import as_ist, create_run_directory, utc_now, write_json_atomic
from align.runtime.drone_runtime import project_root

IMAGE = (
    "nvcr.io/nvidia/isaac-sim@sha256:"
    "5bd94fce4318ca2f8bf887c4ce3220bfc1cedd303008bf6d6976b0e9e556d173"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", choices=("sudo", "direct"), default="sudo")
    args = parser.parse_args()
    root = project_root()
    run = create_run_directory(root / "runs" / "runtime-inventory")
    docker = ["sudo", "-n", "docker"] if args.docker == "sudo" else ["docker"]
    commands = {
        "docker-version": [*docker, "version", "--format", "{{json .}}"],
        "docker-info": [*docker, "info", "--format", "{{json .Runtimes}}"],
        "image": [*docker, "image", "inspect", IMAGE],
        "vendor": [
            *docker,
            "run",
            "--rm",
            "--pull=never",
            "--entrypoint",
            "/isaac-sim/python.sh",
            IMAGE,
            "-c",
            "import json,sys,importlib.metadata as m; "
            "print(json.dumps({'python':sys.version,'executable':sys.executable,"
            "'packages':sorted([{'name':d.metadata['Name'],'version':d.version} "
            "for d in m.distributions()],key=lambda d:d['name'].lower())}))",
        ],
    }
    report = {"started_at_utc": utc_now(), "image": IMAGE, "commands": {}, "status": "running"}
    report["started_at_ist"] = as_ist(report["started_at_utc"])
    write_json_atomic(run / "report.json", report)
    print(f"Inventory: {run}", flush=True)
    for name, command in commands.items():
        result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
        (run / f"{name}.log").write_text(result.stdout + result.stderr)
        report["commands"][name] = {"argv": command, "exit_code": result.returncode}
        if result.returncode == 0:
            write_json_atomic(run / f"{name}.json", json.loads(result.stdout))
        write_json_atomic(run / "report.json", report)
    report["status"] = (
        "passed" if all(c["exit_code"] == 0 for c in report["commands"].values()) else "failed"
    )
    report["finished_at_utc"] = utc_now()
    report["finished_at_ist"] = as_ist(report["finished_at_utc"])
    write_json_atomic(run / "report.json", report)
    print(f"Result: {report['status']}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
