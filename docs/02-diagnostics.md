# Diagnostic command and report reference

## Commands and options

Run from the project root:

~~~sh
uv run --locked align doctor
uv run --locked align doctor --output-dir "runs/laptop check"
uv run --locked align doctor --require-nvidia
uv run --locked align doctor --timeout 10 --log-level WARNING
uv run --locked python -m align --version
~~~

The output directory is resolved relative to the current working directory, with user-home expansion supported. Defaults are runs/diagnostics, a five-second timeout per external command, optional NVIDIA inventory, and INFO console logging. Timeout must be finite, positive, and no greater than 30 seconds. Unknown options are errors.

The timeout is per command, not a total diagnostic deadline. Calls include uv, nvidia-smi, and Git when a source checkout is available. Arguments are passed without a shell. The command does not install drivers, download assets, import simulator packages, or change machine configuration.

## What is checked

| Check | Evidence | Interpretation |
|---|---|---|
| python | Interpreter version/path, virtual-environment status, ALiGn package version | Confirms which Python executed the command |
| system | OS/release/architecture, CPU model/count, Linux host RAM | An inventory; container/scheduler resource limits are not measured |
| uv | uv --version result | Records the uv executable visible to this process |
| storage | Output path, filesystem total/free bytes, successful initial report creation | Confirms basic output access; does not estimate future training capacity |
| nvidia | Per-device index, model, VRAM in MiB, driver, query status/reason | Driver inventory only; no CUDA computation or render test |
| simulation_packages | Metadata for omni-drones, isaacsim, torch, torchrl, tensordict | Packages in the active interpreter; external runtimes may be invisible |
| source | Imported Python-file hashes/bytes, aggregate fingerprint, project-file hashes, Git revision/dirty state when available | Identifies code used by this invocation, including an uncommitted scaffold |

Python distribution names are not always the same as import names. A null version means metadata was not found; it does not mean version zero. A visible package may still fail to import or run. A visible NVIDIA GPU may still be incompatible with a simulator build. The report therefore always records simulation_validation=not_performed.

No minimum VRAM, RAM, disk size, or driver version is certified yet. Those requirements depend on the runtime and workload to be validated. GPU memory is never summed into a fictitious single-device budget.

## Files and lifecycle

~~~text
runs/diagnostics/<IST timestamp>IST-<unique suffix>/
  report.json
  doctor.log
  events.jsonl
~~~

report.json uses schema_version=1. It contains run_id, command, resolved configuration, start/finish IST timestamps (with +05:30 offsets), corresponding UTC timestamps, duration_seconds, exit_code, status, simulation_validation, and the list of checks. Each check has name, status, message, and details. Consumers should use named fields rather than list positions.

Check statuses are ok, warning, error, and not_checked. An unavailable optional GPU query is a warning. With --require-nvidia it becomes an error. Metadata-only simulator inspection remains not_checked even if every listed distribution is present.

The command writes an initial running report and refreshes it after completed probes. Successful completion writes completed; an unmet requested check writes requirements_unmet. Caught failures write failed, and handled keyboard interruption writes interrupted. A hard-killed process may leave running with no finish time: interpret it as unfinished, not successful.

The report is replaced using a same-directory temporary file after JSON serialization and file synchronization. Failed serialization/publication preserves the previous report where the filesystem permits. This small artifact mechanism does not implement training checkpoints, directory-level power-loss durability, or disk-loss backup.

doctor.log and console messages use IST timestamps with explicit +05:30 offsets. New run directory names end their timestamp with IST; older UTC run names are preserved. events.jsonl has one JSON object per event, including full run ID, timestamp_ist, timestamp_utc, level, event name, and message; check events include the check name/status. Console verbosity does not discard INFO events from the files. Each run owns and closes its logging handlers.

The duration covers diagnostic work through final report preparation, excluding the final report write and final console/log output. It is not a training or inference measurement. Source bytes count the imported package's Python files; dependencies, tests, documentation, weights, and runtime memory are different quantities.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Diagnostic completed without a failed required check; warnings and untested simulation may remain |
| 1 | Required NVIDIA check failed, a probe unexpectedly failed, or artifacts could not be accessed/saved |
| 2 | Invalid command/options; no diagnostic run is started |
| 130 | Keyboard interruption handled during probing |

If the initial output directory/report cannot be created, only the terminal error may be available. If storage fails later, an earlier report may remain incomplete. Never infer a successful run solely from a directory existing.

## Source identity and sharing

The source fingerprint is SHA-256 over sorted relative package paths and their content hashes. Per-file entries allow inspection. In an editable src-layout checkout, pyproject.toml, uv.lock, and .python-version hashes are also recorded. Git revision may be null before the first commit or outside a checkout. A wheel installation can still fingerprint the actual imported Python files.

No full source archive or unrestricted environment-variable dump is saved by doctor. Reports include useful local paths such as the interpreter/output location and hardware/software details. Review a report before sharing it externally. Preserve a matching source checkout when a report will support a reproducibility claim.

Diagnostic records are setup evidence. Future training runs will need additional configuration, seeds, learning state, metrics, and recovery semantics. Do not interpret a diagnostic report as a flight result.
