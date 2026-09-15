# Isaac Sim startup check

This check launches the candidate Isaac Sim 4.1.0 container, performs a small CUDA calculation, advances the application 20 times, and requests shutdown. It does not install OmniDrones, simulate a drone, or train a policy. A lab attempt reached app ready but crashed during full extension cleanup; the revised fast-shutdown attempt is pending lab validation. Local checks exercise launcher and probe control flow without the simulator.

## Image and prerequisites

The lab user supplied this repository digest from Docker image inspection:

```text
nvcr.io/nvidia/isaac-sim@sha256:5bd94fce4318ca2f8bf887c4ce3220bfc1cedd303008bf6d6976b0e9e556d173
```

The launcher uses this exact digest and disables automatic pulls. Docker and the NVIDIA runtime must already work, and this image must be present. See [GPU container setup](03-gpu-container-setup.md). The host needs the existing ALiGn uv environment; the container runs the probe using its own bundled Python. This does not settle the final project Python/dependency configuration.

## Run on the lab machine

Copy or sync the updated ALiGn files, including scripts/, to the lab first. From the align directory:

```sh
uv sync --locked
sudo -v
uv run --locked python scripts/run_isaac_sim_smoke.py --accept-eula --gpu 0
```

Read NVIDIA's license linked from its [container instructions](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/installation/install_container.html) before running with --accept-eula. That flag sets ACCEPT_EULA=Y. The launcher does not set the optional privacy-consent flag. Run uv as your ordinary user; only Docker is invoked with sudo. The sudo authentication step prevents a password prompt from being hidden in the captured log.

Use an allocated GPU index. Exposing one GPU is sufficient for this startup check and does not prevent using the other GPUs for future experiments. The exposed host GPU appears as CUDA device 0 inside the container. Multiple GPUs do not automatically combine their memory or distribute training.

The launcher prints a unique run directory under runs/isaac-sim-smoke/. First startup may take several minutes. The default timeout is 1,200 seconds; --timeout accepts 1–3,600 seconds. Shader caches are not persisted by this minimal check, so its duration is not a warmed-up training benchmark.

## Read the result

Each attempt saves:

- report.json: image identity, exact command, selected GPU, probe hash, timing, exit status, and structured probe result.
- console.log: combined container output, including startup errors and the final probe marker.
- kit-logs/: simulator logs, when written by this image; these may be owned by root.

Use tail -f on the printed run directory's console.log from another terminal to watch progress. A passed report requires a zero container exit code and explicit evidence of completed probe checks. In the default fast mode, the pre-close record is used if shutdown terminates Python before a final record can be printed. Full mode additionally requires a returned close call and final result. The probe checks CUDA availability, exactly one visible CUDA device, a sum of 1,024 ones equal to 1,024, an existing USD scene, and 20 application updates. It records runtime Python, PyTorch, and CUDA versions.

These are application updates, not evidence of 20 physics steps or correct drone dynamics. A successful result also does not validate camera output, OmniDrones compatibility, or learning behavior. Those require subsequent checks.

On failure, preserve report.json and console.log before changing drivers or dependencies. Timeout or interruption triggers removal of only this launcher's uniquely named container. Cleanup results are saved. If authentication expires, cleanup may need manual intervention using the container name in the recorded command. A power cut may leave a running-status report without a final result; this smoke check is rerun, not resumed. Training recovery is a separate planned feature.

## Implementation

[scripts/run_isaac_sim_smoke.py](../scripts/run_isaac_sim_smoke.py) manages the run and artifacts on the host. [scripts/isaac_sim_smoke.py](../scripts/isaac_sim_smoke.py) is a standalone Python 3.10-compatible probe executed by /isaac-sim/python.sh. Simulator extensions are imported after SimulationApp creation, following NVIDIA's [standalone application lifecycle](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/features/sensors_simulation/isaac_sim_sensors_camera.html). No simulator packages are imported by the host launcher.

## Shutdown crash and retry

The supplied run 20260915T192028.862699Z-6a871ddd reached app ready, then had a native segmentation fault inside SimulationApp.close. The previous probe explicitly used fast_shutdown=False. Its final marker was never written, so individual checks cannot be confirmed from that run. This is a shutdown failure, not evidence that downloading or Docker GPU exposure failed. The exact native cause remains unproven.

The revised probe uses fast_shutdown=True by default, matching [NVIDIA's 4.1 API default](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/py/source/extensions/omni.isaac.kit/docs/index.html). That path exits the process rather than individually shutting down every extension. It is a targeted workaround to validate on the lab, not a proven repair of full extension cleanup. Run the command above again after syncing the updated source. Use --shutdown-mode full only to investigate full cleanup separately; reports record the chosen mode.

Progress markers preserve application startup, CUDA checks, and the pre-close state. The report stores last_probe_progress separately from probe_result and records shutdown_return_observed. Nonzero container exit codes always fail, even if checks passed before closing. A failed check also remains a failure if fast shutdown exits with code zero. A pass in fast mode does not establish that full extension cleanup works. Training will need to finish saving artifacts before requesting this shutdown path.

The Kit log mount now targets /isaac-sim/kit/logs, the location observed in the supplied log. It remains confined to this run's kit-logs directory.

## Time display

ALiGn prints start/finish times in Indian Standard Time and uses IST in new run-directory names. Reports contain started_at_ist and finished_at_ist with +05:30 offsets and timezone=Asia/Kolkata, alongside existing UTC fields. Elapsed duration_seconds is unchanged; elapsed time is independent of timezone. The smoke-report schema is now version 2 to identify the new shutdown evidence contract.

The container receives TZ=Asia/Kolkata. Native simulator components may still emit UTC or relative elapsed times; console.log preserves their exact output rather than rewriting forensic evidence. ALiGn's displayed/report times are explicitly IST regardless of the host timezone. Historical reports and directory names are not rewritten.

For example, the original start 2026-09-15T19:20:28.862792+00:00 is 2026-09-16T00:50:28.862792+05:30 in IST. The date changes because IST is five hours and thirty minutes ahead of UTC.
