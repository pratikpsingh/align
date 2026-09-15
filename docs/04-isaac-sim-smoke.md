# Isaac Sim startup check

This check launches the candidate Isaac Sim 4.1.0 container, performs a small CUDA calculation, advances the application 20 times, and closes it. It does not install OmniDrones, simulate a drone, or train a policy. GPU execution is pending lab validation; local checks exercise only the launcher.

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

Use tail -f on the printed run directory's console.log from another terminal to watch progress. A passed report requires a zero container exit code and the probe's explicit result after shutdown. The probe checks CUDA availability, exactly one visible CUDA device, a sum of 1,024 ones equal to 1,024, an existing USD scene, and 20 application updates. It records runtime Python, PyTorch, and CUDA versions.

These are application updates, not evidence of 20 physics steps or correct drone dynamics. A successful result also does not validate camera output, OmniDrones compatibility, or learning behavior. Those require subsequent checks.

On failure, preserve report.json and console.log before changing drivers or dependencies. Timeout or interruption triggers removal of only this launcher's uniquely named container. Cleanup results are saved. If authentication expires, cleanup may need manual intervention using the container name in the recorded command. A power cut may leave a running-status report without a final result; this smoke check is rerun, not resumed. Training recovery is a separate planned feature.

## Implementation

[scripts/run_isaac_sim_smoke.py](../scripts/run_isaac_sim_smoke.py) manages the run and artifacts on the host. [scripts/isaac_sim_smoke.py](../scripts/isaac_sim_smoke.py) is a standalone Python 3.10-compatible probe executed by /isaac-sim/python.sh. Simulator extensions are imported after SimulationApp creation, following NVIDIA's [standalone application lifecycle](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/features/sensors_simulation/isaac_sim_sensors_camera.html). No simulator packages are imported by the host launcher.
