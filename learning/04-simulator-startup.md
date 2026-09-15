# Understanding simulator startup

Before asking eight UAVs to learn a formation, we need to know that the lab computer can open and close the simulator reliably. A smoke test is a small check of this basic operation. It helps distinguish installation errors from later errors in drone control or reinforcement learning.

## Image, container, and digest

An image is a packaged software environment. A container is a running instance of it. The tag 4.1.0 is a human-readable version label; the sha256 digest supplied by Docker identifies the downloaded image content precisely. Saving the digest lets us request that same image later. Downloading it proves availability, not that simulation works.

## Why two Python environments appear

uv manages ALiGn's host environment and runs the launcher. Isaac Sim brings its own Python and native libraries inside the container. The small probe uses that bundled runtime without importing ALiGn. This separation lets us test the candidate simulator before choosing the full project's compatible dependency set. It is temporary compatibility validation, not the final OmniDrones installation.

## What happens when you run the check

The host launcher creates a unique results directory and starts the pinned container on one selected GPU. Inside it, SimulationApp opens the simulator. Only then does the probe import simulator extensions and PyTorch. It adds 1,024 ones on the GPU: the expected answer is 1,024. This establishes that a real CUDA operation works, beyond merely listing the GPU with nvidia-smi.

USD is the simulator's scene representation: the probe checks that a scene exists. It advances the application 20 times, closes it, and prints a structured result. The launcher saves that result together with the exit code and logs. Startup or shutdown failures must not produce a passing report.

Application updates let the simulator process work; they do not establish that a UAV has moved or that physics is correct. Later, a controlled drone test can ask a concrete question such as whether a commanded thrust changes its altitude as expected. Only after dependable observations and actions exist can reinforcement learning connect a policy's action to a measured reward.

## Understanding the evidence

A passed report supports basic startup, CUDA arithmetic, scene creation, application updates, and orderly return from shutdown. It does not support claims about formation accuracy or training speed. The report explicitly says drone physics was not tested. CPU unit tests check result handling and timeout cleanup with fake subprocess results; they are not simulator validation.

The two scripts in scripts/ correspond to the host and container responsibilities. Follow [the operational instructions](../docs/04-isaac-sim-smoke.md) on the lab machine. Read report.json first, then console.log when the status is failed or timed_out. Preserve unsuccessful attempts too: they explain how the environment was established.
