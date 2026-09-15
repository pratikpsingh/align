# From simulator startup to drone control

The simulator smoke test passed. That answers a small but important question: can this computer open the selected simulator, execute a CUDA calculation, advance the application, and exit through the selected fast-shutdown path? It does not yet answer whether our drone model flies correctly.

## The layers we are connecting

Docker supplies the packaged Isaac Sim environment. Isaac Sim provides the scene and physics runtime. OmniDrones adds drone models, controllers, and interfaces for parallel simulation. ALiGn will define our observations, actions, rewards, task rules, learning algorithm, and research measurements. A working lower layer does not automatically validate the next layer.

The host ALiGn utilities currently use Python 3.12 or newer, while the successful simulator probe used Python 3.10.14 inside the container. They are separate processes. Their packages do not automatically transfer between them. The next agent must make this boundary reproducible with uv without accidentally replacing the simulator's working PyTorch libraries.

## Why one drone comes before a learning swarm

Consider commanding a small positive vertical velocity. We need to know which axis is vertical, whether the command is in the world or body frame, its units, its limits, and how often the controller applies it. We should then observe movement in the expected direction. Reset must also reset controller state, not just move the visible drone. Otherwise a learning algorithm could receive inconsistent transitions even though the scene looks plausible.

A controller translates a target such as velocity into lower-level actuation. A learned policy will eventually choose that target. We first test the controller with deterministic commands so unexpected movement can be investigated without also debugging policy learning.

The first drone check should record time, target and measured position/velocity, orientation, actions, and reset outcomes. Plot measured motion against the command and inspect finite values and documented tolerances. This is physics evidence. Twenty application updates in an empty scene were runtime evidence. Keep these two claims separate.

## Understanding the successful report

The observed CUDA sum was 1024.0; the runtime reported Python 3.10.14, PyTorch 2.2.2+cu118, and CUDA build 11.8. Total container time was approximately 57.2 seconds. Those numbers describe this setup check, not expected training speed.

The final probe_result is null because fast shutdown can end Python inside close(). The pre-close record reports successful checks, and Docker reported exit 0. That is the intended pass condition. The earlier full-cleanup crash remains unresolved; it is not necessary to pretend it disappeared. Future checkpoints and measurements must be saved before calling close().

ALiGn reports IST with an explicit +05:30 offset. The supplied native simulator log still displays UTC despite the requested timezone. Both can be read together using the offset; preserve raw logs rather than changing their timestamps after a run.

## Continuing on the lab

Use the [lab handoff and continuation prompt](../docs/06-lab-handoff.md). It records the exact image, source boundaries, next acceptance criteria, and the longer research roadmap. After the one-drone check, we can build formation geometry and task behavior, then recurrent learning and recovery. Each implementation should add a useful explanation and measured evidence.
