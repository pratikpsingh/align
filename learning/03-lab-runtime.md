# Understanding the lab report and simulator compatibility

## What the report establishes

The lab successfully ran ALiGn's diagnostic and queried five NVIDIA RTX A4000 devices. Each reports approximately 16 GiB of VRAM. The host reports about 503 GiB of RAM and 192 logical CPUs.

That is useful evidence for trying GPU simulation. It is not yet a flight test. The report explicitly says simulation_validation=not_performed, and it found no simulator/learning-package metadata in the active ALiGn environment.

The report is a snapshot: free memory, GPU availability, and another user's workload can change. Before running a simulation, we will select an available GPU explicitly and record that selection.

## Why five GPUs do not give one 80 GiB GPU

Each GPU has its own memory. If a simulation runs on one A4000, its allocations must fit that device's available VRAM. The memory on the other four cards is not automatically available to it.

Think of five separate workbenches. Five experiments may use separate benches, but one experiment does not automatically gain a bench five times larger. Later, separate training seeds could run on separate GPUs if those devices are available. Making one learning run span multiple GPUs needs additional software and validation.

For the first drone test, one GPU is enough to investigate whether the runtime launches and motion behaves correctly. We will measure memory before increasing the number of parallel environments.

## The pieces of the software stack

| Component | Role in our project |
|---|---|
| NVIDIA driver | Lets the operating system and GPU applications communicate with the hardware |
| Isaac Sim | Provides simulation infrastructure, including physics and rendering |
| Isaac Lab | Supplies robot-learning infrastructure used by the selected OmniDrones version |
| OmniDrones | Supplies drone models, control utilities, and learning-environment building blocks |
| PyTorch | Provides tensor computations and neural-network learning tools |
| TorchRL / TensorDict | Supply RL-related interfaces and structured tensor data used upstream |
| uv | Manages the Python project/environment and its dependencies |
| ALiGn | Implements our task, policy training, evaluation, and reproducibility requirements |

A Python package being installed is different from all these components working together. Some packages contain or load native compiled libraries. Their expected Python and library versions must match.

## Why we are checking versions before installing

The [upstream OmniDrones project](https://github.com/btx0424/OmniDrones) identifies Isaac Sim 4.1.0 and Python 3.10 as its baseline. It also reports maintenance difficulties. Our diagnostic currently runs under Python 3.12, so it cannot simply be assumed to share the eventual simulator environment.

Newer simulator versions can rename interfaces or change how physical forces are applied. A version migration therefore needs more than successful imports. A drone that appears in a scene but responds incorrectly to motor forces would undermine every later learning experiment.

The current plan is to investigate a pinned version combination, test a minimal simulator application, and then verify one drone's response to known commands. A detailed candidate assessment is in [the runtime plan](../plans/06-lab-runtime-assessment.md). Those versions are not yet a tested installation recipe.

## A container and a virtual environment solve different problems

A virtual environment isolates Python packages. A container also packages much of the application's operating-system library environment. A container can help keep an older simulator runtime separate from the lab host's software, while uv still manages our Python project within the chosen arrangement.

A container still uses the host kernel and GPU driver. Docker being installed does not by itself establish that the current account can access the GPU through it. We need to establish both available runtime installations and GPU-container access before selecting the installation procedure.

## What we will observe next

First, a minimal simulator application must open and close successfully. Next, a drone must load, advance through simulated time, expose its position and velocity, and reset correctly.

Then we will issue a known command through a controller. For example, an upward velocity target should produce movement in the declared upward direction, within the controller's limits. A controller translates this target into lower-level actuation; a future learned policy will choose targets/actions based on observations.

This experiment teaches the observation/action loop and verifies the physical interface before reinforcement learning adds another source of uncertainty. Its results will be logs, trajectory samples, timing/memory measurements, and a short visualization when rendering has been validated.

## Check your understanding

- Why start with one GPU? To establish working physics and measure resource use with a clear device allocation.
- Does a successful NVIDIA query prove Isaac Sim works? No; it establishes driver inventory, not simulator execution.
- Can changing the Python version file alone make the whole project compatible? No; code, dependencies, and native libraries also need matching versions and checks.
- Does headless simulation mean CPU simulation? No; it means running without an interactive application window, while GPU computation/rendering may still be required.

## Preparing container access

The lab has now passed the single-GPU Ubuntu container inventory test. Docker and the NVIDIA integration work for that test; Isaac Sim acquisition and execution are next. The [lab setup procedure](../docs/03-gpu-container-setup.md) explains the administrator-managed installation and expected output.

Here, account means the Linux login used on the lab computer, such as pratiksingh. sudo allows an authorized login to perform administrator operations. This is needed for host package installation and service configuration; it is not a requirement to run ordinary Python code in our uv environment. If your login cannot perform those operations, the lab administrator can install the host prerequisites.

The test container runs nvidia-smi. If it shows the selected A4000, we have checked that the container can see that GPU. We have not yet checked that a drone can move correctly. We will use a separate simulator test for that, with its own logs and results.

## Understanding the successful container output

The message Unable to find image locally was normal on the first invocation: Docker downloaded Ubuntu 22.04, then ran the command inside it. Pull complete and the following NVIDIA table show that the operation advanced beyond the download.

The table showed one RTX A4000 because the command requested one device. The 15 MiB / 16376 MiB entry was used memory versus total memory at that moment; it is not a permanent allocation or a measurement of training memory. The digest identifies the downloaded image content more precisely than its human-readable tag.

The CUDA Version field describes driver capability. It does not select our PyTorch version or demonstrate a CUDA calculation. Our next small task is downloading the candidate Isaac Sim image and recording its digest; after that, a separate simulator-launch test will check more than inventory. The [operational guide](../docs/03-gpu-container-setup.md) contains the commands.
