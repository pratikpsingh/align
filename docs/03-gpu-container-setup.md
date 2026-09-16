# Preparing GPU container access on the lab machine

Current status: the pinned Isaac Sim startup test and subsequent OmniDrones calibration have executed on the lab. See [current runtime setup and exact tool versions](07-omnidrones-runtime.md). The acquisition/pending statements below are historical records from initial preparation. Host installation commands were not executed by this assistant.

## Purpose and prerequisites

Docker and NVIDIA container integration now work for the single-device inventory test. The installation instructions below are retained for reproducing setup on another machine. Isaac Sim is not yet installed; the next task is acquiring the candidate simulator image.

Docker packages an application's userspace libraries separately from the host. NVIDIA Container Toolkit supplies GPU access to containers. uv continues to manage our Python project; these tools serve different roles. Docker is an installation choice, not a mathematical requirement of reinforcement learning or a guarantee of simulator compatibility.

The reported host is Ubuntu 24.04 x86_64 with five RTX A4000 GPUs and NVIDIA driver 580.173.02. The driver already answers the host inventory query. This procedure does not require replacing it.

Host installation and Docker-service configuration require administrator privileges. Use a Linux account permitted to run the relevant sudo commands, or have the lab administrator perform them. This is distinct from an NVIDIA website account. If host containers are not permitted, investigate a standalone simulator installation instead; its native-library compatibility still needs checking.

## Install the host components

Use the vendor-maintained installation procedures so repository keys and package instructions remain current:

1. Install Docker Engine using the apt-repository method in [Docker's Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/#install-using-the-apt-repository). Ubuntu 24.04 is listed as supported. Have the administrator review any existing containerd/runc or conflicting packages before removal; do not run a blanket removal command just because the docker executable is missing.
2. Install the production NVIDIA Container Toolkit packages using the apt section of [NVIDIA's installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Include the guide's repository configuration and prerequisite packages. Do not enable experimental repositories for this task.
3. Configure the Docker runtime and restart its service as documented by NVIDIA:

~~~sh
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
~~~

The first command updates Docker's host configuration; the second restarts Docker. Perform them as part of the administrator-managed installation. No changes to .python-version, pyproject.toml, uv.lock, or the ALiGn virtual environment are needed for these host prerequisites.

## Verify one GPU

Choose a GPU allocated for this check. The following example uses host device index 0; change the index if another device is assigned. It downloads a small Ubuntu image if necessary, starts a temporary container, queries the driver, and removes the container on exit:

~~~sh
sudo docker run --rm --runtime=nvidia --gpus device=0 ubuntu:22.04 nvidia-smi
~~~

This adapts NVIDIA's [sample workload](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html) to expose one device. Expected result: the container's query succeeds and shows the selected RTX A4000 with approximately 16 GiB of memory. Device numbering inside a restricted container need not be interpreted as the original host index; use the selected device's identity when recording allocation.

The command does not train a policy, launch Isaac Sim, or establish CUDA/Vulkan/physics correctness. Its useful result is that the GPU is visible inside a container.

Record the installed versions and configured runtime:

~~~sh
docker --version
nvidia-ctk --version
sudo docker info --format '{{json .Runtimes}}'
sudo docker image inspect ubuntu:22.04 --format '{{json .RepoDigests}}'
~~~

Keep the outputs, test date, selected GPU, and exit status with the lab setup records under runs/. Do not treat these as final experiment results. After setup succeeds, capture the actual package versions and image digest rather than describing the installation only as latest.

## Interpreting failures

| Result | What to investigate |
|---|---|
| Command not found | Installation or PATH for that component |
| Permission denied / cannot contact Docker daemon | Account access and whether the Docker service is running |
| NVIDIA runtime is unknown | Toolkit installation and Docker runtime configuration |
| GPU driver/library initialization error | Container-to-host GPU integration; preserve the full error before changing anything |
| Image download fails | Network/registry access; this is not evidence that the GPU is incompatible |
| Query succeeds | GPU inventory works in the container; proceed to a separate simulator launch test |

## What follows

Verify availability of the chosen Isaac Sim runtime/image, pin its exact identity, and build a documented Python/OmniDrones environment around it. The previously identified Isaac Sim 4.1.0/Python 3.10 pairing remains a candidate, not a validated installation. A successful small Ubuntu container does not prove that this historical simulator image is still obtainable or that it works with the host driver.

The next simulation evidence is a minimal application launch followed by one drone responding to known controller commands. Container setup alone does not complete that task.

## Recorded lab result

The user ran the single-GPU test successfully. It showed an RTX A4000 at PCI bus 00000000:01:00.0, driver 580.173.02, 16,376 MiB total VRAM, 15 MiB used, and no reported running processes at that moment. The pulled Ubuntu image digest was sha256:829f6df217bcbae2b371026e81711d1a787c61b2967ad09d015063663ebafbf7. Exact Docker/toolkit package versions have not yet been supplied.

The CUDA Version: 13.0 label describes the driver's supported CUDA level, not proof that CUDA Toolkit 13 is installed or required. See [NVIDIA's nvidia-smi reference](https://docs.nvidia.com/deploy/nvidia-smi/). No CUDA computation, graphics initialization, or drone physics has been tested yet.

## Acquire the candidate simulator image

On the lab machine, the next commands are:

```sh
sudo docker pull nvcr.io/nvidia/isaac-sim:4.1.0
sudo docker image inspect nvcr.io/nvidia/isaac-sim:4.1.0 --format '{{json .RepoDigests}}'
```

The first downloads the version identified by the upstream OmniDrones baseline. The second records the exact image digest for later reproducibility. Neither starts Isaac Sim or changes the ALiGn Python environment. This is a larger download than the Ubuntu test image.

The image reference is documented in [NVIDIA's historical container instructions](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/installation/install_container.html). The lab user has now supplied a successful image inspection for this tag. If the pull reports authorization failure or an unknown manifest, preserve the exact error and reassess access/image availability. Do not substitute a latest tag because its interfaces may differ from the selected OmniDrones version.

After a successful pull, retain the digest and package versions with the setup records. The next runtime test will launch and close a minimal Isaac Sim application on the selected GPU, with persistent logs and explicit runtime/license settings. A downloaded image alone does not establish that this historical simulator works on the host.

## Simulator image received

The supplied image digest is sha256:5bd94fce4318ca2f8bf887c4ce3220bfc1cedd303008bf6d6976b0e9e556d173. Image acquisition is confirmed by the user-provided inspection output; application execution is still unverified. Continue with the [Isaac Sim startup check](04-isaac-sim-smoke.md).
