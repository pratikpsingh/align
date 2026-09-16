# Pinned OmniDrones runtime

## Scope and current evidence

This runtime connects ALiGn to Hummingbird drones through OmniDrones. It contains deterministic one- and four-drone controller checks, not an RL learner. The [single-drone guide](08-single-drone-control.md) records its accepted baseline; the [multi-drone guide](10-multi-drone-construction.md) defines the construction probe.

On 2026-09-16, the derived image built successfully in `runs/runtime-build/20260916T014225.397441IST-4f57661c/`. The build checked the additional packages' dependency closure and verified the vendor versions. This establishes installation, not flight correctness. The existing base-image startup evidence remains `20260916T010519.212064IST-5253d3c9`; its probe hash still matches.

## Two interpreters, one compatible ALiGn package

Host orchestration stays on the `.python-version` Python 3.12 pin, managed by the project `uv.lock`. The shared `src/align` package now permits Python >=3.10. Its Python-3.11-only `datetime.UTC` uses were replaced with `timezone.utc`; the CPU suite was run on Python 3.12 and 3.10.21. The actual vendor interpreter remains 3.10.14. This does not make native simulator extensions importable from host Python.

The derived image installs a built ALiGn wheel into the vendor interpreter. Simulator imports are confined to the simulator probe run functions and happen after application creation. Configuration, command conversion, metrics, orchestration, and ordinary CPU tests import no simulator or PyTorch. There are no manual sys.path changes. An upstream editable installation locates OmniDrones' non-Python USD/YAML assets through normal packaging.

## Exact identities and ownership

| Component | Pin and ownership |
|---|---|
| Isaac Sim | 4.1.0 base image `nvcr.io/nvidia/isaac-sim@sha256:5bd94fce4318ca2f8bf887c4ce3220bfc1cedd303008bf6d6976b0e9e556d173` |
| Python | Vendor 3.10.14 |
| PyTorch / CUDA build | Vendor `2.2.2+cu118` / 11.8 |
| NumPy / SciPy / matplotlib | Vendor 1.26.0 / 1.10.1 / 3.8.4 |
| OmniDrones | [9ce7c2028b71be64d7e748c31f685cd3b54afe27](https://github.com/btx0424/OmniDrones/tree/9ce7c2028b71be64d7e748c31f685cd3b54afe27), with the recorded view-API patch |
| TorchRL / TensorDict | Additional locked wheels 0.3.1 / 0.3.2 |
| cloudpickle / wheel / uv / fontTools | Additional locked wheels 3.0.0 / 0.43.0 / 0.9.28 / 4.53.1 |
| Isaac Lab | Inspected v1.1.0, commit `454905bf08c374b6162e6c4c64a320aa6261fbe6`; **not installed** |

`runtime/stack.json` is the machine-readable selection. `runtime/vendor-packages.json` captures the complete vendor distribution metadata, including pre-existing duplicate distribution records. `runtime/additions.lock` contains hashes for every added package. **The host `uv.lock` does not reproduce the simulator stack.** The stack consists of the immutable base image, the extra wheel lock, pinned source, patch, and ALiGn wheel.

The dependency selection follows the pinned [OmniDrones installation metadata](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/setup.py) and [version table](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/docs/source/installation.rst). The basic drone/controller imports do not need Isaac Lab. Its sensor integrations are imported by other upstream tasks, which this runtime does not expose.

## Deliberately limited dependency closure

The lock is generated with `uv pip compile --no-deps`: it lists the complete **additional** dependency set for this audited path. NumPy, PyTorch, packaging, PyYAML, and native simulator libraries come from the image. `runtime/verify_stack.py` checks every non-extra requirement of the added distributions against installed metadata and verifies exact vendor versions. Installation uses `--no-deps`, `--require-hashes`, and a local wheelhouse; the image build has networking disabled. It cannot silently resolve a replacement PyTorch.

OmniDrones is installed with `--no-deps`. Its broad upstream metadata also declares training/visualization packages such as wandb, moviepy, pandas, and av, which are not all installed here. Therefore a whole-environment `pip check` is not the acceptance criterion, and the upstream training/task scripts are not supported by this image. Expand and re-audit the dependency contract when those capabilities are implemented. No online logging account is needed.

The host downloads wheels using a uv-managed, pinned pip 25.3 tool; uv manages installation in the container. Downloads require network access. Once the prepared context is retained, the Docker build itself is offline. Preserve the wheelhouse/context or an exported derived image if upstream availability is a concern.

## Compatibility patches and licenses

The pinned upstream overrides `get_world_poses` without the `usd` keyword that Isaac Sim 4.1's constructor supplies. `runtime/patches/omnidrones-isaac41-views.patch` adds that accepted keyword to both overrides. It preserves upstream pose handling; it does not select between USD and physics pose sources. This adapter is for the tested physics path, not a general repair of every view API. NVIDIA's files are unchanged. The upstream [troubleshooting page](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/docs/source/troubleshooting.md) documents the original error, but its suggested vendor-file edit is not used.

runtime/patches/omnidrones-lazy-runtime-imports.patch prevents importing every unrelated upstream task when ALiGn requests the IsaacEnv base class. It also defers einops and tqdm until the unused upstream rendering callback is instantiated. Direct task-module imports and callback behavior are preserved when their declared optional packages are installed. The audited ALiGn path uses its own task and does not advertise the other upstream tasks or rendering callback as supported.

OmniDrones is MIT licensed, copyright 2023 Botian Xu, Tsinghua University. The complete source archive and LICENSE are retained in the image; the patch is separate. The selected USD/YAML assets are from that repository. The original USD references an external NVIDIA MDL plastic material; the offline probe makes an explicit untextured copy and verifies nonmaterial properties remain unchanged (see the control guide). No separate asset-license notice was found for the Hummingbird files; preserve upstream attribution and do not infer rights to arbitrary external assets. NVIDIA's image retains its own license. Isaac Lab's inspected source uses BSD-3-Clause; it is not included in the runtime.

The patch runtime/patches/omnidrones-cloned-body-views.patch explicitly sets reset_xform_properties=False for the multirotor base-link and rotor views. It exposes prepare_contact_sensors and disable_stablization on multirotor initialization while retaining the upstream True defaults for existing callers. The vector task passes False for both and disables contact tracking on the base-link motion view. It prepares contact-report and sleep-threshold schemas on source bodies before cloning, then creates a separate read-only RigidContactView. This is necessary because Isaac Sim 4.1's RigidPrimView forces late rigid-body preparation whenever track_contact_forces is True, even if prepare_contact_sensors is False. The observed batch failures and the accepted native result are recorded in the [vector-task guide](13-vectorized-task-environment.md). NVIDIA's [view API](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/py/source/extensions/isaacsim.core.prims/docs/index.html) documents disabling transform rewriting for cloned objects; the exact 4.1 constructor source is retained with the failed run evidence.

## Build and run

From the discovered `align/` checkout, as the ordinary user:

```sh
uv sync --locked
sudo -v
uv run --locked python scripts/inspect_isaac_runtime.py
uv run --locked python scripts/build_omnidrones_runtime.py
sudo -v
uv run --locked python scripts/run_single_drone.py --accept-eula --gpu 0
# After rebuilding with the multi-drone source:
uv run --locked python scripts/run_multi_drone.py --accept-eula --gpu 0
# After rebuilding with the vector task source:
uv run --locked python scripts/run_vector_task.py --accept-eula --gpu 0
```

Use `--docker direct` on each helper only when this session has Docker socket access. `sudo -v` authenticates in your terminal; the scripts use noninteractive `sudo -n` so a captured password prompt cannot hang a job. Do not run uv as root. The `--accept-eula` flag has the same meaning as in the [base startup guide](04-isaac-sim-smoke.md).

The build archives the exact source commit into a unique context, applies the patch there, downloads hash-checked Python 3.10/Linux wheels, builds the ALiGn wheel, and records hashes of all inputs. It does not alter sibling repositories or the pinned upstream source cache. The built image has a unique tag and is launched by immutable image ID, recorded in the report. `.runtime/latest-build.json` is only a convenience pointer. Use `--build-report runs/runtime-build/<id>/report.json` to select a particular build.

For a split workflow without Docker access:

```sh
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only
# In a terminal with Docker access, using the printed directory:
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py --prepared-run runs/runtime-build/<id>
```

Prepared input hashes are checked before building. Rebuild after changing simulator code; a previously built image contains its saved wheel, not your live edits. A source checkout with new analysis code does not change an old image. Probe JSON files use explicit mode 0644 so the ordinary host user can read files created by the root container; host-private reports retain their existing mode.

## Records and backup

Inventory: `runs/runtime-inventory/<id>/` contains Docker server/client, image metadata, vendor distributions, exact commands, and logs. Recorded versions: Docker Engine/client 29.8.1, containerd 2.3.5, runc 1.5.1, NVIDIA Container Toolkit/libnvidia-container 1.20.0. Host GPU 0 was idle with 16 MiB used before integration; availability is checked again for each launch and is not a permanent reservation.

Builds: runs/runtime-build/<id>/ contains the context, manifest, build log, source identity, exact build command, and resulting image metadata. One-drone flights use runs/single-drone/<id>/; group construction uses runs/multi-drone/<id>/; cloned task checks use runs/vector-task/<id>/. Run IDs and ALiGn displays use IST; UTC fields remain available. Native Kit logs retain their native timestamps.

`runs/` and `.runtime/` are ignored. Back up required run folders separately from Git, including unsuccessful attempts, the successful build context, source/asset hashes, and trajectories. Do not delete the earlier failed full-cleanup startup run. A copied virtual environment is not a reproducible installation.
