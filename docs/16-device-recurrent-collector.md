# Device recurrent rollout collector

## Status and purpose

ALiGn now connects the accepted four-world OmniDrones task to the shared
recurrent actor, centralized critic, and sequence rollout buffer. The collector
runs inference and storage on one CUDA device, preserves recurrent state across
ordinary steps, resets only completed environments, and keeps the final
pre-reset critic value needed at a time limit.

This is a collection acceptance test. It initializes a policy from a recorded
seed and performs no optimizer update or training.

## Resolved contract

The default command combines these explicit configurations:

- `configs/recurrent-collector-task.json`: four environments and an
  80-step episode limit;
- `configs/recurrent-policy.json`: shared 55-input actor, centralized
  80-input critic, four bounded action values, and 256-value LSTM state;
- `configs/recurrent-rollout.json`: 128 collection steps and
  16-step training chunks;
- `configs/recurrent-collector-probe.json`: policy seed 29 and one
  forced safety termination for exercising both boundary types.

One accepted rollout contains 512 environment transitions and 2,048 agent
transitions. Actor memory has one lane per environment and drone. Critic memory,
team reward, value, advantage, and return have one lane per environment. The
team reward is the mean of the four per-agent reward totals.

## Boundary handling

The collector evaluates the final next state before resetting an environment.
It then applies these rules:

| Boundary | Stored bootstrap | GAE crosses boundary | Memory for next frame |
|---|---:|---:|---:|
| ordinary step | critic value | yes | carried |
| true termination | zero | no | zero |
| time-limit truncation | final-state critic value | no | zero |

Only completed environment lanes are reset. The acceptance check compares all
unaffected actor observations, critic states, actor memories, and critic
memories before and after each partial reset.

## Device and reference checks

The device buffer preallocates its time-major tensors on CUDA. After collection,
the run:

- recomputes GAE and returns with the simulator-independent CPU reference;
- creates actor chunks per environment and drone and critic chunks per
  environment;
- compares chunk metadata with the CPU reference;
- reevaluates stored actions through the actor distribution and compares old
  log probabilities;
- reevaluates critic chunks and compares stored values;
- checks finite tensors, bounded actions, reset memory, and termination masks;
- audits the saved CSV row identities, hashes, team fields, and terminal rows
  on the host.

Chunks never cross an episode boundary. Padding uses a valid-sample mask, and
each chunk carries the recurrent state from its first real timestep.

## Run the acceptance check

Prepare and build a fresh immutable runtime context, then run:

~~~sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only

# Replace <build-run> with the printed runs/runtime-build directory.
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run <build-run> &&
uv run --locked python scripts/run_recurrent_collector.py \
  --build-report <build-run>/report.json \
  --accept-eula \
  --gpu 0
~~~

The launcher checks that the selected GPU is available, disables container
networking, and saves the resolved configuration and runtime identity. A
prepared build context is single-use after a build attempt; create a new one if
the build report is already `built` or `failed`.

## Artifacts

Each run under `runs/recurrent-collector/<id>/` contains:

- `collector.csv`: one row per environment, drone, and step;
- `rollout.npz`: raw observations, states, actions, probabilities,
  rewards, values, masks, returns, and recurrent states;
- `initial-policy.pt`: the exact untrained actor/critic parameters and
  policy configuration used for collection;
- `metrics.json`: device checks, event records, errors, memory use, and
  artifact hashes;
- `host-audit.json`: independent CSV and hash checks;
- `config.json`, `build-report.json`, and
  `context-manifest.json`: resolved inputs and runtime provenance;
- `console.log` and `events.jsonl`: native and structured
  execution records.

The checkpoint is evidence for exact collection reevaluation. It is not a
training checkpoint because it contains no optimizer, counters, normalization,
or RNG restoration state.

## Accepted lab evidence

Run `20260916T190924.021233IST-667b7fa8` passed all 22 device checks
and all 12 host audit checks. The container exited zero. The full host-observed
run took 76.645 seconds, including Isaac Sim startup; collection and validation
inside the active task took 2.535 seconds.

Measured results include:

- actions remained in [-0.972658, 0.977489] and required no environment
  clipping;
- actor log-probability reevaluation differed by at most 1.907e-6;
- critic value reevaluation differed by at most 5.960e-7;
- CUDA and CPU GAE differed by at most 3.427e-7;
- CUDA and CPU returns differed by at most 3.576e-7;
- one safety termination used a zero bootstrap;
- four time-limit truncations retained finite nonzero critic bootstraps;
- unaffected partial-reset tensors and recurrent memories differed by exactly
  zero;
- the preallocated rollout tensors occupied 5,951,168 bytes, and measured peak
  CUDA allocation increased by 8,760,832 bytes;
- the raw rollout was 4,915,697 bytes and the policy checkpoint was 4,376,814
  bytes.

The first attempt `20260916T185858.564225IST-c88e14c2` completed the
full physics rollout but exposed an axis-order bug in the post-run critic-memory
check. The corrected implementation moves the environment axis ahead of the
layer axis before applying the time/environment boundary mask, with a host
regression test for that ordering.

## Limits and next implementation

This result establishes live collection, boundary handling, sequence
construction, and CPU/device parity for an untrained stochastic policy. It does
not establish a correct PPO objective, optimizer update, learned flight, or
recoverable training.

The next bounded component is a recurrent MAPPO update probe. It should compute
masked clipped actor and value losses over these chunks, apply finite optimizer
updates, report KL/clip/entropy/gradient diagnostics, and verify that padding
cannot change an update. Recoverable checkpoint state follows before sustained
training.
