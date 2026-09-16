# Shared recurrent actor and centralized critic

## Status and purpose

ALiGn now defines the neural policy that will sit between the accepted vector task and recurrent rollout storage. It uses one actor for every drone, a separate centralized training critic, explicit LSTM state and masks, and a transformed continuous-action distribution whose samples stay inside the controller limits.

The implementation was checked with the pinned simulator image's vendor PyTorch on an RTX A4000. This check did not start Isaac Sim, collect an environment rollout, optimize PPO, or establish that the untrained policy can fly.

## Runtime boundary

The host package has no PyTorch dependency. `align.policies.config` is safe under the uv-managed Python 3.12 and simulator-compatible Python 3.10 host checks. `align.policies.torch_recurrent` is imported only inside the derived image, which supplies Python 3.10.14 and vendor PyTorch 2.2.2+cu118.

The checked [policy configuration](../configs/recurrent-policy.json) uses:

| Quantity | Value |
|---|---:|
| local actor input | 55 values per drone |
| centralized critic input | 80 values per environment |
| action | 4 values per drone |
| input projection | 256 units |
| LSTM | 1 layer, 256 hidden and cell values |
| initial log standard deviation | -0.5 |
| log-standard-deviation limits | [-5, 1] |
| action limits | [-1, 1] in every dimension |

The 55-value input follows the accepted local-observation contract, which adds explicit neighbor-validity masks to the draft's feature description. The actor therefore has 542,838 parameters rather than treating the draft's 51-value deployment table as the executable schema.

## Actor and critic separation

The shared actor accepts tensors shaped `[agent_sequence, time, 55]`. Each drone has its own hidden and cell state, but every sequence uses the same actor parameters. The actor API has no centralized-state argument.

The training critic accepts `[environment_sequence, time, 80]` and produces one cooperative state value per environment and timestep. It has its own LSTM and parameters. The device collector stores this value against the cooperative team reward, defined as the mean of per-agent totals, and broadcasts the resulting team advantage to the shared actor samples.

Both networks use:

~~~text
LayerNorm -> Linear(256) -> tanh -> LayerNorm -> masked LSTM -> LayerNorm -> head
~~~

The LSTM is explicitly unrolled over time. A memory mask of zero clears hidden and cell state before processing the current observation. A valid-sample mask preserves memory across trailing padding and zeros the padded outputs.

## Bounded Gaussian actions

The old code applied `tanh` to the Gaussian mean and then sampled an ordinary Gaussian. Such a sample can exceed the controller range even when its mean cannot.

ALiGn samples an unconstrained latent value and transforms the sample:

~~~text
z ~ Normal(mu, sigma)
a = bias + scale * tanh(z)
~~~

For the current `[-1, 1]` limits, `bias = 0`, `scale = 1`, and every finite sample is strictly between -1 and 1. PPO must evaluate the probability of the executed action after this transformation. The implementation includes the change-of-variables correction:

~~~text
log pi(a) = sum(log Normal(z; mu, sigma)
                - log(scale)
                - log(1 - tanh(z)^2))
~~~

The correction uses a numerically stable expression. Evaluation clamps only the inverse transform by `1e-6` at the boundary; collection stores the actual transformed sample and its corrected log probability.

## Reproduce the tensor acceptance

Build a context containing the current source, then run the policy probe:

~~~sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only

# Replace <build-run> with the printed runs/runtime-build directory.
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run <build-run>

uv run --locked python scripts/run_recurrent_policy.py \
  --build-report <build-run>/report.json \
  --accept-eula \
  --gpu 0
~~~

The launcher verifies that the selected GPU is idle, disables network access in the container, saves the exact image/build report and configuration, and writes the result under `runs/recurrent-policy/<id>/`. The command does not launch the simulator or train a policy.

## Accepted evidence

Run `20260916T183350.312358IST-f57f1110` passed all 18 checks in 6.435 seconds of host-observed time; the tensor probe itself took 0.540 seconds. It used Python 3.10.14, PyTorch 2.2.2+cu118, CUDA 11.8, and one RTX A4000.

Key measurements:

- full-sequence and two-chunk actor/critic outputs and final memories agreed exactly;
- changing pre-reset history produced zero post-reset difference;
- changing earlier observations without a reset changed the final actor output by 0.00190046;
- 6,144 sampled action components stayed strictly inside the bounds, with observed extrema -0.989691 and 0.989496;
- sampled and reevaluated log probabilities differed by at most 4.768e-6;
- all 15 actor and 14 critic trainable tensors received finite, nonzero gradients;
- CPU/CUDA maximum differences were 1.304e-8 for the actor and 3.576e-7 for the critic;
- the actor has 542,838 parameters (2,171,352 FP32 parameter bytes) and the critic has 548,513 parameters (2,194,052 bytes).

The earlier run `20260916T182952.983917IST-8099ba56` computed the same passing tensor result but the host could not read its root-owned `0600` metrics file. The probe now writes container-produced metrics as `0644`; the failed artifact is retained as evidence of that launcher defect.

## Limits and next implementation

Parameter bytes alone do not establish deployment feasibility. Working memory, exported artifact size, inference latency, quantization error, and real target-hardware behavior remain unmeasured.

The device collector feeds actual vector-task tensors through this policy, maintains separate actor memory per drone and critic memory per environment, stores pre-reset final values correctly, and matches the pure rollout reference; see [the collector contract](16-device-recurrent-collector.md). Masked PPO losses and optimizer updates are validated separately in [the optimizer contract](17-recurrent-ppo-update.md). The reset-mode recovery contract now preserves declared normalization state and the complete learner state; collecting and applying empirical observation statistics remains pending.
