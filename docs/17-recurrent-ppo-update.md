# Recurrent MAPPO optimizer update

## Status and purpose

ALiGn now has a masked recurrent PPO update for the shared actor and
centralized critic. It evaluates complete temporal chunks, reduces losses over
valid timesteps only, and applies separate Adam updates with gradient clipping.

The accepted check uses a deterministic synthetic sequence fixture with the
same actor, critic, action distribution, recurrent states, and chunk types as
the live collector. It performs real optimizer updates on CUDA without
launching Isaac Sim. It is an optimizer acceptance test, not a sustained
training run or evidence that the policy improves flight behavior.

## Configured update

The checked [PPO configuration](../configs/recurrent-ppo.json) uses:

| Quantity | Value |
|---|---:|
| actor learning rate | 0.0003 |
| critic learning rate | 0.001 |
| Adam epsilon | 0.00001 |
| policy clip ratio | 0.2 |
| value clip range | 0.2 |
| sampled-action entropy coefficient | 0.01 |
| value-loss coefficient | 0.5 |
| maximum gradient norm | 0.5 |
| update epochs in the probe | 2 |
| advantage normalization | enabled |

These values are executable defaults for the acceptance probe. They have not
been tuned for formation learning.

## Actor objective

For each valid actor timestep, the updater reevaluates the stored bounded
action under the current transformed Gaussian policy. It forms the importance
ratio:

~~~text
ratio = exp(new_log_probability - old_log_probability)
~~~

The team advantage is repeated for each drone that acted in that environment.
Valid advantages are normalized over the actor batch. The policy loss uses the
usual clipped surrogate:

~~~text
policy_loss = -mean(min(ratio * advantage,
                        clip(ratio, 1-epsilon, 1+epsilon) * advantage))
~~~

The actor also receives a Monte Carlo entropy bonus from a reparameterized
sample of the bounded distribution. The reported entropy is therefore a
sampled estimate, rather than the analytic entropy of an ordinary Gaussian.

## Critic objective

The critic processes one global-state sequence per environment. Its clipped
value estimate is:

~~~text
clipped_value = old_value + clip(new_value - old_value,
                                 -value_clip, value_clip)
~~~

The value loss is half the mean of the larger squared error from the unclipped
and clipped estimates. This prevents the clipping term from making an
otherwise worse critic update look artificially good.

Actor and critic use separate optimizers and parameter sets. Both recurrent
networks receive the exact initial hidden and cell states saved at each chunk
boundary.

## Padding contract

The network sees the whole padded sequence so recurrent execution retains its
normal shape. Loss inputs are selected only where `valid_mask` is one.

The acceptance probe deliberately replaces every padded actor observation,
action, old probability, advantage, and return, plus every padded critic state,
old value, and return, with large unrelated values. It repeats both update
epochs from identical parameters, optimizer states, and random seeds. Clean and
corrupted-padding updates must produce identical diagnostics and parameters.

This catches a common recurrent PPO error where zero padding silently enters
advantage normalization or a reduced loss.

## Run the CUDA acceptance check

Prepare and build a fresh immutable runtime context, then run:

~~~sh
sudo -v
uv run --locked python scripts/build_omnidrones_runtime.py --prepare-only

# Replace <build-run> with the printed runs/runtime-build directory.
uv run --locked python scripts/build_omnidrones_runtime.py \
  --prepared-run <build-run> &&
uv run --locked python scripts/run_recurrent_ppo.py \
  --build-report <build-run>/report.json \
  --accept-eula \
  --gpu 0
~~~

The launcher verifies the selected GPU, disables container networking, and
saves the policy and PPO configurations, build identity, metrics, console log,
and updated tensor-probe checkpoint under `runs/recurrent-ppo/<id>/`.

## Accepted evidence

Run `20260916T210604.064887IST-87a056c2` passed all 14 checks on one RTX
A4000 with Python 3.10.14 and vendor PyTorch 2.2.2+cu118. The host-observed run
took 8.903 seconds; the probe took 2.492 seconds. It used 28 valid actor samples
and seven valid critic samples across two recurrent update epochs.

Measured results include:

- actor parameters changed by at most `0.000600244` and critic parameters by
  `0.001998268`;
- both pre-clipping gradient norms were finite and positive in both epochs;
- clean and corrupted-padding actor and critic parameters differed by exactly
  zero after both updates;
- epoch-two approximate KL was `0.0049514`, policy clip fraction was
  `0.0357143`, and value clip fraction was `1.0`;
- the saved updated policy artifact was 4,377,838 bytes and its SHA-256 was
  verified by the host launcher.

The second epoch's explained variance was `-17.3738` and its value loss was
`4.36899`. On this small synthetic fixture, that shows the critic update moved
far from its targets. The probe requires finite, observable optimizer behavior;
it does not interpret a lower loss or positive explained variance as an
acceptance condition.

## Artifact limitation and next work

`updated-policy.pt` contains actor and critic parameters plus the probe
configuration and diagnostics. It intentionally does not claim resumable
training: optimizer states, observation normalization, RNG states, counters,
rollout position, and lineage are absent.

Before a long learning run, ALiGn needs an atomic recovery checkpoint that
preserves those states and falls back safely after an interrupted or corrupt
write. After recovery passes, the collector and updater can be joined into a
short task-connected simulator learning run with raw per-update and flight
metrics.
