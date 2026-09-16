# Recurrent rollout contract

## Status and purpose

ALiGn now has a simulator-independent reference for storing recurrent multi-agent rollouts, computing generalized advantage estimates (GAE), and producing episode-safe sequence chunks. It corrects the old implementation's length-one recurrent training path by preserving time order and the actual LSTM hidden and cell states at each chunk boundary.

This component does not yet collect tensors from OmniDrones, define neural networks, sample policy actions, or perform a PPO update. It defines and checks the data contract those components must obey.

## Checked production configuration

The default [rollout configuration](../configs/recurrent-rollout.json) matches the accepted four-environment task:

| Quantity | Value |
|---|---:|
| rollout horizon | 128 environment steps |
| environments | 4 |
| drones per environment | 4 |
| actor observation | 55 values per drone |
| centralized critic state | 80 values per environment |
| action | 4 values per drone |
| LSTM state | 1 layer × 128 values for both hidden and cell state |
| training chunk | at most 16 consecutive steps |
| discount / GAE lambda | 0.99 / 0.95 |

The actor observation and critic state are stored in separate fields. A future shared actor can therefore consume only its declared local 55-value input. Each critic sequence receives the centralized state separately.

## One stored transition

A rollout begins with a `RecurrentFrame`, which contains the observations and actor/critic LSTM states immediately before an action. Each append stores:

- the bounded action actually executed;
- its old policy log probability;
- per-agent reward and value estimate;
- the value of the final, pre-reset next observation for bootstrapping;
- true-termination and time-limit-truncation flags;
- the next frame used for the following action, after any required environment reset.

This separation matters at a time limit. The critic may bootstrap from the final observation, while the actor and critic memories must still reset before the next episode.

## Three different masks

| Boundary | Bootstrap next value? | Continue GAE trace? | Carry LSTM memory? |
|---|---:|---:|---:|
| ordinary transition | yes | yes | yes |
| true termination | no | no | no |
| time-limit truncation | yes | no | no |

The implementation requires zero bootstrap values for true terminations and zero actor/critic hidden and cell states after either kind of episode boundary. `terminated` and `truncated` cannot both be true.

For transition `t`, ALiGn computes:

~~~text
delta_t = reward_t + gamma * bootstrap_mask_t * final_value_t - value_t
adv_t   = delta_t + gamma * lambda * trace_mask_t * adv_(t+1)
return_t = adv_t + value_t
~~~

The bootstrap and trace masks are intentionally different for truncation.

## Temporal sequence chunks

The buffer splits each environment and drone trajectory at every episode boundary, then divides each episode segment into chunks of at most `chunk_length`. A chunk contains:

- ordered actor observations, critic states, actions, old log probabilities, values, advantages, and returns;
- terminal flags for each timestep;
- the exact actor and critic hidden/cell state at the chunk's first timestep;
- a valid-sample mask so padded rows never affect a loss.

Chunks may be shuffled reproducibly between minibatches, but timesteps inside a chunk are never shuffled. No chunk crosses from one episode into another.

## Run the CPU report

From `align/`:

~~~sh
uv run --locked python scripts/check_recurrent_rollout.py
~~~

The command creates `runs/recurrent-rollout/<id>/` with:

- `config.json`: copied resolved rollout configuration;
- `report.json`: dimensions, checks, advantages, chunk boundaries, and masks;
- `rollout.log` and `events.jsonl`: IST display timestamps and offset-aware UTC fields.

The accepted local report is `20260916T181145.446683IST-41dc17f2`. Its worked example passed all checks and produced advantages `[3.5, 3.0, 7.0]` for an ordinary transition, a true termination, and a time-limit truncation.

## Validation and limits

CPU tests cover finite values, exact shapes, bounded actions, mutually exclusive terminal flags, zero terminal bootstrap, recurrent reset enforcement, truncation bootstrap, stopped cross-episode GAE traces, padding masks, sequence order, episode-safe chunks, initial LSTM states, and reproducible chunk shuffling.

The reference uses immutable Python tuples so it can be checked without PyTorch or Isaac Sim. The future collector will need a device-resident PyTorch implementation or adapter with parity tests against this reference. A passing rollout report does not establish a correct LSTM network, action distribution, PPO loss, gradient update, learning result, or checkpoint recovery.

## Next implementation

The next bounded component is the shared LSTM actor and centralized recurrent critic, including a transformed bounded continuous-action distribution. Its tests must prove actor/critic input isolation, action/log-probability consistency, dependence on earlier observations, correct reset masks, and parity between full-sequence and chunked unrolling. A device-resident collector can then populate this rollout contract before PPO optimization is added.
