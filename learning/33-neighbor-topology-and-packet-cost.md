# Neighbor topology and packet cost

## Why count links separately from bytes?

Imagine drone 0 sees drones 1 and 2. Its actor has two valid neighbor slots. That tells us which information it used, but not how the information arrived. Two unicast packets, one broadcast received by both, an onboard camera, and simulator ground truth can all create the same observation. We must state a communication mechanism before estimating bytes.

ALiGn first applies a physical radius and then keeps up to `max_neighbors` nearest drones, breaking distance ties by identity. The selected relationship is directed: “sender 1 supplies receiver 0.” A pair can contribute zero, one, or two directed relationships. The actor input remains the same size because unused slots are masked and zero-padded.

## Worked example

Suppose drones 1 and 2 both select drone 0, and drone 0 selects drone 1. There are three directed edges but only two distinct senders. With the baseline 40-byte state packet, the unicast estimate is `3 × 40 = 120` bytes per snapshot; an ideal one-packet-per-sender broadcast estimate is `2 × 40 = 80` bytes. Neither value includes radio retries or contention. The example is checked in `tests/test_communication_accounting.py`.

`src/align/tasks/communication.py` defines the validated packet contract and counts edges/senders. `src/align/tasks/communication_report.py` reconstructs topology from the saved trajectory under each budget and writes raw rows plus totals. The source flight stays fixed. Consequently, the budget sweep can tell us how many observation edges and proxy bytes *would* result from different filters on those positions; it cannot tell us how a policy would fly with those filters.

## Validation and limits

The accepted CPU audit is documented in [communication accounting](../docs/34-communication-accounting.md). It used 2,210 four-drone snapshots and passed the source trajectory audit. Tests cover radius exclusion, budget saturation, directed links, and broadcast sender deduplication. No radio system was instrumented, and message age is not yet represented. A later live experiment must say whether positions come from localization, sensing, or received packets, then vary the policy's actual input budget under matched evaluation conditions.
