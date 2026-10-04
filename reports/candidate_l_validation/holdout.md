# Candidate L frozen holdout and promotion evidence

2026-10-04. Explicit user approval to process Pulso TransMi private observations and promote if validation passes. The previous access blocker is resolved.

Spec unchanged from `187b1ef10bcafea83411d9e0b2d038024adb4806`: 40 recent real observations per station, minimum 32, two harmonics, period search 8–64 slots, ridge 0.05. No tuning on these holdout labels.

| Previously unused cutoff (virtual UTC) | Observable truth | K | L |
|---|---:|---:|---:|
| 2026-09-21 04:00 | 47/48 | 81.2702% | 91.1486% |
| 2026-09-21 05:00 | 46/48 | 88.5002% | 93.0143% |
| Pooled canonical station accuracy | 93/96 | 85.2719% | 91.9613% |

L gains 6.6893 percentage points, wins both cycles and improves 11/12 pooled station scores. Station 06000 regresses 2.840 pp (91.823% to 88.983%); this limited regression is accepted in exchange for the overall improvement, not hidden by per-station tuning. Coverage is 96.875%. Source-missing truth remains excluded, not zero-filled; this is not the teacher's complete official leaderboard.

History was filtered by both `observed_at <= data_cutoff` and `ingested_at <= original prediction created_at`. Metrics reuse `drift_monitor.accuracy_summary`. The already-delivered 06:00 cycle had no revealed labels in this snapshot and was not included. This is an untouched-label retrospective holdout, not a prospective submission result, and only two cycles; future 90% accuracy is not guaranteed.

## Release controls

- Frozen L refits real recent observations each cycle; records per-station dates, counts, hashes, periods and fitted coefficients.
- Checks exact target count/order, cutoff/horizon consistency, finite nonnegative output and fresh sufficient history.
- K remains registered and receives same-cycle comparison predictions where available.
- If L fitting or provenance recording fails, use the existing K path and submit under K's actual identity, not L's label.
- Record latest refit in model metadata; archive per-cycle manifests as Actions artifacts.
- Official submission contract, idempotency and Candidate G parameters are unchanged.
- 103 local tests passed, including 16 L model/runtime tests. Hosted full CI and read-only production integration are required before activation.

## Rollback

If L degrades, restore K model ID `8a0e5ad3-f8c8-4cc3-9c1e-e183a80326d4` as the sole active model in a transaction. Do not delete models or overwrite accepted submissions. The runtime fallback addresses execution failure, not automatic performance-based rollback.

Activation status is recorded in the model registry and subsequent incident updates; this report does not itself activate L.
