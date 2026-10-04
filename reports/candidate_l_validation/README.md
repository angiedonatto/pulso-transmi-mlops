# Candidate L: frozen development result

Candidate L fits two harmonics per station to the latest 40 real post-change observations. Frequency and regression weights are re-estimated at each cutoff. It excludes unavailable observations using both observed_at and ingested_at, excludes missing labels, rejects stale history and records training hashes, coefficients and dates. K remains active; L is not connected to official submissions.

## Paired development replay

The saved snapshot has 141 revealed targets from three repaired K cycles (141/144, 97.92% observable coverage). All predictions use only history available at their original prediction time. Evaluation reuses `drift_monitor.accuracy_summary`.

| Virtual cutoff | Revealed | K | L |
|---|---:|---:|---:|
| 2026-09-21 01:00 UTC | 46/48 | 75.9412% | 86.8583% |
| 2026-09-21 02:00 UTC | 48/48 | 78.9448% | 92.4760% |
| 2026-09-21 03:00 UTC | 47/48 | 81.2610% | 92.8879% |
| Pooled station accuracy | 141/144 | 81.4803% | 90.6901% |

Improvement: +9.2098 percentage points. Pooled station accuracy is not the arithmetic mean of the cycle scores.

## Limits and promotion decision

The 40-observation configuration was selected using these development cycles. Therefore 90.6901% is selection-biased development evidence, not an independent backtest, prospective result or official leaderboard score. No 90% future-performance claim is supported. The teacher's evaluation truth includes source-missing values that this snapshot cannot score.

The next saved cycle (04:00 UTC) has predictions but no labels in this local snapshot. Automatic approval review blocked both downloading fresh observations and a server-side aggregate comparison against those observations, citing authorization to process private data. No new labels were retrieved by those attempts. Independent validation and promotion remain blocked; no champion or official submission change was made.

Before promotion: compare frozen L to K on previously unused resolved cycles, report identical target coverage and station regressions, then verify exact-target runtime integration and rollback. Preserve the frozen spec rather than tuning on the holdout.

## Reproduce

```bash
PYTHONPATH=.:src python scripts/evaluate_candidate_l_snapshot.py \
  --observations PATH_TO_SAVED_OBSERVATIONS \
  --predictions PATH_TO_SAVED_K_ROWS \
  --created-since 2026-10-04T11:45Z \
  --output reports/candidate_l_validation/output
PYTHONPATH=.:src python -m pytest -q tests/test_candidate_l_spectral.py
```

Raw observation snapshots are not committed. The evaluator produces training manifests for each replayed cycle, including the unscored next cycle.
