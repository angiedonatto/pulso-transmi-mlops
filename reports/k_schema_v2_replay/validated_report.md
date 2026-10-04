# Frozen K validation after schema-v2 repair

Evaluated at 2026-10-04T12:25:12.149996+00:00. Workflow: https://github.com/angiedonatto/pulso-transmi-mlops/actions/runs/37201940229

This is a causal retrospective replay of unchanged K v2.1 using repaired collector inputs, not the official leaderboard or replacement submissions. Inputs respect virtual cutoff and schema-v2 real publication time. Missing truth is excluded; metrics use src/drift_monitor.py mean station accuracy.

| Window | Revealed/expected | K replay accuracy | Historical official as sent | Latest-value baseline |
| --- | --- | --- | --- | --- |
| Latest 6 cycles | 282/288 | 80.9529% | 17.2097% | 75.1965% |
| Latest 12 cycles | 564/576 | 76.1621% | 16.3820% | 75.3894% |

Restricting to fully revealed cycles: K 82.8574% across 3 complete recent cycles, 76.8926% across 5 complete cycles in the 12-cycle window. Live repaired official cycle has 48 native K traces, but 0/48 revealed targets at this snapshot; live accuracy remains unavailable.

Schema changed from top-level demand to schema_version=2 with measurement.value numeric text, unit=passengers and quality observed/missing. Snapshot contains 624 unique rows, 612 observed and 12 missing across the same 12 stations. Collector parser, watermark advancement and monitoring pagination were repaired. No K parameters changed.

Demand also changed: recent same-slot means vs 7 days earlier show station 06111 at 5.32x, 07107 at 4.42x and 07105 at 0.50x. This comparison describes a regime shift, not an isolated causal diagnosis.

Recommendation: keep repaired native K active while evaluating adaptations in shadow. K wins the recent 6-cycle baseline by 5.76 percentage points, but only 0.77 across 12; an automatic model replacement is not supported. Prioritize adaptation immediately after shifts and station-specific residual calibration, especially 09000 (74.04% K vs 79.34% latest baseline) and 09122 (74.48% vs 74.87%). Require causal rolling validation before promotion. Seasonal-only expert scored 46.03% recent and 41.44% over 12, so restoring old seasonality alone is insufficient.

Validation: 87 local tests passed; hosted replay and causality/missing-value tests completed successfully. Audit has no competition API key and rejects database writes.
