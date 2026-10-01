# Model Zoo Recent Replay Benchmark

- Generated at: `2026-10-01T12:56:41.661553+00:00`
- Real arrival-time window: `2026-09-30T05:00:00+00:00` -> `latest available`
- Historical data_cutoff window: `2026-09-18T06:00:00Z` -> `2026-09-18T19:00:00Z`
- Historical target_at max: `2026-09-18T20:00:00Z`
- Complete resolved cycles: `14`
- Common-cycle count: `14`

## Reproducibility Inventory

| model | version | implementation | artifact | replayable | reason_if_not_replayable | production_or_experimental | notes |
|---|---|---|---|---:|---|---|---|
| baseline_lag96 | daily persistence | scripts/model_zoo_recent_replay.py |  | True |  | baseline | Uses target_at - 24h, causal for all horizons. |
| baseline_latest | latest observation | scripts/model_zoo_recent_replay.py |  | True |  | baseline | Repeats latest station observation at data_cutoff. |
| candidate_b_v1 | candidate_b_v1 | src/candidate_b_shadow.py | artifacts/shadow/candidate_b_v1/model.pkl | True |  | shadow | Frozen HGB artifact. |
| candidate_b_v2 | candidate_b_v2 | src/candidate_b_shadow.py | artifacts/shadow/candidate_b_v2/model.pkl | True |  | artifact-only | Frozen HGB artifact with later training cutoff. |
| candidate_c | candidate_c experiments | scripts/candidate_c_experiments.py | reports/candidate_c/ | False | Experiment reports exist, but no frozen production artifact for this live window. | experimental/lab |  |
| candidate_d1 | residual calibration | scripts/candidate_d1_residual_calibration.py | reports/candidate_d1/ | False | Report artifacts only; dispatch/runtime wrapper is not available in this worktree. | experimental/lab |  |
| candidate_d2 | context audit |  |  | False | No replayable model implementation in this worktree. | experimental/lab |  |
| candidate_e | recency lab |  |  | False | No script/artifact present on origin/student-work worktree. | experimental/lab |  |
| candidate_f | profile-ratio HGB base | src/candidate_g_adaptive.py::build_candidate_f_base_frame; production G traces | implicit inside Candidate G | True |  | production component | Scored from G trace base_prediction. |
| candidate_g | candidate_g_v1 | src/candidate_g_adaptive.py | artifacts/shadow/candidate_g_v1/metadata.json | True |  | registered/production-or-shadow | Scored from persisted prediction_traces final_prediction. |
| candidate_h | guard lab |  |  | False | No script/artifact present on origin/student-work worktree. | experimental/lab |  |
| candidate_i_v1 | candidate_i_v1 | src/candidate_i_snr_gate.py | artifacts/shadow/candidate_i_v1/metadata.json | True |  | shadow | Causal OOS F history rebuilt per cycle; skipped by default in retrospective replay because it refits HGB history for every cycle. |
| candidate_j_v1_1 | candidate_j_v1_1 | src/candidate_j_shadow.py | artifacts/shadow/candidate_j_v1_1/metadata.json | True |  | shadow/official-capable | No tuning; wraps Candidate G trace causally. |

## Available Data View

| model | rows | cycles | mean_station_accuracy | raw_wape | global_wape_accuracy | mae | signed_bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_latest | 672 | 14 | 48.143 | 0.538 | 46.222 | 307.010 | 5.796 |
| candidate_j_v1_1 | 672 | 14 | 43.425 | 0.575 | 42.534 | 328.065 | -35.417 |
| candidate_f | 672 | 14 | 37.968 | 0.589 | 41.128 | 336.095 | -110.740 |
| candidate_b_v1 | 672 | 14 | 21.632 | 0.729 | 27.050 | 416.461 | -221.963 |
| candidate_g | 672 | 14 | 21.040 | 0.778 | 22.231 | 443.972 | 164.200 |
| candidate_b_v2 | 672 | 14 | 20.974 | 0.733 | 26.739 | 418.236 | -220.595 |
| baseline_lag96 | 672 | 14 | 19.977 | 0.755 | 24.451 | 431.301 | -131.310 |

## Common Cycles View

| model | rows | cycles | mean_station_accuracy | raw_wape | global_wape_accuracy | mae | signed_bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_latest | 672 | 14 | 48.143 | 0.538 | 46.222 | 307.010 | 5.796 |
| candidate_j_v1_1 | 672 | 14 | 43.425 | 0.575 | 42.534 | 328.065 | -35.417 |
| candidate_f | 672 | 14 | 37.968 | 0.589 | 41.128 | 336.095 | -110.740 |
| candidate_b_v1 | 672 | 14 | 21.632 | 0.729 | 27.050 | 416.461 | -221.963 |
| candidate_g | 672 | 14 | 21.040 | 0.778 | 22.231 | 443.972 | 164.200 |
| candidate_b_v2 | 672 | 14 | 20.974 | 0.733 | 26.739 | 418.236 | -220.595 |
| baseline_lag96 | 672 | 14 | 19.977 | 0.755 | 24.451 | 431.301 | -131.310 |

## Wins / Losses / Ties

| model | reference | cycles | wins | losses | ties | mean_delta_pp |
|---|---|---:|---:|---:|---:|---:|
| baseline_lag96 | candidate_g | 14 | 5 | 9 | 0 | -0.923 |
| baseline_latest | candidate_g | 14 | 11 | 3 | 0 | 4.701 |
| candidate_b_v1 | candidate_g | 14 | 6 | 8 | 0 | -0.051 |
| candidate_b_v2 | candidate_g | 14 | 6 | 8 | 0 | -0.240 |
| candidate_f | candidate_g | 14 | 6 | 8 | 0 | 3.443 |
| candidate_j_v1_1 | candidate_g | 14 | 8 | 6 | 0 | 3.038 |
| baseline_lag96 | candidate_j_v1_1 | 14 | 4 | 10 | 0 | -3.961 |
| baseline_latest | candidate_j_v1_1 | 14 | 3 | 1 | 10 | 1.663 |
| candidate_b_v1 | candidate_j_v1_1 | 14 | 5 | 9 | 0 | -3.089 |
| candidate_b_v2 | candidate_j_v1_1 | 14 | 5 | 9 | 0 | -3.278 |
| candidate_f | candidate_j_v1_1 | 14 | 8 | 6 | 0 | 0.405 |
| candidate_g | candidate_j_v1_1 | 14 | 6 | 8 | 0 | -3.038 |

## Severe Shock / Regime Notes

| model | regime | cycles | mean_cycle_accuracy | stability_std | worst_cycle_accuracy | best_cycle_accuracy |
|---|---|---:|---:|---:|---:|---:|
| baseline_lag96 | SEVERE_SHOCK | 10.000 | 35.418 | 4.485 | 25.038 | 41.075 |
| baseline_lag96 | SUSTAINED_DRIFT | 4.000 | 33.636 | 5.329 | 27.875 | 39.686 |
| baseline_latest | SEVERE_SHOCK | 10.000 | 40.099 | 1.656 | 36.875 | 43.121 |
| baseline_latest | SUSTAINED_DRIFT | 4.000 | 41.620 | 1.314 | 40.139 | 43.387 |
| candidate_b_v1 | SEVERE_SHOCK | 10.000 | 36.805 | 5.028 | 27.801 | 43.656 |
| candidate_b_v1 | SUSTAINED_DRIFT | 4.000 | 33.222 | 7.338 | 23.663 | 40.746 |
| candidate_b_v2 | SEVERE_SHOCK | 10.000 | 36.664 | 5.084 | 27.556 | 43.523 |
| candidate_b_v2 | SUSTAINED_DRIFT | 4.000 | 32.915 | 7.162 | 23.269 | 39.910 |
| candidate_f | SEVERE_SHOCK | 10.000 | 39.997 | 8.509 | 26.951 | 53.228 |
| candidate_f | SUSTAINED_DRIFT | 4.000 | 37.471 | 5.742 | 29.735 | 45.918 |
| candidate_g | SEVERE_SHOCK | 10.000 | 34.994 | 5.734 | 22.614 | 41.922 |
| candidate_g | SUSTAINED_DRIFT | 4.000 | 37.927 | 1.305 | 36.184 | 39.360 |
| candidate_j_v1_1 | SEVERE_SHOCK | 10.000 | 40.099 | 1.656 | 36.875 | 43.121 |
| candidate_j_v1_1 | SUSTAINED_DRIFT | 4.000 | 35.800 | 6.627 | 25.533 | 44.056 |

## Best / Worst Patterns

- Worst cycles and station/horizon slices are in `per_cycle.csv`, `per_station.csv`, and `per_horizon.csv`.
- `temporal_wide.csv` contains cycle-by-cycle model accuracy plus deltas vs Candidate G and Candidate J v1.1.
- Figures are in `figures/`.

## Replay Failures / Non-scored Cases

| model | cycle_id | reason |
|---|---|---|
| candidate_i_v1 |  | skipped by default: causal replay rebuilds OOS Candidate F HGB history per cycle; use --include-candidate-i for a slow full replay. |

## Candidate K Design Hints

- Reuse robust behavior by regime rather than a single global policy.
- Compare station/horizon slices where older baselines beat adaptive models; those are useful fallback signals.
- Treat severe-shock hours as test evidence only; do not tune Candidate K from these outputs without a new protocol.
