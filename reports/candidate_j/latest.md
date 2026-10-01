# Candidate J — regime-aware shadow challenger (retrospective replay)

- Generated (execution clock): `2026-10-01T04:03:50.309449+00:00` · data up to (historical): `2026-09-18T10:00:00Z` · snapshot `/Users/AngieDonato/Documents/Proyectos/Academico/MLOPS/code/P1-Transmilenio/pulso-transmi-sdk/data/interim/drift_regime_analysis/snapshot_live`
- Scope: **shadow replay only**. Nothing was written to Supabase, Predict, `model_versions.is_active` or the submission API. Nothing is promoted.
- Accuracy: mean_station_accuracy = leaderboard formula (verified vs /v1/leaderboard); global_wape_accuracy = all rows pooled. Δ = J − G in percentage points on the **same targets**; phase accuracy pools each station over the whole phase, like the leaderboard window.

## Decision

| variant | designed | decision | reason | Δ stable | Δ phase 1 | Δ phase 2 | Δ shock | worst stable-station Δ |
|---|---|---|---|---:|---:|---:|---:|---:|
| `candidate_j_v1` | pre-registered | **reject** | harms the regime/stations where Candidate G already works | +0.00 | +0.03 | +1.65 | +17.94 | -2.03 |
| `candidate_j_v1_1` | post-hoc (after diagnosing v1 on these data) | **continue_shadow** | pending checks: C_shock_evidence_long_enough, D_recovery_observed | +0.04 | +0.31 | +1.54 | +17.94 | -0.43 |

Pre-registered checks (thresholds fixed in `candidate_j_regime.DecisionCriteria` before the first replay):

| check | `candidate_j_v1` | `candidate_j_v1_1` |
|---|---|---|
| A_preserves_normal | pass | pass |
| B_not_worse_phase2 | pass | pass |
| C_shock_gain | pass | pass |
| C_shock_evidence_long_enough | FAIL / pending | FAIL / pending |
| D_recovery_observed | FAIL / pending | FAIL / pending |
| E_stable_stations_not_harmed | FAIL / pending | pass |

Shock observed: 5 h, still active at the last observation; recovery observed: False.

## 1. Architecture and evidence

| rule | evidence (drift_regime_analysis unless marked post-hoc) | action |
|---|---|---|
| NORMAL → Candidate G unchanged | G reaches 83.3% (leaderboard formula) in regime_1 and 90.6-92.2% on the stations still stable in regime_2 | base_F × G2 exactly |
| SUSTAINED_DRIFT → keep G2 only where it helped the station in the last 24h | G2 adds +4.0 pp in regime_2 (corr(factor, needed) = 0.74), up to +38.5 pp at 03000 and +26 pp at 05100; G2 effect by station in regime_2 ranges from -2.8 pp (07111) to +38.5 pp (03000) | base_F × G2 if realised G2 effect ≥ 0 pp, else base_F |
| SUSTAINED_DRIFT ≥ 24h, repeatable new profile, no jump vs. yesterday → recent profile | lag_96 naive accuracy on day 1 of regime_2: 67.6% (0-42% at 02300/03000/05000/05100); day 2: 86.8% (ref 83.6-84.3%); day-to-day profile correlation in regime_2: 0.82-0.98 (02300, 03000, 05000, 07111, 09122); 0.72 at 05100 | y(t − 24h) × (last hour / same hour yesterday) |
| v1.1 only: recent profile also needs to have beaten G and base_F on the station's last 24h | the drift analysis recommends 'keep current behavior' where G already absorbs a sustained shift (6 stations at 86.8-92.2% in regime_2) and a recent-profile refit only where it does not; post-hoc: in the J v1 replay the recent-profile path cost -4.9 pp at 07107 and -5.2 pp at 06000 in phase 1 (stations where G worked) while gaining +4.3 pp overall on its rows in phase 2 | otherwise keep G |
| SEVERE_SHOCK (≥ 75% of stations jump vs. yesterday, kept 3 h) → latest observation | regime_3 hits 12/12 stations; the analysis calls a change global when >= 75% of stations are affected; regime_3 naive accuracy: lag_15 76.6%, lag_30 59.9%, lag_60 32.1%, lag_96 29.4%, weekly 23.0% vs G 26.0% and base F 38.5%; G2 effect -9.6 pp in regime_3; corr(factor, needed) ~ 0; 30% of rows at the 2.5 clip vs 0.2% in regime_2; the drift analysis segments the global series with a 3 h minimum regime length; regime_3 has lasted >= 5 h | y(data_cutoff); no base_F, no G2, no lag_96/weekly |
| Compounding guard: factor at the G2 clip + base already beyond the station's stable range in the same direction → factor 1 | 0% cycle (cutoff 2026-09-18 01:00): base 1,531 already reacted, x2.5 factor -> 3,827 vs actual 450-1,317 at 07111; G2 effect -9.6 pp in regime_3; corr(factor, needed) ~ 0; 30% of rows at the 2.5 clip vs 0.2% in regime_2 | drops the extreme factor only |

Thresholds are each station's extremes of the causal statistics over a stable calibration window `2026-08-25T06:00:00Z` → `2026-09-04T22:00:00Z` (3084 station-hours), which ends before the first evaluated cycle (log units vs. the causal 4-week same-slot median). No parameter of Candidate G (alpha, clip, EWMA window) was changed or tuned; J only chooses between existing signals.

| station_id | level24_abs_max | dispersion24_max | jump_abs_max | hour_anomaly_abs_max |
|---|---:|---:|---:|---:|
| 02300 | 0.086 | 0.197 | 0.821 | 0.588 |
| 03000 | 0.075 | 0.136 | 0.479 | 0.326 |
| 05000 | 0.076 | 0.121 | 0.463 | 0.312 |
| 05100 | 0.056 | 0.123 | 0.403 | 0.272 |
| 06000 | 0.076 | 0.128 | 0.492 | 0.281 |
| 06111 | 0.104 | 0.174 | 0.620 | 0.478 |
| 07105 | 0.114 | 0.154 | 0.615 | 0.418 |
| 07107 | 0.099 | 0.131 | 0.583 | 0.414 |
| 07111 | 0.072 | 0.111 | 0.404 | 0.248 |
| 09000 | 0.059 | 0.129 | 0.401 | 0.295 |
| 09122 | 0.076 | 0.145 | 0.571 | 0.434 |
| 10009 | 0.116 | 0.175 | 0.536 | 0.416 |

Design change made before any accuracy was computed: a first version pooled all stations into one threshold (the max over stations). Share of jumping stations over the shock cutoffs ['2026-09-18 00:00', '2026-09-18 01:00', '2026-09-18 02:00', '2026-09-18 03:00', '2026-09-18 04:00']: pooled [0.167, 0.667, 0.417, 0.5, 0.75] vs per-station [0.25, 0.75, 0.667, 0.833, 0.833] (shock declared at ≥ 0.75). Before the shock the maximum share was 0.33 (pooled) and 0.33 (per-station): no false alarm in either design. Per-station thresholds are what the drift analysis itself uses.

## 2. Why v1 was rejected and what v1.1 changes

J vs G restricted to the rows each path produced (pooled accuracy on those rows; `abs_error_saved` > 0 = J better):

| phase | path | rows_v1 | g_global_wape_accuracy_v1 | j_global_wape_accuracy_v1 | abs_error_saved_v1 | rows_v1_1 | j_global_wape_accuracy_v1_1 | abs_error_saved_v1_1 |
|---|---|---|---:|---:|---:|---|---:|---:|
| phase1 | g | 3664 | 84.94 | 84.94 | 0.00 | 4268 | 84.42 | 0.00 |
| phase1 | g_base_only | 196 | 84.85 | 81.32 | -3385.99 | 216 | 81.35 | -3861.94 |
| phase1 | recent_profile | 1420 | 80.15 | 81.35 | 7956.87 | 796 | 82.23 | 13269.99 |
| phase2 | g | 1208 | 82.00 | 81.90 | -570.63 | 1360 | 82.10 | -570.63 |
| phase2 | g_base_only | 4 | 55.19 | 0.00 | -139.20 | 12 | 83.80 | 605.55 |
| phase2 | recent_profile | 708 | 81.12 | 85.40 | 17789.64 | 548 | 85.09 | 14668.35 |
| phase3_shock | g | 24 | 57.10 | 29.45 | -2861.48 | 24 | 29.45 | -2861.48 |
| phase3_shock | g_base_only | 4 | 75.95 | 34.05 | -4233.66 | 4 | 34.05 | -4233.66 |
| phase3_shock | latest_observation | 192 | 18.99 | 42.72 | 26120.87 | 192 | 42.72 | 26120.87 |
| phase3_shock | recent_profile | 20 | 62.32 | 55.77 | -557.02 | 20 | 55.77 | -557.02 |
| stable | g | 7648 | 87.15 | 87.15 | 0.00 | 7720 | 87.15 | 0.00 |
| stable | g_base_only | 340 | 86.65 | 87.64 | 1193.10 | 344 | 87.60 | 1189.92 |
| stable | recent_profile | 76 | 85.74 | 80.02 | -659.09 | 0 |  |  |

v1 on the stations the drift analysis marked “keep current behavior” (J − G, pp) — the recent-profile path replaced a G that already worked:

| station_id | stable | phase1 | phase2 | phase3_shock |
|---|---:|---:|---:|---:|
| 06000 | 0.00 | -1.56 | -1.21 | 18.85 |
| 06111 | 0.00 | 0.00 | 0.00 | 6.58 |
| 07105 | -0.21 | -0.26 | 0.00 | 24.59 |
| 07107 | 0.00 | -2.03 | 0.28 | 45.65 |
| 09000 | -0.04 | -0.61 | 0.00 | 6.71 |
| 10009 | 0.00 | 0.00 | 0.00 | 30.38 |

## 3. Results by phase — `candidate_j_v1_1` (same targets)

| phase | model | cycles | rows | mean_station_accuracy | global_wape_accuracy | global_wape | accuracy_at_20 | mae | signed_bias | relative_bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stable | G | 168 | 8064 | 87.035 | 87.126 | 0.129 | 75.818 | 46.937 | -1.103 | -0.003 |
| stable | J | 168 | 8064 | 87.072 | 87.167 | 0.128 | 75.856 | 46.789 | -1.181 | -0.003 |
| stable | base_F | 168 | 8064 | 87.327 | 87.492 | 0.125 | 77.493 | 45.603 | 8.920 | 0.024 |
| phase1 | G | 110 | 5280 | 83.778 | 83.305 | 0.167 | 68.030 | 61.281 | 8.499 | 0.023 |
| phase1 | J | 110 | 5280 | 84.093 | 83.790 | 0.162 | 67.992 | 59.499 | 5.121 | 0.014 |
| phase1 | base_F | 110 | 5280 | 79.386 | 78.366 | 0.216 | 60.758 | 79.408 | 24.191 | 0.066 |
| phase2 | G | 40 | 1920 | 82.188 | 81.614 | 0.184 | 72.083 | 91.517 | -16.169 | -0.032 |
| phase2 | J | 40 | 1920 | 83.732 | 83.152 | 0.168 | 75.729 | 83.859 | -18.686 | -0.038 |
| phase2 | base_F | 40 | 1920 | 74.842 | 77.333 | 0.227 | 62.187 | 112.825 | 67.455 | 0.136 |
| phase3_shock | G | 5 | 240 | 25.241 | 28.617 | 0.714 | 15.833 | 413.486 | -111.905 | -0.193 |
| phase3_shock | J | 5 | 240 | 43.182 | 41.902 | 0.581 | 11.250 | 336.533 | 50.462 | 0.087 |
| phase3_shock | base_F | 5 | 240 | 39.997 | 38.660 | 0.613 | 9.167 | 355.315 | 245.111 | 0.423 |

| phase | G | J | Δ (pp) | G pooled | J pooled | Δ pooled (pp) | cycles won | lost | tied |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stable | 87.04 | 87.07 | 0.04 | 87.13 | 87.17 | 0.04 | 50 | 24 | 94 |
| phase1 | 83.78 | 84.09 | 0.31 | 83.30 | 83.79 | 0.49 | 44 | 53 | 13 |
| phase2 | 82.19 | 83.73 | 1.54 | 81.61 | 83.15 | 1.54 | 28 | 12 | 0 |
| phase3_shock | 25.24 | 43.18 | 17.94 | 28.62 | 41.90 | 13.28 | 1 | 4 | 0 |

Cycle wins/losses use the leaderboard formula on a one-cycle window, which caps each station-cycle at 0%: a catastrophic cycle costs G at most the station's share, while over a phase (as on the leaderboard) the full error counts. That is why J can lose several shock cycles yet win the phase.

Worst station per phase:

| phase | g_worst_station | g_worst_accuracy | j_worst_station | j_worst_accuracy |
|---|---|---:|---|---:|
| stable | 09000 | 85.99 | 10009 | 86.08 |
| phase1 | 09122 | 75.88 | 09122 | 76.24 |
| phase2 | 05100 | 59.40 | 05100 | 62.30 |
| phase3_shock | 07107 | 0.00 | 03000 | 33.96 |

Paths used (share of rows per phase):

| phase | regime | path | rows | share |
|---|---|---|---:|---:|
| phase1 | NORMAL | g | 2108 | 0.399 |
| phase1 | SUSTAINED_DRIFT | g | 2160 | 0.409 |
| phase1 | SUSTAINED_DRIFT | g_base_only | 216 | 0.041 |
| phase1 | SUSTAINED_DRIFT | recent_profile | 796 | 0.151 |
| phase2 | NORMAL | g | 336 | 0.175 |
| phase2 | SUSTAINED_DRIFT | g | 1024 | 0.533 |
| phase2 | SUSTAINED_DRIFT | g_base_only | 12 | 0.006 |
| phase2 | SUSTAINED_DRIFT | recent_profile | 548 | 0.285 |
| phase3_shock | NORMAL | g | 4 | 0.017 |
| phase3_shock | SEVERE_SHOCK | latest_observation | 192 | 0.800 |
| phase3_shock | SUSTAINED_DRIFT | g | 20 | 0.083 |
| phase3_shock | SUSTAINED_DRIFT | g_base_only | 4 | 0.017 |
| phase3_shock | SUSTAINED_DRIFT | recent_profile | 20 | 0.083 |
| stable | NORMAL | g | 7116 | 0.882 |
| stable | SUSTAINED_DRIFT | g | 604 | 0.075 |
| stable | SUSTAINED_DRIFT | g_base_only | 344 | 0.043 |

## 4. Answers (`candidate_j_v1_1`)

**A. Does J preserve G in the normal regime?** Δ stable = +0.04 pp; per-station Δ in the stable week from -0.21 to +0.32 pp; false SEVERE_SHOCK alarms before the shock: 0.

**B. Does J improve the structural changes of phase 2?** Δ phase 2 = +1.54 pp (phase 1: +0.31 pp). Per station:

| station_id | g_accuracy | j_accuracy | delta_pp | g_relative_bias | j_relative_bias | share_rows_changed |
|---|---:|---:|---:|---:|---:|---:|
| 07107 | 86.70 | 86.53 | -0.16 | 0.03 | 0.02 | 0.30 |
| 07105 | 91.42 | 91.42 | 0.00 | 0.01 | 0.01 | 0.00 |
| 06000 | 90.38 | 90.38 | 0.00 | 0.01 | 0.01 | 0.00 |
| 06111 | 90.75 | 90.75 | 0.00 | -0.01 | -0.01 | 0.00 |
| 10009 | 92.08 | 92.08 | 0.00 | -0.00 | -0.00 | 0.00 |
| 09000 | 91.23 | 91.23 | 0.00 | -0.01 | -0.01 | 0.00 |
| 05000 | 74.35 | 75.72 | 1.37 | -0.09 | -0.10 | 0.38 |
| 02300 | 74.26 | 75.75 | 1.48 | -0.10 | -0.10 | 0.33 |
| 03000 | 74.15 | 76.85 | 2.70 | -0.12 | -0.10 | 0.35 |
| 05100 | 59.40 | 62.30 | 2.90 | -0.15 | -0.15 | 0.23 |
| 07111 | 80.64 | 83.56 | 2.92 | -0.00 | 0.01 | 0.95 |
| 09122 | 80.88 | 88.21 | 7.33 | 0.07 | -0.01 | 1.00 |

**C. Does J avoid or reduce the phase-3 collapse?** Δ shock = +17.94 pp over 5 shock cycles (4 flagged SEVERE_SHOCK; first detection `2026-09-18T06:00:00Z`, 1.0 h after the shock start `2026-09-18T05:00:00Z`).

| station_id | g_accuracy | j_accuracy | delta_pp | g_relative_bias | j_relative_bias |
|---|---:|---:|---:|---:|---:|
| 07111 | 31.12 | 35.32 | 4.20 | -0.19 | 0.12 |
| 02300 | 48.58 | 54.36 | 5.78 | -0.20 | 0.06 |
| 06111 | 33.56 | 40.14 | 6.58 | 0.12 | 0.19 |
| 09000 | 39.91 | 46.63 | 6.71 | -0.16 | 0.15 |
| 03000 | 25.74 | 33.96 | 8.21 | -0.29 | 0.14 |
| 05000 | 32.30 | 43.61 | 11.31 | -0.09 | 0.04 |
| 09122 | 30.32 | 47.10 | 16.78 | 0.00 | 0.10 |
| 06000 | 20.91 | 39.75 | 18.85 | -0.28 | 0.13 |
| 07105 | 25.88 | 50.47 | 24.59 | -0.10 | 0.09 |
| 10009 | 10.33 | 40.71 | 30.38 | -0.26 | -0.10 |
| 05100 | 4.24 | 40.48 | 36.23 | -0.73 | -0.18 |
| 07107 | 0.00 | 45.65 | 45.65 | -0.58 | -0.06 |

| data_cutoff | g_mean_station_accuracy | j_mean_station_accuracy | delta_pp | g_global_wape_accuracy | j_global_wape_accuracy | regimes | paths |
|---|---:|---:|---:|---:|---:|---|---|
| 09-18 00:00 | 45.02 | 35.48 | -9.55 | 65.21 | 38.79 | NORMAL;SUSTAINED_DRIFT | g;g_base_only;recent_profile |
| 09-18 01:00 | 38.06 | 36.88 | -1.19 | 0.00 | 20.49 | SEVERE_SHOCK | latest_observation |
| 09-18 02:00 | 41.66 | 40.82 | -0.84 | 45.57 | 49.11 | SEVERE_SHOCK | latest_observation |
| 09-18 03:00 | 29.70 | 40.13 | 10.43 | 20.26 | 47.52 | SEVERE_SHOCK | latest_observation |
| 09-18 04:00 | 38.48 | 37.89 | -0.58 | 38.34 | 49.68 | SEVERE_SHOCK | latest_observation |

**D. Does it recover quickly?** The shock is still active at the last observation (`True`): recovery *after* the shock cannot be measured yet. Measurable now: 1 h detection latency, the shock state is held through the wave's troughs, and the first, undetected shock cycle is handled by the sustained-drift paths (table above).

**E. Does J avoid harming stations that stayed stable?** Stations marked “keep current behavior” by the drift analysis: 06000, 06111, 07105, 07107, 09000, 10009. J − G (pp):

| station_id | stable | phase1 | phase2 | phase3_shock |
|---|---:|---:|---:|---:|
| 06000 | 0.00 | -0.43 | 0.00 | 18.85 |
| 06111 | 0.00 | 0.00 | 0.00 | 6.58 |
| 07105 | -0.21 | 0.12 | 0.00 | 24.59 |
| 07107 | 0.00 | -0.10 | -0.16 | 45.65 |
| 09000 | 0.32 | -0.18 | 0.00 | 6.71 |
| 10009 | 0.00 | 0.00 | 0.00 | 30.38 |

## 5. Accuracy by horizon

| phase | horizon_minutes | g_accuracy | j_accuracy | delta_pp | g_relative_bias | j_relative_bias |
|---|---:|---:|---:|---:|---:|---:|
| stable | 15 | 86.96 | 87.01 | 0.04 | -0.01 | -0.01 |
| stable | 30 | 87.36 | 87.37 | 0.00 | 0.00 | 0.00 |
| stable | 45 | 86.98 | 87.01 | 0.03 | -0.00 | -0.00 |
| stable | 60 | 86.84 | 86.90 | 0.06 | -0.00 | -0.00 |
| phase1 | 15 | 85.20 | 85.49 | 0.28 | 0.01 | 0.01 |
| phase1 | 30 | 84.07 | 84.40 | 0.32 | 0.02 | 0.02 |
| phase1 | 45 | 83.48 | 83.42 | -0.06 | 0.02 | 0.01 |
| phase1 | 60 | 82.37 | 83.07 | 0.70 | 0.04 | 0.02 |
| phase2 | 15 | 84.44 | 84.77 | 0.33 | -0.08 | -0.07 |
| phase2 | 30 | 82.56 | 84.44 | 1.88 | -0.05 | -0.05 |
| phase2 | 45 | 82.27 | 83.81 | 1.54 | -0.02 | -0.03 |
| phase2 | 60 | 79.38 | 81.82 | 2.45 | 0.02 | 0.00 |
| phase3_shock | 15 | 44.81 | 71.39 | 26.58 | -0.27 | 0.07 |
| phase3_shock | 30 | 29.66 | 53.24 | 23.57 | -0.19 | 0.12 |
| phase3_shock | 45 | 19.98 | 31.84 | 11.86 | -0.21 | 0.07 |
| phase3_shock | 60 | 16.16 | 16.51 | 0.35 | -0.11 | 0.09 |

## 6. Replay fidelity (official cycles)

Replay G vs the official G predictions on 78 official cycles (3744 rows): 94.9% identical, median |Δ| 0.000, max |Δ| 465.6. Differences come from production G2 feedback skipping the cycles the pipeline missed (the replay has every hourly cycle). Mean per-cycle accuracy on the official cycles: official G 80.49%, replay G 80.76%, J 81.30% (`tables/official_cycles_comparison.csv`).

## 7. Limitations

- The shock path was chosen from the drift-analysis lag table, measured on the same shock hours: phase-3 results are **in-sample for the design**. Only future shock hours are a clean test.
- v1.1 was designed after seeing v1's replay on these data; its numbers are post-hoc and its decision is capped at `continue_shadow`.
- The shock has 5 h of data and has not ended: no post-shock recovery is observable.
- The stable-week result is out-of-sample for the thresholds (calibrated before it).

## 8. Figures

- `figures/01_cycle_accuracy_g_vs_j.png`
- `figures/02_station_delta_by_phase.png`
- `figures/03_horizon_by_phase.png`

## 9. Reproduce

```bash
python scripts/candidate_j_replay.py --from-snapshot /Users/AngieDonato/Documents/Proyectos/Academico/MLOPS/code/P1-Transmilenio/pulso-transmi-sdk/data/interim/drift_regime_analysis/snapshot_live
python scripts/run_candidate_j_shadow.py            # dry run on the latest live cycle (read-only)
```
