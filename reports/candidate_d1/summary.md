# Candidate D1 - Residual Calibration Challenger

## Scope

- Experimental only; champion, Predict, scheduler, submissions, and Supabase production state were not modified.
- Candidate D1 learns `residual_target = actual - candidate_b_prediction` and predicts `candidate_b_prediction + predicted_residual`.
- Hard holdout is the latest temporal block and is not used to choose the residual corrector.

## Windows

- Observations: 2026-07-26T05:00:00Z to 2026-09-09T04:45:00Z.
- Walk-forward: 2026-09-01T04:00:00Z to 2026-09-08T03:00:00Z (168 cutoffs).
- Hard holdout: 2026-09-08T04:00:00Z to 2026-09-09T03:00:00Z (24 cutoffs).

## Comparison

| model | wf_mean_station_accuracy | wf_delta_pp_vs_b | wf_global_accuracy | wf_wape | wf_mae | wf_bias | wf_signed_relative_bias | hard_mean_station_accuracy | hard_delta_pp_vs_b | hard_global_accuracy | hard_wape | hard_mae | hard_bias | hard_signed_relative_bias | wf_stations_improve | wf_stations_worsen | hard_stations_improve | hard_stations_worsen | train_seconds | complexity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Candidate B v2 | 87.280 | 0.000 | 87.418 | 0.126 | 45.476 | -9.594 | -0.027 | 88.463 | 0.000 | 88.494 | 0.115 | 45.051 | -11.917 | -0.030 | 0 | 0 | 0 | 0 | 264.932 | Champion pipeline simulation: Candidate B HistGradientBoostingRegressor per temporal fold. |
| Candidate D1 ridge raw | 87.098 | -0.182 | 87.263 | 0.127 | 46.036 | -5.141 | -0.014 | 88.283 | -0.179 | 88.284 | 0.117 | 45.874 | -5.830 | -0.015 | 0 | 12 | 3 | 9 | 0.771 | Residual corrector ridge; correction raw. |
| Candidate D1 ridge clipped | 87.098 | -0.182 | 87.263 | 0.127 | 46.036 | -5.141 | -0.014 | 88.283 | -0.179 | 88.284 | 0.117 | 45.874 | -5.830 | -0.015 | 0 | 12 | 3 | 9 | 0.702 | Residual corrector ridge; correction clipped by training residual quantiles. |
| Candidate D1 hgb raw | 87.332 | 0.053 | 87.459 | 0.125 | 45.328 | -8.522 | -0.024 | 88.555 | 0.092 | 88.603 | 0.114 | 44.624 | -10.781 | -0.028 | 8 | 4 | 9 | 3 | 4.525 | Residual corrector hist_gradient_boosting; correction raw. |
| Candidate D1 hgb clipped | 87.332 | 0.053 | 87.459 | 0.125 | 45.328 | -8.522 | -0.024 | 88.555 | 0.092 | 88.603 | 0.114 | 44.624 | -10.781 | -0.028 | 8 | 4 | 9 | 3 | 4.732 | Residual corrector hist_gradient_boosting; correction clipped by training residual quantiles. |
| Candidate D1 extra trees clipped | 87.246 | -0.034 | 87.367 | 0.126 | 45.658 | -3.979 | -0.011 | 88.383 | -0.080 | 88.473 | 0.115 | 45.132 | -7.086 | -0.018 | 3 | 9 | 6 | 6 | 15.590 | Residual corrector extra_trees; correction clipped by training residual quantiles. |

## Best D1 By Walk-Forward

- Model: Candidate D1 hgb raw.
- WF delta vs Candidate B v2: 0.053 pp.
- Hard holdout delta vs Candidate B v2: 0.092 pp.

## Leakage And Safety

- `leakage_check = passed`.
- `residual_training_predictions = out_of_fold`.
- `hard_holdout_isolated = true`.
- All feature rows satisfy `target_at = data_cutoff + horizon`.
- Candidate B base predictions and residual correctors are trained per temporal fold using only training cutoffs before the validation fold.
- Clipped variants learn residual bounds from training residual quantiles by horizon, falling back to global training quantiles when needed.
- No weather or event features were used.
- D1 was not promoted.
