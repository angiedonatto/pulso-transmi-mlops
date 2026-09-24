# Candidate C v1 experiment

## Scope

- Experimental only; production champion, submissions, scheduler, and workflows were not changed.
- Candidate C starts from Candidate B and adds only features available at or before each `data_cutoff`.
- Hard holdout is the latest temporal block and is not used for model selection.

## Windows

- Observations: 2026-07-26T05:00:00Z to 2026-09-11T17:00:00Z.
- Walk-forward: 2026-09-03T16:00:00Z to 2026-09-10T15:00:00Z.
- Hard holdout: 2026-09-10T16:00:00Z to 2026-09-11T16:00:00Z.

## Walk-forward comparison

| model | walk_forward_accuracy | improvement_pp_vs_candidate_b | stations_improve | stations_worsen | worst_station | worst_station_accuracy | worst_horizon | worst_horizon_accuracy | estimator | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Candidate B actual | 86.916 | 0.000 | 0 | 0 | 10009 | 85.315 | 30 | 86.960 | hist_gradient_boosting | 41.739 |
| candidate C global | 86.843 | -0.072 | 5 | 7 | 09000 | 85.426 | 60 | 86.800 | hist_gradient_boosting | 16.308 |
| candidate C por horizonte | 86.575 | -0.341 | 1 | 11 | 07105 | 84.759 | 60 | 86.626 | hist_gradient_boosting_by_horizon | 31.607 |
| candidate C hibrido por estacion | 87.110 | 0.194 | 8 | 4 | 09000 | 85.426 | 60 | 86.943 | extra_trees | 28.778 |

## Hard holdout comparison

| model | hard_holdout_accuracy | worst_station | worst_station_accuracy | worst_horizon | worst_horizon_accuracy |
| --- | --- | --- | --- | --- | --- |
| Candidate B actual | 85.886 | 06111 | 82.974 | 30 | 85.165 |
| candidate C global | 86.071 | 06111 | 83.137 | 30 | 85.288 |
| candidate C por horizonte | 86.059 | 06111 | 83.209 | 30 | 85.187 |
| candidate C hibrido por estacion | 86.253 | 06111 | 84.124 | 30 | 85.793 |

## Candidate B diagnosis in this window

- Mean station accuracy: 86.916%.
- Worst station: 10009 (85.315%).
- Worst horizon: +30 (86.960%).

## Best experimental result

- Walk-forward best: candidate C hibrido por estacion at 87.110%.
- Hard-holdout best: candidate C hibrido por estacion at 86.253%.

## Leakage controls

- Target/cutoff consistency: True.
- Lags and rolling windows are shifted by the forecast horizon, so they end at `data_cutoff`.
- Profile features use prior same-slot observations only (`shift(1)`), never the target row.
- Station mapping is deterministic from station ids in training/static observations.

## Recommendation

Candidate C requiere mas experimentos: mejora agregada, pero degrada demasiadas estaciones.
