# Model improvement candidates

## Scope

- No production files, scheduler, submission logic, Supabase state, or active champion were modified.
- Validation is temporal walk-forward by daily folds over the last 7 days available locally.
- Every sample uses only sources at or before its `data_cutoff`.

## Data window

- Observations: 2026-07-26T05:00:00Z to 2026-09-09T04:45:00Z.
- Stations: 12.
- Validation cutoffs: 669 from 2026-09-02T04:45:00Z to 2026-09-09T03:45:00Z.

## Champion diagnosis

- Mean station accuracy: 77.880%.
- Mean station WAPE: 0.2212.
- Worst station: 02300 with accuracy 69.353% and WAPE 0.3065.
- Worst horizon: +15 min with accuracy 78.980% and WAPE 0.2102.
- Worst station x horizon: 02300 +15 min with accuracy 69.345%.
- Highest WAPE tables are saved by station, horizon, hour, weekday, and station x horizon.

## Candidate results

### candidate A

- Mean station accuracy: 82.692%.
- Mean station WAPE: 0.1731.
- Worst station: 02300 (76.384%).
- Worst horizon: +60 min (82.648%).
- Training/evaluation time: 42.48s.
- Complexity: Baja: blend por horizonte de lags recientes, lag diario y perfil historico.

### candidate B

- Mean station accuracy: 87.479%.
- Mean station WAPE: 0.1252.
- Worst station: 02300 (85.603%).
- Worst horizon: +60 min (87.468%).
- Training/evaluation time: 25.57s.
- Complexity: Media: HistGradientBoostingRegressor global, reentrenado por fold diario.

## Comparison

| model | accuracy_oficial_simulada | mejora_pp | estaciones_mejoran | estaciones_empeoran | complejidad_operativa | tiempo_entrenamiento_s |
| --- | --- | --- | --- | --- | --- | --- |
| champion actual | 77.880 | 0.000 | 0 | 0 | Muy baja: lag diario + fallback estacion x quarter-hour. | 0.074 |
| candidate A | 82.692 | 4.813 | 12 | 0 | Baja: blend por horizonte de lags recientes, lag diario y perfil historico. | 42.477 |
| candidate B | 87.479 | 9.599 | 12 | 0 | Media: HistGradientBoostingRegressor global, reentrenado por fold diario. | 25.568 |

## Recommendation

Candidate recomendado para siguiente revision: candidate B. No promover automaticamente; revisar estaciones degradadas y probar en shadow run antes de produccion.

## Artifacts

- `reports/model_improvement/summary.md`
- `reports/model_improvement/summary.json`
- `reports/model_improvement/tables/*.csv`
