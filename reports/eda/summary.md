# Resumen EDA Pulso TransMi

## Datos

- Dataset: `pulso-transmi-starter-v1`.
- Rango local: 2026-07-26 00:00:00-05:00 a 2026-09-08 23:45:00-05:00.
- Estaciones: 12.
- Observaciones: 51,840.
- Contexto: 4,320 registros.

## Hallazgos iniciales

- Estacion con mayor demanda total: Ricaurte - NQS (`07111`).
- Cutoff de validacion temporal: 2026-09-01 23:45:00-05:00; se reservan los ultimos 7 dias.
- Mejor baseline inicial: `promedio_estacion_dia_bloque` con accuracy promedio 87.27.
- La validacion es temporal; no se usa split aleatorio porque mezclaria pasado y futuro.

## Baselines

| baseline                     |   mean_accuracy |   median_accuracy |   mean_wape |
|:-----------------------------|----------------:|------------------:|------------:|
| promedio_estacion_dia_bloque |         87.2713 |           87.6188 |    0.127287 |
| rezago_24h                   |         77.8908 |           80.5758 |    0.221092 |

## Figuras generadas

- `figures/01_demanda_diaria.png`
- `figures/02_perfil_horario.png`
- `figures/03_perfil_semanal.png`
- `figures/04_ranking_estaciones.png`
- `figures/05_heatmap_estacion_hora.png`
- `figures/06_distribucion_estaciones.png`
- `figures/07_distribucion_lluvia.png`
- `figures/08_distribucion_temperatura.png`
- `figures/09_demanda_vs_lluvia.png`
- `figures/10_demanda_contexto.png`
- `figures/11_accuracy_baselines.png`
- `figures/12_accuracy_estacion_baseline.png`
