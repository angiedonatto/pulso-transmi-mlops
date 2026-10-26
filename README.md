# Pulso TransMi — MLOps SDK y Control Center

Starter kit oficial del reto MLOps **Pulso TransMi**. Incluye un cliente Python,
ejemplos reproducibles y una plantilla de GitHub Actions para construir un
pipeline que descargue datos, entrene, monitoree y posteriormente envíe
predicciones.

Este repositorio tambien contiene la capa operativa del proyecto: pipeline de
prediccion oficial, Candidate G, monitoreo canonico de drift, snapshots de
observabilidad y un dashboard Next.js listo para desplegar con root directory
`dashboard/`.

> **Disponible públicamente:** la API de lectura está en
> `https://pulso-transmi.72-60-245-2.sslip.io` y su documentación interactiva en
> [`/docs`](https://pulso-transmi.72-60-245-2.sslip.io/docs).

## El reto

Se pronostica demanda sintética cada 15 minutos para 12 estaciones reales de
TransMilenio. El sistema liberará observaciones con el tiempo y cambiará algunos
patrones durante la competencia. Un modelo entrenado una sola vez puede perder
desempeño: el objetivo es operar un pipeline capaz de medir, decidir y
reentrenar.

La demanda, clima y eventos son sintéticos. Los nombres y coordenadas de las
estaciones provienen de datos oficiales de TransMilenio.

## Inicio rápido

Requiere Python 3.11 o superior.

```bash
git clone https://github.com/uexternadojz/pulso-transmi-sdk.git
cd pulso-transmi-sdk
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml]'
cp .env.example .env
python examples/01_download.py
python examples/02_naive_baseline.py
```

En Windows PowerShell, la activación es `.venv\Scripts\Activate.ps1`.

Para trabajar con el pipeline completo y las pruebas actuales:

```bash
python -m pip install -e '.[dev,ml]'
PYTHONPATH=.:src pytest -q
```

## Estado actual del proyecto

La version actual ya no es solo un SDK base. Incluye:

- ingesta incremental de observaciones;
- sincronizacion de ciclos de forecast;
- generacion de predicciones oficiales;
- submission oficial idempotente;
- Candidate B y Candidate G;
- observabilidad de Candidate G via `prediction_traces`;
- monitoreo canonico con `src/drift_monitor.py`;
- snapshots persistidos en `monitoring_snapshots`;
- dashboard MLOps en `dashboard/`;
- workflow GitHub Actions que genera snapshot despues del ciclo oficial.

Consulta [AGENTS.md](AGENTS.md) antes de hacer cambios automatizados o de agente.

## Uso del SDK

```python
from pulso_transmi import PulsoTransmiClient

client = PulsoTransmiClient()

print(client.meta())
stations = client.stations()
observations = client.observations_dataframe(station_id="07107")
context = client.context_dataframe()

print(stations.head())
print(observations.tail())
```

El SDK recorre automáticamente todas las páginas. Si prefieres controlar cada
página, usa `client.observations_page(...)` y conserva `next_cursor` exactamente
como lo entrega la API.

## Datos iniciales

| Recurso | Tamaño |
|---|---:|
| Estaciones | 12 |
| Frecuencia | 15 minutos |
| Historia | 45 días |
| Periodos por estación | 4.320 |
| Observaciones | 51.840 |

Para evaluación local, usa una división temporal: por ejemplo, primeros 38 días
para entrenamiento y últimos 7 para validación. Una partición aleatoria mezcla
futuro y pasado y genera métricas engañosas.

## EDA reproducible

Este fork de trabajo incluye una primera capa de ciencia de datos para auditar
los datos, generar visualizaciones y comparar baselines antes de modelar. El
flujo mantiene los datos crudos fuera de Git en `data/raw/` y versiona solo
código y reportes livianos en `reports/eda/`.

```bash
python -m pip install -e '.[dev,ml]'
python -m src.ingest
python -m src.data_quality
python -m src.eda
jupyter notebook notebooks/01_eda_pulso_transmi.ipynb
```

Los comandos generan:

- `reports/eda/data_quality.md`: validaciones de calidad y cobertura;
- `reports/eda/summary.md`: resumen ejecutivo del EDA;
- `reports/eda/tables/`: tablas resumidas para análisis y entrega;
- `reports/eda/figures/`: visualizaciones reproducibles.

El notebook `notebooks/01_eda_pulso_transmi.ipynb` funciona como informe
narrativo: explica por qué se usa validación temporal, qué patrones aparecen por
estación y cómo interpretar los baselines iniciales.

## API de lectura `0.2.0`

| Método | Ruta | Uso |
|---|---|---|
| `GET` | `/health` | Estado básico |
| `GET` | `/v1/meta` | Versión, rango, hashes y enlaces |
| `GET` | `/v1/stations` | Catálogo geográfico |
| `GET` | `/v1/observations` | Demanda paginada |
| `GET` | `/v1/context` | Clima y eventos |
| `GET` | `/v1/downloads/{filename}` | Descarga completa |

Swagger está disponible en `/docs`. Consulta [docs/api.md](docs/api.md) para
filtros, paginación y errores.

## Estructura esperada del proyecto estudiantil

```text
mi-pulso-transmi/
├── src/
│   ├── ingest.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── monitor.py
├── tests/
├── artifacts/
├── requirements.txt o pyproject.toml
└── .github/workflows/pipeline.yml
```

El repositorio de cada equipo debe dejar trazabilidad de:

- cutoff de datos usado;
- versión o commit del código;
- features y modelo entrenado;
- métricas de validación temporal;
- momento y razón de cada reentrenamiento;
- errores de ingesta o inferencia.

## GitHub Actions

[`templates/pipeline.yml`](templates/pipeline.yml) es una plantilla manual para
equipos que empiezan desde cero.

Este repo de trabajo ya incluye workflows productivos:

- `.github/workflows/predict.yml`: sincroniza stream, genera predicciones,
  envia submission oficial y genera `monitoring_snapshots`;
- `.github/workflows/shadow.yml`: ejecuta Candidate B shadow;
- `.github/workflows/ci.yml`: corre pruebas.

Nunca escribas API keys, contraseñas de Supabase ni tokens dentro del código.

## Supabase, monitoreo y Vercel

Supabase persiste ejecuciones, modelos, predicciones, submissions, calidad de
datos, trazas de runtime y snapshots de monitoreo.

Tablas principales existentes:

- `stations`, `time_slots`, `observations`, `context_observations`;
- `pipeline_runs`, `model_versions`, `predictions`;
- `forecast_cycles`, `forecast_cycle_targets`, `submission_receipts`;
- `metrics_by_station`, `data_quality_checks`, `stream_cursors`;
- `runtime_ingestion_events`.

Tablas de observabilidad agregadas:

- `prediction_traces`: trazas de Candidate G (`base_prediction`,
  `calibration_factor`, `final_prediction`, `feedback_samples`, `data_cutoff`);
- `monitoring_snapshots`: estado canonico consumido por el dashboard.

La arquitectura completa esta documentada en
[docs/monitoring-architecture.md](docs/monitoring-architecture.md).

El dashboard vive en [dashboard/](dashboard/) y se despliega en Vercel usando
`dashboard/` como Root Directory.

```bash
cd dashboard
npm install
npm run typecheck
npm run build
```

Variables para Vercel:

- `NEXT_PUBLIC_SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `NEXT_PUBLIC_DASHBOARD_REFRESH_SECONDS`

No usar `NEXT_PUBLIC_SUPABASE_SERVICE_ROLE_KEY`.

Consulta [docs/student-project.md](docs/student-project.md) para el flujo completo
y los entregables.

## Métrica

La referencia actual es:

```text
WAPE = sum(abs(real - predicción)) / sum(real)
Accuracy = 100 × max(0, 1 - WAPE)
```

La métrica se calcula por estación y luego se promedia. El contrato definitivo
de submissions y leaderboard se publicará antes de iniciar la ventana competitiva.

## Monitoreo canonico

No duplicar calculos de accuracy, bias o drift en el dashboard. La fuente
canonica es `src/drift_monitor.py`, especialmente:

- `official_scored_predictions(...)`;
- `performance_drift(...)`;
- `data_drift(...)`;
- `residual_drift(...)`;
- `build_drift_report(...)`.

El snapshot operacional se genera con:

```bash
python scripts/run_monitoring_snapshot.py
```

Tambien se puede escribir una copia local:

```bash
python scripts/run_monitoring_snapshot.py --output reports/monitoring/latest.json
```

## Candidate G

Candidate G usa una capa adaptativa G2 con feedback de error resuelto. Sus
parametros productivos no deben cambiarse incidentalmente:

- `ALPHA = 1.0`;
- `FEEDBACK_WINDOW = 8`;
- `CLIP_LOW = 0.35`;
- `CLIP_HIGH = 2.5`;
- estimador `ewma`.

La observabilidad de Candidate G se guarda aparte en `prediction_traces` y no
cambia el payload oficial. La reconstruccion de feedback historico normaliza las
observaciones production-shaped antes de generar muestras supervisadas.

## Tracking histórico con MLflow

Los reportes existentes pueden reconstruirse en MLflow sin reentrenar ni
modificar producción: **16 experimentos y 108 runs históricos**. Consulta
[la guía de instalación, evidencia y reproducibilidad](docs/mlflow-tracking.md).

## Desarrollo del SDK

```bash
python -m pip install -e '.[dev,ml]'
PYTHONPATH=.:src pytest -q
```

Este repositorio es público para estudiantes. No debe contener ground truth
futuro, semillas, configuración privada del escenario ni parámetros de drift.
