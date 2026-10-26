# MLflow: experimentos históricos reproducibles

Este tracking es independiente del pipeline productivo. No entrena, no envía
submissions, no consulta ni modifica Supabase, no activa modelos y no modifica
el dashboard. Todos los runs llevan `historical=true` y
`execution=historical_import_no_training`. La hora de creación del run es la
**fecha de importación**, no una fecha de entrenamiento histórica.

## Ejecutar en macOS o Linux

Desde un checkout de la rama `feat/mlflow-historical-tracking`, con Python 3.11+
(validado con 3.12):

```bash
python3 -m venv .venv-mlflow
source .venv-mlflow/bin/activate
python -m pip install -e '.[tracking,dev]'
python -m pip check
python scripts/import_mlflow_history.py
python scripts/import_mlflow_history.py
python scripts/start_mlflow.py
```

En el Mac, abrir `http://127.0.0.1:5001`. El primer import crea 108 runs;
el segundo informa `created: 0`, `reused: 108`, `verified: 108`.
Para otro puerto: `python scripts/start_mlflow.py --port 5002`.
Para otro directorio persistente, usar el mismo `--store /ruta/absoluta` en
ambos scripts. Detener con Ctrl-C. Reiniciar el servidor conserva los datos.
No requiere API keys, Docker ni servicios externos.

La dependencia opcional `tracking` fija MLflow 3.5.1 y SQLAlchemy 2.0.44:
SQLAlchemy 2.1 elimina una clase importada por esta versión de MLflow.
Instalar en un virtualenv separado evita cambiar el runtime de inferencia.

## Auditoría e inventario

No existía integración MLflow en scripts, dependencias o documentación.
Los reportes versionados permiten recuperar métricas sin entrenamiento:

| Experimento (prefijo `Pulso /`) | Runs | Métricas conservadas |
|---|---:|---|
| `AB / temporal-walk-forward` | 3 | mean_station_accuracy, mean_station_wape, train_seconds |
| `C / walk-forward-selection` | 6 | mean_station_accuracy |
| `D1 / walk-forward` | 6 | mean_station_accuracy, global_accuracy, wape, mae, bias, signed_relative_bias, deltas y conteos |
| `D1 / hard-holdout` | 6 | mismas métricas, otro periodo y protocolo |
| `K / retrospective-phase-{stable,phase1,phase2,phase3_shock,oos_shock}` | 9 por fase (45) | mean_station_accuracy, global_wape_accuracy, global_wape, accuracy_at_20, mae, signed_bias, relative_bias, cycles, rows |
| `K / retrospective-window-{all_cycles,last_7d,last_24h,normal_phases}` | 9 por ventana (36) | mismas métricas, agregaciones separadas |
| `L / retrospective-holdout-{2026-09-21T04:00,2026-09-21T05:00,pooled}` | 2 por grupo (6) | mean_station_accuracy, observed_truth, target_count |

Total: **16 experimentos, 108 runs**. A/B incluye el baseline llamado
`champion actual` en el reporte original: ese nombre histórico no afirma que
siga activo. C conserva cada trial y sus hiperparámetros, incluidos los
estimadores híbridos. D1 separa los scores de walk-forward y hard holdout.
K incluye G y J como comparadores junto a siete variantes K.

Fuentes prioritarias:

- `reports/model_improvement/summary.json`
- `reports/candidate_c/tables/model_trials.csv` y `reports/candidate_c/summary.json`
- `reports/candidate_d1/tables/comparison.csv` y `reports/candidate_d1/summary.json`
- `reports/candidate_k/summary.json`
- `reports/candidate_l_validation/holdout.md`

No se inventan WAPE, MAE, bias, hiperparámetros, versiones o commits ausentes.
No se deriva WAPE del promedio de accuracy. Los porcentajes se guardan en
escala 0–100; los WAPE reportados mantienen su escala fraccional original.
Los nombres distinguen WAPE global de WAPE medio por estación. Los parámetros
disponibles aparecen en Params; configuraciones anidadas se conservan como
JSON. No se adjuntan pesos ni datos privados: cada run incluye copias pequeñas
de sus reportes y `provenance.json`. Los reportes apuntan a los artefactos
históricos cuando los describen; no se afirma que los pesos sean recuperables
para todas las variantes.

### Límites de comparación

Comparar modelos dentro del mismo experimento. Los ensayos C son selección
sobre walk-forward, no resultados de su hard holdout posterior. K es una
evaluación retrospectiva con diseño adaptado a esa ventana: `oos_shock` es
posterior al freeze de J, no un holdout ciego de K. Su comparador J combina
replay previo al freeze con predicciones oficiales posteriores; no son runs
de entrenamiento productivo nuevos. El protocolo K declara tres seeds mientras
las configuraciones de variantes enumeran cinco; se preservan ambas fuentes
sin afirmar cuántas se ejecutaron realmente para cada agregado.

L solo tiene dos ciclos retrospectivos con verdad incompleta: 47/48 y 46/48;
el agregado cubre 93/96. No es la clasificación oficial completa ni una
validación prospectiva. No comparar directamente su 91.9613% con protocolos
anteriores. No se importan resultados productivos en un experimento común.

## Versionamiento e idempotencia

`import_id` es SHA-256 del esquema del importador, la fila normalizada, su
localizador y los hashes completos de los archivos de origen. Repetir la
importación reutiliza runs y verifica valores, parámetros y copias de fuentes.
Modificar una fuente genera otra revisión histórica, conservando la anterior;
recommitear bytes idénticos no duplica runs. Un lock local serializa importadores.
Un run interrumpido se retoma con su mismo ID; no se considera completo hasta
terminar y verificarlo. Una modificación manual de sus métricas produce un
error de verificación, no un éxito silencioso. La importación no borra runs.

Tags: modelo/variante, versión conocida o `not_reported`, evaluación, fuente,
localizador, hash, último commit que modificó el archivo y fecha del reporte
cuando existe. `source_file_commit` no es el commit de entrenamiento:
`training_commit=not_reported` lo distingue explícitamente. Para L se conserva
el commit de la especificación congelada. Los commits de fuentes pueden faltar
en un ZIP sin `.git`; los hashes siguen permitiendo reproducir el contenido.

Supabase mantiene su registro operacional `model_versions` (`model_id`,
`model_name`, `model_type`, `artifact_uri`, `trained_until`, `feature_config`,
`params`, `is_active`), ligado a `pipeline_runs`. MLflow no escribe en él ni
reinterpreta su UUID como un ID de run MLflow. Las versiones reportadas y las
referencias del reporte permiten relacionar ambos, sin inventar un UUID para
experimentos que nunca se registraron. No se implementa MLflow Model Registry:
no es necesario para comparar reportes ni para la evidencia y duplicaría la
responsabilidad del registro operativo.

El backend local es `.mlflow/mlflow.db` (SQLite) y los artefactos están en
`.mlflow/artifacts/`. Ambos quedan fuera de Git. No subir SQLite, virtualenvs,
credenciales ni datos privados. El backend y sus artefactos deben conservarse
juntos; las URI de artefactos son absolutas. Para otro equipo es preferible
reconstruir desde los reportes, en vez de copiar SQLite con rutas de Codex.

## Evidencia auténtica y datos generados en Codex

Captura: `evidence/mlflow-experiments.png`, tomada en la UI real de MLflow.
También se incluyen `evidence/mlflow-parameters.png` (trials e hiperparámetros C)
y `evidence/mlflow-catalog.png` (catálogo de experimentos).
Para reproducirla en Mac: iniciar con los comandos anteriores, seleccionar
`Pulso / D1 / walk-forward`, abrir la tabla de runs y mostrar columnas
`mean_station_accuracy`, `wape`, `mae`, `bias` y parámetros. Abrir un run muestra
Params, Tags y Artifacts. Otra captura puede mostrar la lista de los 16
experimentos. Usar Cmd-Shift-4 para guardar la evidencia real.

Los datos necesarios son **los reportes ya versionados**: no hay que descargar
una base SQLite desde Codex. Obtener la rama publicada y reconstruir:

```bash
git clone --branch feat/mlflow-historical-tracking https://github.com/angiedonatto/pulso-transmi-mlops.git
cd pulso-transmi-mlops
# ejecutar la instalación y los dos imports indicados arriba
```

Si ya existe el checkout, usar `git fetch origin` y
`git switch --track origin/feat/mlflow-historical-tracking` (sin sobrescribir
cambios locales). La captura y el manifiesto JSON se obtienen con la misma rama
o se pueden descargar desde los archivos de esta tarea.

## Validación

```bash
MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false python -m pytest -q tests/test_mlflow_history.py
```

Las pruebas usan SQLite temporal real: valores originales, aislamiento de
protocolos, persistencia entre clientes, importación doble sin duplicados,
recuperación de runs interrumpidos, detección de métricas manipuladas y revisión
de fuentes. La importación verifica cada run contra los reportes y sus hashes.
Consultar `evidence/mlflow-validation.json` para los resultados de esta ejecución.
