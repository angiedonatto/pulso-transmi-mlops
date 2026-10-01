# AGENTS.md

Guia operativa para agentes que trabajen en este repositorio.

## Contexto del proyecto

Pulso TransMi es un proyecto MLOps para pronosticar demanda cada 15 minutos y operar un pipeline de ingesta, prediccion, submission oficial, monitoreo de drift y dashboard de observabilidad.

El repo combina:

- SDK y pipeline Python en `src/`.
- Scripts productivos y experimentales en `scripts/`.
- Schema Supabase en `supabase/migrations/`.
- Dashboard Next.js en `dashboard/`.
- Documentacion en `docs/`.

## Reglas criticas

- No cambiar parametros de Candidate G: `ALPHA`, `FEEDBACK_WINDOW`, `CLIP_LOW`, `CLIP_HIGH`, EWMA, ni la logica de calibracion salvo que el usuario lo pida explicitamente.
- No cambiar el payload oficial de submission ni el scoring oficial.
- No reimplementar metricas de drift o accuracy en el frontend. La fuente canonica es `src/drift_monitor.py`.
- No exponer secretos. Nunca crear variables `NEXT_PUBLIC_*` para service-role keys, API keys o tokens.
- No descargar observaciones completas desde el frontend. El dashboard debe leer `monitoring_snapshots`.
- No hacer commit, push o deploy si el usuario no lo pidio explicitamente.

## Flujo de monitoreo

La ruta canonica es:

1. Supabase guarda tablas operativas existentes.
2. `scripts/run_monitoring_snapshot.py` reconstruye el estado.
3. `src/monitoring_snapshot.py` reutiliza `build_drift_report(...)` de `src/drift_monitor.py`.
4. El resultado se persiste en `monitoring_snapshots`.
5. `dashboard/` lee solo el ultimo snapshot.

Candidate G guarda observabilidad en `prediction_traces`. Esa tabla no afecta el payload oficial.

## Comandos de verificacion

Python:

```bash
PYTHONPATH=.:src pytest -q
```

Dashboard:

```bash
cd dashboard
npm install
npm run typecheck
npm run build
```

Servidor local:

```bash
cd dashboard
npm run dev -- --port 3000
```

## Areas importantes

- `src/cycle_predictions.py`: generacion productiva de predicciones y persistencia de traces.
- `src/candidate_g_adaptive.py`: Candidate G y reconstruccion de feedback historico.
- `src/drift_monitor.py`: metricas canonicas de accuracy, drift, bias, PSI, KS y retrain recommendation.
- `src/monitoring_snapshot.py`: snapshot operacional para dashboard.
- `scripts/run_monitoring_snapshot.py`: genera y persiste snapshots.
- `dashboard/`: control center Next.js para Vercel.
- `docs/monitoring-architecture.md`: arquitectura del monitoreo.

## Supabase y seguridad

Las migraciones nuevas deben ser idempotentes con `if not exists` cuando sea razonable.

RLS esta habilitado y las tablas operativas revocan acceso a `anon` y `authenticated`. Si el dashboard necesita service role, debe usarse solo en Server Components o server-side code de Next.js.

Variables esperadas para dashboard en Vercel:

- `NEXT_PUBLIC_SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `NEXT_PUBLIC_DASHBOARD_REFRESH_SECONDS`

Variables productivas de pipeline:

- `PULSO_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`

## Trabajo con cambios existentes

Este repo puede tener muchos archivos sin trackear en `reports/`, `scripts/` y `tests/` que no pertenecen al cambio actual. No borrarlos ni revertirlos. Trabaja solo sobre los archivos necesarios para la tarea.
