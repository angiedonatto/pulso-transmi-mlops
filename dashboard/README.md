# Pulso TransMi MLOps Control Center

Next.js App Router dashboard for operational observability.

## Local development

```bash
npm install
npm run dev
```

The app reads the latest row from `monitoring_snapshots`. If Supabase env vars are not present, it renders an explicit demo state so the UI can be reviewed safely.

## Vercel

Use `dashboard/` as the Vercel Root Directory.

Configure only these variable names:

- `NEXT_PUBLIC_SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `NEXT_PUBLIC_DASHBOARD_REFRESH_SECONDS`

Never expose `SUPABASE_SERVICE_ROLE_KEY` as a `NEXT_PUBLIC_*` variable.

## Pages

- `/` executive overview
- `/cycles` cycle health and official receipt state
- `/accuracy` official accuracy, rolling windows, station and horizon metrics
- `/drift` canonical drift monitor output
- `/data` freshness, stream cursors, quality and ingestion events
- `/models` active champion and Candidate G calibration trace
- `/operations` pipeline, alerts and runtime events
