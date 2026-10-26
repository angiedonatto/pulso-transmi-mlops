# Pulso TransMi — dashboard de observabilidad

Next.js App Router existente, con lecturas de Supabase exclusivamente en servidor.
Conserva su identidad visual. No ejecuta ingesta, entrenamiento, submissions ni
cambios en el registro de modelos.

## Desarrollo y validación

Usar Node.js 24 y trabajar desde `dashboard/`:

```bash
npm ci
cp .env.example .env.local
# Introducir SUPABASE_SERVICE_ROLE_KEY de forma privada en .env.local.
npm run typecheck
npm run lint
npm run build
npm test
npm run dev -- --port 3000
```

No ejecutar build y dev simultáneamente porque ambos escriben `.next/`.
`npm test` requiere un build previo y usa servidores locales con datos
**sintéticos controlados**, nunca credenciales ni datos de producción. Arranca y
cierra sus propios servidores. Prueba las siete rutas renderizadas, fallos de
conexión/configuración/esquema, snapshots vacíos o malformados, datos
insuficientes, champion ambiguo, concordancia con métricas Python y ausencia de
claves en HTML/RSC y bundles del navegador.

Variables de Vercel:

- `NEXT_PUBLIC_SUPABASE_URL=https://pxayayhsmrzqjctlsrzs.supabase.co`
- `SUPABASE_SERVICE_ROLE_KEY`: introducir su valor real solo en ajustes seguros.
- `NEXT_PUBLIC_DASHBOARD_REFRESH_SECONDS=60` (intervalo entre 15 y 3600 segundos).

También se admiten los alias de servidor `SUPABASE_URL` y
`SUPABASE_SERVICE_KEY` para desarrollo. Nunca crear
`NEXT_PUBLIC_SUPABASE_SERVICE_ROLE_KEY` ni otra variable pública para secretos.
La clave no forma parte del cliente: los lectores importan `server-only`.

## Contrato de datos

`/`, `/accuracy`, `/drift`, `/cycles`, `/data` y `/operations` leen el último
`monitoring_snapshots` por `generated_at`. `/models` consulta además el registro
y metadatos de ciclos, predicciones, recibos y trazas con SELECT; identifica el
champion por una única fila con `is_active=true`, sin fijar Candidate G ni L
por nombre. Candidate L muestra sus parámetros armónicos persistidos.

Accuracy, WAPE, rolling windows y drift proceden de
`report.drift.performance` y demás secciones del snapshot, calculados por
`src/drift_monitor.py`. El frontend no descarga observations, no vuelve a
calcular scoring y no estima métricas ausentes. Si faltan valores muestra n/a;
la cobertura solo se usa cuando pertenece al mismo ciclo del snapshot.

El servidor consulta datos en cada petición (`force-dynamic`, fetch `no-store`),
con deduplicación del snapshot dentro de la petición y timeout de 8 segundos.
No se congelan errores ni datos demo al compilar. La actualización automática
vuelve a consultar el backend; **no hace actuales los datos históricos**.

Todas las páginas indican la fecha exacta UTC y que el estado pertenece al
snapshot. Más de 15 minutos de antigüedad, fechas futuras anómalas o fechas
inválidas disparan advertencia. Los errores, snapshots vacíos, contratos
incompatibles y ausencia de configuración son explícitos. No hay fallback demo.
Los detalles de error devueltos por Supabase no se exponen al navegador.

## Publicar en Vercel

1. En Vercel, **Add New → Project**, importar
   `angiedonatto/pulso-transmi-mlops`. Usar la rama
   `feat/dashboard-vercel` y **Root Directory: `dashboard`**,
   **Framework: Next.js**, **Node.js: 24.x**, instalación `npm ci`, build
   `npm run build`. Si el importador usa otra rama por defecto, configurar
   **Settings → Git → Production Branch: `feat/dashboard-vercel`** y desplegar
   de nuevo esa rama antes de evaluar el resultado.
2. En **Settings → Environment Variables**, guardar las tres variables anteriores
   para el entorno que se vaya a desplegar (Production y/o Preview). Introducir
   la service-role key únicamente allí. Ejecutar **Redeploy** después de guardar.
3. Para la URL que se vaya a entregar, revisar **Settings → Deployment Protection**
   y desactivar la autenticación Vercel que impida acceso público a ese deployment.
4. Abrir la URL HTTPS real en una ventana privada, sin sesión Vercel, y comprobar
   `/`, `/accuracy`, `/models` y `/drift`. Confirmar el champion que devuelva el
   registro, la fecha del snapshot y los avisos históricos. Un HTTP 200 con
   “Información no disponible” **no valida la conexión productiva**.

No se debe entregar una URL hasta comprobar acceso público y lecturas reales.
No es necesario cambiar workflows, migrar Supabase ni activar otro champion.

## Alcance de esta verificación

No había integración ni credenciales de Supabase/Vercel en el entorno de trabajo.
La conexión HTTPS al proyecto Supabase fue rechazada por el proxy de red con
403; por tanto, no se verificó el esquema productivo ni se efectuó despliegue.
Las consultas se contrastaron con las migraciones versionadas; las pruebas
controladas no sustituyen la comprobación de producción.

Las cifras y el snapshot de octubre proporcionados en la solicitud son
referencias históricas, no mediciones de esta ejecución, y no se hardcodean en
la aplicación. MLflow y el pipeline productivo permanecen intactos.
