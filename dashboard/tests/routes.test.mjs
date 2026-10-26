// Controlled PostgREST fixtures only. This test never contacts production Supabase.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { spawn } from 'node:child_process';
import { readFile, readdir } from 'node:fs/promises';
import { setTimeout as pause } from 'node:timers/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const fixture = JSON.parse(await readFile(new URL('./fixtures/canonical-snapshot.json', import.meta.url)));
const KEY = 'controlled-test-service-key-not-a-credential';
const routes = ['/', '/models', '/accuracy', '/drift', '/cycles', '/data', '/operations'];

async function freePort() {
  const server = http.createServer();
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const port = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function startApp(env) {
  const port = await freePort();
  let log = '';
  const child = spawn(process.execPath, ['node_modules/next/dist/bin/next', 'start', '--hostname', '127.0.0.1', '--port', String(port)], {
    cwd: ROOT, env: { ...process.env, NEXT_TELEMETRY_DISABLED: '1', SUPABASE_URL: '', SUPABASE_SERVICE_KEY: '', NEXT_PUBLIC_SUPABASE_URL: '', SUPABASE_SERVICE_ROLE_KEY: '', ...env },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout.on('data', (data) => { log += data; });
  child.stderr.on('data', (data) => { log += data; });
  const base = `http://127.0.0.1:${port}`;
  for (let attempt = 0; attempt < 80; attempt++) {
    if (child.exitCode !== null) throw new Error(`Next failed to start: ${log}`);
    try {
      const response = await fetch(base);
      if (response.status === 200) return { base, stop: async () => { child.kill('SIGTERM'); await once(child, 'exit'); } };
    } catch { /* wait for startup */ }
    await pause(200);
  }
  child.kill('SIGTERM');
  throw new Error(`Next startup timed out: ${log}`);
}

async function html(app, route) {
  const response = await fetch(app.base + route);
  assert.equal(response.status, 200, route);
  const text = await response.text();
  assert.ok(text.includes('Centro de control MLOps'), route);
  assert.ok(!text.includes(KEY), 'server key must not reach HTML/RSC');
  assert.ok(!text.includes('data of undefined'), route);
  return text;
}

async function files(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  return (await Promise.all(entries.map((e) => e.isDirectory() ? files(path.join(directory, e.name)) : [path.join(directory, e.name)]))).flat();
}

test('production Next routes use real query paths with controlled fixtures', { timeout: 120000 }, async (t) => {
  let mode = 'healthy';
  const requests = [];
  const mock = http.createServer((request, response) => {
    const url = new URL(request.url, 'http://localhost');
    const table = url.pathname.split('/').at(-1);
    requests.push({ method: request.method, table, select: url.searchParams.get('select') });
    response.setHeader('content-type', 'application/json');
    if (!['GET', 'HEAD'].includes(request.method) || table === 'observations' || request.headers.apikey !== KEY) {
      response.writeHead(403); response.end(JSON.stringify({ message: 'Forbidden test operation' })); return;
    }
    if (mode === 'denied') {
      response.writeHead(401); response.end(JSON.stringify({ code: 'PGRST301', message: 'Controlled authentication failure' })); return;
    }
    if (mode === 'schema') {
      response.writeHead(400); response.end(JSON.stringify({ code: '42703', message: 'Controlled missing column' })); return;
    }
    let rows = [];
    if (table === 'monitoring_snapshots') {
      const snapshot = structuredClone(fixture.snapshot);
      if (mode === 'insufficient') snapshot.report.drift.performance = { status: 'INSUFFICIENT_DATA', cycle_count: 0 };
      rows = mode === 'empty' ? [] : [mode === 'invalid' ? { ...snapshot, report: null } : snapshot];
    } else if (table === 'model_versions') {
      rows = structuredClone(fixture.models);
      if (mode === 'multiple') rows.push({ ...rows[0], model_id: 'another-active-id' });
      const name = url.searchParams.get('model_name');
      if (name?.startsWith('eq.')) rows = rows.filter((row) => row.model_name === name.slice(3));
    } else if (!['predictions', 'submission_receipts', 'forecast_cycles', 'forecast_cycle_targets', 'prediction_traces'].includes(table)) {
      response.writeHead(404); response.end(JSON.stringify({ message: 'Unknown table' })); return;
    }
    if (request.method === 'HEAD') { response.setHeader('content-range', `*/${rows.length}`); response.end(); return; }
    response.end(JSON.stringify(rows));
  });
  mock.listen(0, '127.0.0.1');
  await once(mock, 'listening');
  t.after(() => new Promise((resolve) => mock.close(resolve)));
  const app = await startApp({ SUPABASE_URL: `http://127.0.0.1:${mock.address().port}`, SUPABASE_SERVICE_ROLE_KEY: KEY });
  t.after(app.stop);

  await t.test('all seven routes render; L, canonical accuracy and historical warnings are visible', async () => {
    for (const route of routes) {
      const content = await html(app, route);
      assert.match(content, /Datos desactualizados/);
      assert.match(content, /no representa tiempo real/);
      assert.match(content, /2026-01-20 00:00 UTC/);
    }
    assert.match(await html(app, '/models'), /candidate_l_v1_recent_harmonic/);
    const accuracy = await html(app, '/accuracy');
    assert.match(accuracy, /78\.1%/); // Canonical mean-station accuracy, NOT global WAPE accuracy (86%).
    assert.match(accuracy, /drift_monitor.py/);
    assert.match(await html(app, '/drift'), /DRIFT/);
  });
  for (const [state, expected] of [
    ['denied', /No se pudo leer monitoring_snapshots/], ['schema', /No se pudo leer monitoring_snapshots/],
    ['empty', /No hay snapshots/], ['invalid', /formato incompatible/],
  ]) {
    await t.test(`${state}: explicit unavailable state, never a demo`, async () => {
      mode = state;
      for (const route of ['/', '/accuracy', '/drift']) {
        const content = await html(app, route);
        assert.match(content, expected);
        assert.match(content, /Información no disponible/);
        assert.ok(!content.includes('87.4%'), 'do not display legacy demo accuracy');
      }
      if (['denied', 'schema'].includes(state)) assert.match(await html(app, '/models'), /Algunas lecturas fallaron/);
    });
  }
  await t.test('insufficient data remains unknown instead of manufactured scores', async () => {
    mode = 'insufficient';
    assert.match(await html(app, '/accuracy'), /Aún no hay historial evaluable/);
  });
  await t.test('multiple active rows never select an arbitrary champion', async () => {
    mode = 'multiple';
    assert.match(await html(app, '/models'), /No hay un champion activo único/);
  });
  assert.ok(requests.length > 0);
  assert.ok(requests.every((r) => ['GET', 'HEAD'].includes(r.method)));
  assert.ok(requests.every((r) => r.table !== 'observations'));
  for (const filename of (await files(path.join(ROOT, '.next/static'))).filter((f) => f.endsWith('.js'))) {
    const content = await readFile(filename, 'utf8');
    assert.ok(!content.includes(KEY));
    assert.ok(!content.includes('SUPABASE_SERVICE_ROLE_KEY'), 'service client must not be bundled in browser');
  }
});

test('missing or invalid Supabase configuration renders unavailable states on all routes', { timeout: 60000 }, async () => {
  for (const env of [{}, { SUPABASE_URL: 'invalid-url', SUPABASE_SERVICE_ROLE_KEY: KEY }]) {
    const app = await startApp(env);
    try {
      for (const route of routes) {
        const content = await html(app, route);
        assert.match(content, /Supabase no está configurado/);
      }
    } finally { await app.stop(); }
  }
});
