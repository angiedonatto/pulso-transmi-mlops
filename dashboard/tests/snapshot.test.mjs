import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseSnapshot, snapshotIsStale } from '../lib/snapshot.ts';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/canonical-snapshot.json', import.meta.url)));

test('canonical Python snapshot is accepted without recomputing metrics', () => {
  const snapshot = parseSnapshot(fixture.snapshot);
  assert.ok(snapshot);
  assert.equal(snapshot.report.drift.performance.recent_accuracy, 78.125);
  assert.equal(snapshot.report.active_model.model_name, 'candidate_l_v1_recent_harmonic');
});

test('malformed snapshots fail closed and old or future dates are stale', () => {
  assert.equal(parseSnapshot({ report: null }), null);
  const malformed = structuredClone(fixture.snapshot);
  malformed.report.drift.data_drift.station_metrics = 'not an array';
  assert.equal(parseSnapshot(malformed), null);
  assert.equal(snapshotIsStale(fixture.snapshot.generated_at, Date.parse('2026-10-25T12:00:00Z')), true);
  assert.equal(snapshotIsStale('2026-10-25T12:00:00Z', Date.parse('2026-10-25T12:05:00Z')), false);
  assert.equal(snapshotIsStale('2026-10-25T13:00:00Z', Date.parse('2026-10-25T12:00:00Z')), true);
});
