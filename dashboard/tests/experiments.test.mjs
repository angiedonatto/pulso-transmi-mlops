import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync}from'node:fs';import{createHash}from'node:crypto';
const evidence=JSON.parse(readFileSync(new URL('../data/experiments.json',import.meta.url)));
test('historical evidence matches versioned sources and keeps evaluation protocols separate',()=>{
 for(const [relative,sha]of Object.entries(evidence.sources)){const file=new URL('../../'+relative,import.meta.url);assert.equal(createHash('sha256').update(readFileSync(file)).digest('hex'),sha);}
 assert.equal(evidence.tracking.runs,108);assert.equal(evidence.tracking.experiments,16);assert.equal(evidence.groups.length,5);
 const ab=JSON.parse(readFileSync(new URL('../../reports/model_improvement/summary.json',import.meta.url)));
 for(const row of evidence.groups[0].rows)assert.equal(row.accuracy,ab.models[row.model].mean_station_accuracy);
 const k=JSON.parse(readFileSync(new URL('../../reports/candidate_k/summary.json',import.meta.url)));
 for(const row of evidence.groups.at(-1).rows)assert.equal(row.accuracy,k.pooled.find(r=>r.window==='all_cycles'&&r.model===row.model).mean_station_accuracy);
});
