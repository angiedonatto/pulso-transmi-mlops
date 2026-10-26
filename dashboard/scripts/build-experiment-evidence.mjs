// Explicit offline export of small, versioned reports. No MLflow DB or secrets.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=fileURLToPath(new URL('../../',import.meta.url));
const sources={};
function read(relative){const bytes=readFileSync(path.join(root,relative));sources[relative]=createHash('sha256').update(bytes).digest('hex');return JSON.parse(bytes);}
const ab=read('reports/model_improvement/summary.json');
const c=read('reports/candidate_c/summary.json');
const d=read('reports/candidate_d1/summary.json');
const k=read('reports/candidate_k/summary.json');
const tracking=read('evidence/mlflow-validation.json');
const groups=[
 {name:'A / B · validación temporal',protocol:`${ab.validation_start} → ${ab.validation_end}`,source:'reports/model_improvement/summary.json',rows:Object.entries(ab.models).map(([model,r])=>({model,accuracy:r.mean_station_accuracy,wape:r.mean_station_wape}))},
 {name:'C · selección walk-forward',protocol:`${c.walk_forward_start} → ${c.walk_forward_end}. Scores de selección, no holdout.`,source:'reports/candidate_c/summary.json',rows:c.trials.map(r=>({model:`${r.family} / ${r.trial}`,accuracy:r.accuracy,params:r.params}))},
 ...[['wf','D1 · walk-forward'],['hard','D1 · hard holdout']].map(([prefix,name])=>({name,protocol:prefix==='wf'?`${d.walk_forward_start} → ${d.walk_forward_end}`:`${d.hard_holdout_start} → ${d.hard_holdout_end}`,source:'reports/candidate_d1/summary.json',rows:d.comparison.map(r=>({model:r.model,accuracy:r[prefix+'_mean_station_accuracy'],wape:r[prefix+'_wape'],mae:r[prefix+'_mae'],bias:r[prefix+'_bias']}))})),
 {name:'K · backtest retrospectivo / all_cycles',protocol:'Diseño adaptado a la ventana. Comparador J: replay y predicciones oficiales. No es holdout ciego de K.',source:'reports/candidate_k/summary.json',rows:k.pooled.filter(r=>r.window==='all_cycles').map(r=>({model:r.model,accuracy:r.mean_station_accuracy,wape:r.global_wape,mae:r.mae,bias:r.signed_bias,params:k.variants[r.model]??null}))},
];
for(const g of groups)for(const r of g.rows)if(!Number.isFinite(r.accuracy))throw Error('Missing historical accuracy: '+JSON.stringify(r));
const output={notice:'Evidencia histórica versionada. No son métricas actuales ni entrenamientos nuevos. Comparar únicamente dentro de cada protocolo. WAPE de K es global; el de AB es medio por estación.',tracking:{experiments:Object.keys(tracking.first_import.experiments).length,runs:tracking.first_import.verified},sources,groups};
writeFileSync(path.join(root,'dashboard/data/experiments.json'),JSON.stringify(output,null,2)+'\n');
