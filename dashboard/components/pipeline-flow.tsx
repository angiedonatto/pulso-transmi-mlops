"use client";
import { useState } from "react";
import { Database, GitBranch, Send, Activity, FlaskConical } from "lucide-react";
const steps = [
 { title: "Ingesta", icon: Database, detail: "Observaciones y contexto se persisten con cortes temporales. La interfaz no descarga observaciones crudas." },
 { title: "Modelado", icon: GitBranch, detail: "El registro conserva versiones y un único champion activo. Los experimentos históricos tienen protocolos de evaluación separados." },
 { title: "Entrega", icon: Send, detail: "Predicciones y receipts permiten distinguir targets generados de submissions oficialmente aceptadas." },
 { title: "Monitoreo", icon: Activity, detail: "drift_monitor.py calcula las métricas canónicas; monitoring_snapshots conserva el estado y sus alertas." },
 { title: "Evidencia", icon: FlaskConical, detail: "Los reportes versionados permiten reconstruir el tracking histórico en MLflow sin reentrenar." },
];
export function PipelineFlow() {
 const [selected,setSelected]=useState(0);
 return <section className="pipeline-section" aria-label="Arquitectura MLOps">
  <div className="section-heading"><div><p className="eyebrow">01 / EL SISTEMA</p><h2>Del dato a la evidencia.</h2></div><span className="caption">Esquema de arquitectura · no actividad en vivo</span></div>
  <div className="pipeline-flow">{steps.map((step,i)=><button key={step.title} className="pipeline-step" aria-pressed={selected===i} aria-controls="pipeline-detail" onClick={()=>setSelected(i)}><span className="pipeline-object"><span className="pipeline-layer"/><step.icon size={22}/></span><span className="pipeline-index">0{i+1}</span><strong>{step.title}</strong></button>)}</div>
  <p id="pipeline-detail" className="pipeline-detail" aria-live="polite"><strong>{steps[selected].title}.</strong> {steps[selected].detail}</p>
 </section>;
}
