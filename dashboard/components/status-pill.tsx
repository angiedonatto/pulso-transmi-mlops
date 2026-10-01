import { AlertTriangle, CheckCircle2, Clock3, CircleSlash, RadioTower, XCircle } from "lucide-react";
import type { Status } from "@/lib/types";

const styles: Record<string, string> = {
  HEALTHY: "border-emerald-400/30 bg-emerald-400/12 text-emerald-200",
  OK: "border-emerald-400/30 bg-emerald-400/12 text-emerald-200",
  PASSED: "border-emerald-400/30 bg-emerald-400/12 text-emerald-200",
  ACCEPTED: "border-cyan-300/30 bg-cyan-300/12 text-cyan-100",
  WATCH: "border-amber-300/35 bg-amber-300/12 text-amber-100",
  WAITING: "border-blue-300/30 bg-blue-300/12 text-blue-100",
  "NOT RESOLVED": "border-slate-300/20 bg-slate-300/10 text-slate-200",
  INSUFFICIENT_DATA: "border-slate-300/20 bg-slate-300/10 text-slate-200",
  UNKNOWN: "border-slate-300/20 bg-slate-300/10 text-slate-200",
  DRIFT: "border-red-400/35 bg-red-400/12 text-red-100",
  FAILED: "border-red-400/35 bg-red-400/12 text-red-100",
};

const labels: Record<string, string> = {
  HEALTHY: "SANO",
  OK: "OK",
  PASSED: "APROBADO",
  ACCEPTED: "ACEPTADO",
  WATCH: "ALERTA",
  WAITING: "EN ESPERA",
  "NOT RESOLVED": "SIN RESOLVER",
  INSUFFICIENT_DATA: "DATOS INSUF.",
  UNKNOWN: "SIN DATO",
  DRIFT: "DRIFT",
  FAILED: "FALLA",
};

export function StatusPill({ status, compact = false }: { status: Status | string; compact?: boolean }) {
  const normalized = String(status || "UNKNOWN").toUpperCase();
  const Icon = iconFor(normalized);
  return (
    <span className={`${styles[normalized] || styles.UNKNOWN} inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.08em]`}>
      <Icon className={compact ? "size-3" : "size-3.5"} />
      {labels[normalized] ?? normalized.replace("_", " ")}
    </span>
  );
}

function iconFor(status: string) {
  if (["HEALTHY", "OK", "PASSED", "ACCEPTED"].includes(status)) return CheckCircle2;
  if (["WATCH", "WAITING"].includes(status)) return Clock3;
  if (["DRIFT", "FAILED"].includes(status)) return XCircle;
  if (status === "INSUFFICIENT_DATA") return CircleSlash;
  return AlertTriangle || RadioTower;
}
