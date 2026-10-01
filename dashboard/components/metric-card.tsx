import type { LucideIcon } from "lucide-react";
import { StatusPill } from "@/components/status-pill";

export function MetricCard({
  label,
  value,
  detail,
  status,
  icon: Icon,
}: {
  label: string;
  value: string;
  detail?: string;
  status?: string;
  icon: LucideIcon;
}) {
  return (
    <section className="min-h-[132px] rounded-lg border border-white/10 bg-[#10151d]/88 p-4 shadow-[0_18px_60px_rgba(0,0,0,0.22)]">
      <div className="flex items-start justify-between gap-3">
        <div className="rounded-md border border-white/10 bg-white/[0.04] p-2 text-cyan-100">
          <Icon className="size-4" />
        </div>
        {status ? <StatusPill status={status} compact /> : null}
      </div>
      <p className="mt-4 text-xs font-medium uppercase tracking-[0.16em] text-slate-400">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-white">{value}</p>
      {detail ? <p className="mt-2 text-sm leading-5 text-slate-400">{detail}</p> : null}
    </section>
  );
}
