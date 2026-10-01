"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, BarChart3, Boxes, Database, GitBranch, Home, RadioTower, Route, ServerCog } from "lucide-react";
import { StatusPill } from "@/components/status-pill";
import type { MonitoringSnapshot } from "@/lib/types";
import { shortDate } from "@/lib/format";

const nav = [
  { href: "/", label: "Vista general", icon: Home },
  { href: "/cycles", label: "Ciclos", icon: Route },
  { href: "/accuracy", label: "Accuracy", icon: BarChart3 },
  { href: "/drift", label: "Drift", icon: RadioTower },
  { href: "/data", label: "Datos", icon: Database },
  { href: "/models", label: "Modelos", icon: Boxes },
  { href: "/operations", label: "Operaciones", icon: ServerCog },
];

export function Shell({ children, snapshot }: { children: React.ReactNode; snapshot: MonitoringSnapshot & { source: "supabase" | "demo" } }) {
  const pathname = usePathname();
  const routeOwnsRealData = (pathname === "/models" || pathname === "/accuracy") && snapshot.source === "demo";
  const shellStatus = routeOwnsRealData ? "UNKNOWN" : snapshot.overall_status;
  const driftStatus = routeOwnsRealData ? "UNKNOWN" : snapshot.drift_status;
  const sourceLabel = routeOwnsRealData ? "sin configurar" : snapshot.source === "demo" ? "datos de muestra" : "Supabase";

  return (
    <div className="relative min-h-screen">
      <div className="ops-grid pointer-events-none absolute inset-0" />
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-64 border-r border-white/10 bg-[#080b10]/92 px-4 py-5 backdrop-blur-xl lg:block">
        <div className="flex items-center gap-3">
          <div className="grid size-10 place-items-center rounded-lg bg-cyan-300/12 text-cyan-100 ring-1 ring-cyan-200/20">
            <GitBranch className="size-5" />
          </div>
          <div>
            <p className="text-sm font-semibold text-white">Pulso TransMi</p>
            <p className="text-xs text-slate-400">Centro de control MLOps</p>
          </div>
        </div>
        <nav className="mt-8 space-y-1">
          {nav.map((item) => (
            <Link key={item.href} href={item.href} className="flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium text-slate-300 transition hover:bg-white/[0.06] hover:text-white">
              <item.icon className="size-4 text-slate-500" />
              {item.label}
            </Link>
          ))}
        </nav>
        <div className="absolute bottom-5 left-4 right-4 rounded-lg border border-white/10 bg-white/[0.04] p-3">
          <p className="text-[11px] uppercase tracking-[0.14em] text-slate-500">Instantánea</p>
          <div className="mt-2 flex items-center justify-between gap-2">
            <StatusPill status={shellStatus} compact />
            <span className="text-xs text-slate-400">{sourceLabel}</span>
          </div>
          <p className="mt-2 text-xs text-slate-400">{shortDate(snapshot.generated_at)}</p>
        </div>
      </aside>
      <main className="relative z-10 px-4 py-4 lg:ml-64 lg:px-8 lg:py-6">
        <header className="mb-6 flex flex-col gap-4 rounded-lg border border-white/10 bg-[#10151d]/76 p-4 backdrop-blur md:flex-row md:items-center md:justify-between">
          <div>
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-cyan-100">
              <Activity className="size-3.5" />
              Observabilidad operativa
            </div>
            <h1 className="mt-2 text-2xl font-semibold text-white md:text-3xl">Centro de control MLOps</h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill status={shellStatus} />
            <StatusPill status={driftStatus} />
          </div>
        </header>
        {children}
      </main>
    </div>
  );
}
