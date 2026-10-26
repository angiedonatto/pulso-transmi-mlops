"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { Activity, BarChart3, Boxes, Database, GitBranch, Home, RadioTower, Route, ServerCog } from "lucide-react";
import { StatusPill } from "@/components/status-pill";
import type { SnapshotResult } from "@/lib/types";
import { utcTimestamp } from "@/lib/format";

const nav = [
  { href: "/", label: "Vista general", icon: Home },
  { href: "/cycles", label: "Ciclos", icon: Route },
  { href: "/accuracy", label: "Accuracy", icon: BarChart3 },
  { href: "/drift", label: "Drift", icon: RadioTower },
  { href: "/data", label: "Datos", icon: Database },
  { href: "/experiments", label: "Experimentación", icon: GitBranch },
  { href: "/models", label: "Modelos", icon: Boxes },
  { href: "/operations", label: "Operaciones", icon: ServerCog },
];

export function Shell({ children, monitoring, stale }: { children: React.ReactNode; monitoring: SnapshotResult; stale: boolean }) {
  const pathname = usePathname();
  const router = useRouter();
  const snapshot = monitoring.snapshot;
  const shellStatus = snapshot?.overall_status ?? "UNKNOWN";

  const sourceLabel = snapshot ? "Supabase · histórico" : "sin datos";
  useEffect(() => {
    const configured = Number(process.env.NEXT_PUBLIC_DASHBOARD_REFRESH_SECONDS || 60);
    const seconds = Number.isFinite(configured) ? Math.min(3600, Math.max(15, configured)) : 60;
    const timer = setInterval(() => { if (document.visibilityState === "visible") router.refresh(); }, seconds * 1000);
    return () => clearInterval(timer);
  }, [router]);

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
            <Link key={item.href} href={item.href} aria-current={pathname === item.href ? "page" : undefined} className="flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium text-slate-300 transition hover:bg-white/[0.06] hover:text-white">
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
          <p className="mt-2 text-xs text-slate-400">{utcTimestamp(snapshot?.generated_at)}</p>
        </div>
      </aside>
      <main className="relative z-10 px-4 py-4 lg:ml-64 lg:px-8 lg:py-6">
        <header className="mb-6 flex flex-col gap-4 rounded-lg border border-white/10 bg-[#10151d]/76 p-4 backdrop-blur md:flex-row md:items-center md:justify-between">
          <div>
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-cyan-100">
              <Activity className="size-3.5" />
              Observabilidad operativa
            </div>
            <h1 className="mt-2 text-2xl font-semibold text-white md:text-3xl">Plataforma MLOps</h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="caption">Lectura de snapshots · solo consulta</span>

          </div>
        </header>
        <nav aria-label="Navegación móvil" className="mb-4 flex flex-wrap gap-3 text-sm text-cyan-100 lg:hidden">
          {nav.map((item) => <Link key={item.href} href={item.href} aria-current={pathname === item.href ? "page" : undefined}>{item.label}</Link>)}
        </nav>
        <section role="status" className="mb-5 rounded-lg border border-amber-300/25 bg-amber-300/[0.06] p-4 text-sm text-slate-300">
          {snapshot ? <>
            <p className="font-semibold text-amber-100">{stale ? "Datos desactualizados" : "Snapshot histórico"} · no representa tiempo real</p>
            <p className="mt-1">Snapshot generado: {utcTimestamp(snapshot.generated_at)}. {pathname === "/experiments" ? "Esta fecha corresponde al monitoreo operacional. Cada experimento conserva su propio periodo de evaluación." : "Los estados, métricas y alertas corresponden a ese momento."}</p>
          </> : <p>{monitoring.message}</p>}
          <button onClick={() => router.refresh()} className="mt-2 text-cyan-100 underline">Volver a consultar</button>
        </section>
        {children}
      </main>
    </div>
  );
}
