"use client";

import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { GroupMetric, PerformancePoint } from "@/lib/performance";

type TooltipValue = string | number | Array<string | number> | undefined;
type ChartDatum = Record<string, string | number | null>;

export function PerformanceCharts({ history, stations, horizons }: { history: PerformancePoint[]; stations: GroupMetric[]; horizons: GroupMetric[] }) {
  const historyData = history.map((point, index) => ({
    ...point,
    cycle_label: cycleLabel(point, index),
    cycle_order: `Ciclo ${index + 1}`,
  }));
  const stationData = stations.slice(0, 12).map((station) => ({
    ...station,
    fill: accuracyColor(station.accuracy),
  }));
  const horizonData = horizons.map((horizon) => ({
    ...horizon,
    fill: accuracyColor(horizon.accuracy),
  }));
  const latest = history.at(-1);
  const weakestStation = stations[0] ?? null;
  const strongestStation = stations.filter((station) => station.accuracy !== null).at(-1) ?? null;

  return (
    <section className="grid gap-4 xl:grid-cols-2">
      <ChartPanel
        title="Tendencia del champion"
        subtitle="Cada punto es un ciclo oficial resuelto"
        legend={[
          { label: "Accuracy del ciclo", color: "#41d19b" },
          { label: "Tendencia 6 ciclos", color: "#4dbed5" },
        ]}
        insight={latest?.accuracy === null || latest?.accuracy === undefined ? "Sin último ciclo resuelto." : `Último ciclo: ${latest.accuracy.toFixed(1)}% accuracy.`}
        axisHelp="Y: accuracy (%) · X: ciclos en orden temporal"
      >
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={historyData} margin={{ top: 8, right: 12, bottom: 10, left: -4 }}>
            <defs>
              <linearGradient id="accuracyFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="5%" stopColor="#41d19b" stopOpacity={0.42} />
                <stop offset="95%" stopColor="#41d19b" stopOpacity={0} />
              </linearGradient>
            </defs>
            <Grid />
            <XAxis dataKey="cycle_label" tick={tick} tickLine={false} axisLine={false} minTickGap={16} />
            <YAxis domain={[0, 100]} tick={tick} tickLine={false} axisLine={false} tickFormatter={(value) => `${value}%`} />
            <ReferenceLine y={85} stroke="rgba(255,255,255,0.22)" strokeDasharray="4 4" />
            <ChartTooltip suffix="%" />
            <Area type="monotone" dataKey="accuracy" name="Accuracy del ciclo" stroke="#41d19b" strokeWidth={2.5} fill="url(#accuracyFill)" activeDot={{ r: 5 }} />
            <Line type="monotone" dataKey="rolling6_accuracy" name="Tendencia 6 ciclos" stroke="#4dbed5" strokeWidth={2} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartPanel>

      {history.some((point) => point.resolved_targets !== null) ? (
      <ChartPanel
        title="Cobertura por ciclo"
        subtitle="Cuántos targets ya tienen observation real"
        legend={[
          { label: "Targets resueltos", color: "#6aa8ff" },
          { label: "Accuracy", color: "#41d19b" },
        ]}
        insight="Barras completas indican ciclos comparables; caídas de línea indican peor desempeño."
        axisHelp="Izquierda: targets · Derecha: accuracy (%)"
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={historyData} margin={{ top: 8, right: 8, bottom: 10, left: -4 }}>
            <Grid />
            <XAxis dataKey="cycle_label" tick={tick} tickLine={false} axisLine={false} minTickGap={16} />
            <YAxis yAxisId="left" tick={tick} tickLine={false} axisLine={false} />
            <YAxis yAxisId="right" orientation="right" domain={[0, 100]} tick={tick} tickLine={false} axisLine={false} tickFormatter={(value) => `${value}%`} />
            <ChartTooltip />
            <Bar yAxisId="left" dataKey="resolved_targets" name="Targets resueltos" fill="#6aa8ff" radius={[4, 4, 0, 0]} />
            <Line yAxisId="right" type="monotone" dataKey="accuracy" name="Accuracy" stroke="#41d19b" strokeWidth={2.5} dot={{ r: 3 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartPanel>
      ) : null}

      <ChartPanel
        title="Estaciones que más pesan"
        subtitle="Top 12 con menor accuracy"
        legend={[
          { label: "< 80%", color: "#ef6f6c" },
          { label: "80-88%", color: "#f1b84b" },
          { label: ">= 88%", color: "#41d19b" },
        ]}
        insight={weakestStation ? `Peor estación: ${weakestStation.label} (${formatPct(weakestStation.accuracy)}). Mejor: ${strongestStation?.label ?? "n/a"}.` : "Sin estaciones evaluables."}
        axisHelp="Y: accuracy (%) · X: estación"
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={stationData} margin={{ top: 8, right: 12, bottom: 22, left: -4 }}>
            <Grid />
            <XAxis dataKey="label" tick={tick} tickLine={false} axisLine={false} minTickGap={8} angle={-18} textAnchor="end" height={42} />
            <YAxis domain={[0, 100]} tick={tick} tickLine={false} axisLine={false} tickFormatter={(value) => `${value}%`} />
            <ReferenceLine y={80} stroke="rgba(239,111,108,0.55)" strokeDasharray="4 4" />
            <ReferenceLine y={88} stroke="rgba(65,209,155,0.45)" strokeDasharray="4 4" />
            <ChartTooltip suffix="%" />
            <Bar dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]}>
              {stationData.map((entry) => (
                <Cell key={entry.key} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel
        title="Horizonte de forecast"
        subtitle="Qué tan bien predice a 15, 30, 45 y 60 min"
        legend={[{ label: "Accuracy por horizonte", color: "#4dbed5" }]}
        insight={bestWorstHorizon(horizonData)}
        axisHelp="Y: accuracy (%) · X: minutos hacia adelante"
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={horizonData} margin={{ top: 8, right: 12, bottom: 10, left: -4 }}>
            <Grid />
            <XAxis dataKey="label" tick={tick} tickLine={false} axisLine={false} />
            <YAxis domain={[0, 100]} tick={tick} tickLine={false} axisLine={false} tickFormatter={(value) => `${value}%`} />
            <ReferenceLine y={85} stroke="rgba(255,255,255,0.22)" strokeDasharray="4 4" />
            <ChartTooltip suffix="%" />
            <Bar dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]}>
              {horizonData.map((entry) => (
                <Cell key={entry.key} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>
    </section>
  );
}

function ChartPanel({
  title,
  subtitle,
  legend,
  insight,
  axisHelp,
  children,
}: {
  title: string;
  subtitle: string;
  legend: Array<{ label: string; color: string }>;
  insight: string;
  axisHelp: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-white/10 bg-[#10151d]/88 p-4">
      <div className="mb-3 flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <h2 className="text-base font-semibold text-white">{title}</h2>
          <p className="mt-1 text-xs text-slate-400">{subtitle}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          {legend.map((item) => (
            <span key={item.label} className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.035] px-2 py-1 text-[11px] text-slate-300">
              <span className="size-2 rounded-full" style={{ backgroundColor: item.color }} />
              {item.label}
            </span>
          ))}
        </div>
      </div>
      <div className="h-[270px]">{children}</div>
      <div className="mt-3 grid gap-2 text-xs leading-5 text-slate-400 md:grid-cols-[1fr_auto]">
        <p className="rounded-md border border-white/10 bg-white/[0.03] px-3 py-2 text-slate-300">{insight}</p>
        <p className="rounded-md border border-white/10 bg-white/[0.03] px-3 py-2 font-mono text-[11px] text-slate-500">{axisHelp}</p>
      </div>
    </div>
  );
}

function Grid() {
  return <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />;
}

function ChartTooltip({ suffix = "" }: { suffix?: string }) {
  return (
    <Tooltip
      contentStyle={{ background: "#10151d", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, color: "#edf4f2" }}
      formatter={(value, name) => [formatValue(value as TooltipValue, suffix), String(name ?? "")]}
      labelFormatter={(label) => `Punto: ${String(label)}`}
      labelStyle={{ color: "#edf4f2" }}
    />
  );
}

function formatValue(value: TooltipValue, suffix: string): string {
  if (typeof value === "number" && Number.isFinite(value)) return `${value.toFixed(1)}${suffix}`;
  if (value === undefined) return "n/a";
  return String(value);
}

const tick = { fill: "#90a0ae", fontSize: 12 };

function accuracyColor(value: number | null): string {
  if (value === null) return "#7b8794";
  if (value < 80) return "#ef6f6c";
  if (value < 88) return "#f1b84b";
  return "#41d19b";
}

function formatPct(value: number | null): string {
  return value === null ? "n/a" : `${value.toFixed(1)}%`;
}

function cycleLabel(point: PerformancePoint, index: number): string {
  if (!point.accepted_at) return `c_${index + 1}`;
  const date = new Date(point.accepted_at);
  if (Number.isNaN(date.getTime())) return `c_${index + 1}`;
  return new Intl.DateTimeFormat("es-CO", { day: "2-digit", month: "short", hour: "2-digit" }).format(date);
}

function bestWorstHorizon(rows: ChartDatum[]): string {
  const valid = rows.filter((row) => typeof row.accuracy === "number") as Array<ChartDatum & { accuracy: number }>;
  if (!valid.length) return "Sin horizontes evaluables.";
  const best = valid.reduce((winner, row) => (row.accuracy > winner.accuracy ? row : winner), valid[0]);
  const worst = valid.reduce((winner, row) => (row.accuracy < winner.accuracy ? row : winner), valid[0]);
  return `Mejor horizonte: ${best.label} (${best.accuracy.toFixed(1)}%). Más débil: ${worst.label} (${worst.accuracy.toFixed(1)}%).`;
}
