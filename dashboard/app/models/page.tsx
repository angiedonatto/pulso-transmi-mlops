import type { LucideIcon } from "lucide-react";
import { AlertTriangle, CheckCircle2, CircleSlash, Database, GitCompareArrows, HelpCircle, Layers, Send, ShieldCheck, Timer, XCircle } from "lucide-react";
import { MetricCard } from "@/components/metric-card";
import { StatusPill } from "@/components/status-pill";
import { text, utcTimestamp } from "@/lib/format";
import { getModelsOverview, type Adaptation, type Check, type CheckState, type CycleComparison, type ModelVersion, type ModelsOverview, type ShadowHealth } from "@/lib/models";

export const revalidate = 60;

type ParamSpec = { label: string; source: "params" | "feature_config"; paths: string[] };

const CHAMPION_PARAMS: ParamSpec[] = [
  { label: "Base model", source: "feature_config", paths: ["base_model.name"] },
  { label: "Capa adaptativa", source: "params", paths: ["name", "adaptive_layer.name"] },
  { label: "Estimator", source: "params", paths: ["feedback_estimator", "adaptive_layer.feedback_estimator"] },
  { label: "Ventana feedback", source: "params", paths: ["feedback_window", "adaptive_layer.feedback_window"] },
  { label: "Alpha", source: "params", paths: ["alpha", "adaptive_layer.alpha"] },
  { label: "Clip", source: "params", paths: ["clip", "adaptive_layer.clip"] },
];

const SHADOW_PARAMS: ParamSpec[] = [
  { label: "Base model", source: "feature_config", paths: ["base_model.name"] },
  { label: "Level window", source: "params", paths: ["adaptive_layer.level_window", "level_window"] },
  { label: "Noise window", source: "params", paths: ["adaptive_layer.noise_window", "noise_window"] },
  { label: "Kappa", source: "params", paths: ["adaptive_layer.kappa", "kappa"] },
  { label: "Clip", source: "params", paths: ["adaptive_layer.clip", "clip"] },
  { label: "Level estimator", source: "params", paths: ["adaptive_layer.level_estimator", "level_estimator"] },
];

const CHECK_PILL: Record<CheckState, string> = { pass: "PASSED", watch: "WATCH", fail: "FAILED", unknown: "UNKNOWN" };

const panel = "rounded-lg border border-white/10 bg-[#10151d]/88 p-5";

export default async function ModelsPage() {
  const data = await getModelsOverview();

  return (
    <div className="space-y-6">
      <PageHeader data={data} />

      {data.source === "not_configured" ? (
        <Notice tone="slate" title="Supabase no está configurado">
          Configura SUPABASE_URL / SUPABASE_SERVICE_KEY o NEXT_PUBLIC_SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY. Esta página no usa datos demo.
        </Notice>
      ) : null}
      {data.issues.length ? (
        <Notice tone="amber" title="Algunas lecturas fallaron">
          <ul className="space-y-1">
            {data.issues.map((issue) => (
              <li key={`${issue.table}-${issue.message}`} className="break-words">
                <span className="font-mono text-amber-100">{issue.table}</span> · {issue.message}
              </li>
            ))}
          </ul>
        </Notice>
      ) : null}

      <section className="grid gap-4 xl:grid-cols-2">
        <ChampionCard data={data} />
        <ShadowCard model={data.shadow} />
      </section>

      <ComparisonSection comparison={data.comparison} />
      <ShadowStatusSection health={data.shadow_health} />
      <AdaptationSection adaptation={data.adaptation} />
    </div>
  );
}

function PageHeader({ data }: { data: ModelsOverview }) {
  const live = data.source === "supabase";
  return (
    <section className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
      <div className="min-w-0">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-100">Registro de modelos</p>
        <h2 className="mt-2 text-xl font-semibold text-white">Champion y shadow challenger</h2>
        <p className="mt-1 text-sm text-slate-400">Vista read-only de model_versions, predictions, forecast cycles y submission_receipts.</p>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-semibold uppercase tracking-[0.08em] ${
            live ? "border-emerald-400/30 bg-emerald-400/12 text-emerald-200" : "border-slate-300/20 bg-slate-300/10 text-slate-200"
          }`}
        >
          <Database className="size-3" />
          {live ? "Live · Supabase" : "No configurado"}
        </span>
        <span>Leído {utcTimestamp(data.fetched_at)}</span>
      </div>
    </section>
  );
}

function ChampionCard({ data }: { data: ModelsOverview }) {
  const model = data.champion;
  const health = data.champion_health;
  if (!model) {
    return (
      <EmptyCard
        badges={<RoleBadge tone="cyan">Champion</RoleBadge>}
        title="No hay un champion activo único"
        status={health.status}
        checks={health.checks}
      />
    );
  }
  return (
    <ModelCard
      model={model}
      badges={
        <>
          <RoleBadge tone="emerald">Activo</RoleBadge>
          <RoleBadge tone="cyan">Champion</RoleBadge>
        </>
      }
      subtitle="Modelo oficial · el único que puede hacer submissions"
      status={health.status}
      params={CHAMPION_PARAMS}
    >
      <div className="rounded-md border border-white/10 bg-white/[0.025] p-3">
        <div className="flex items-center justify-between gap-3">
          <SectionLabel>Salud del champion</SectionLabel>
          <span className="min-w-0 truncate font-mono text-[11px] text-slate-500" title={text(health.latest_cycle_id)}>
            {text(health.latest_cycle_id, "sin ciclo")}
          </span>
        </div>
        <CheckList checks={health.checks} className="mt-2" />
      </div>
    </ModelCard>
  );
}

function ShadowCard({ model }: { model: ModelVersion | null }) {
  if (!model) {
    return (
      <EmptyCard
        badges={<RoleBadge tone="violet">Shadow</RoleBadge>}
        title="Candidate I no está registrado"
        status="FAILED"
        checks={[{ label: "Registrado", state: "fail", detail: "No existe fila en model_versions para el shadow challenger." }]}
      />
    );
  }
  const layer = adaptiveLayer(model);
  const clip = Array.isArray(layer.clip) ? layer.clip : [];
  const sealedTest = at(model.params, "selection.sealed_test");
  return (
    <ModelCard
      model={model}
      badges={
        <>
          <RoleBadge tone="violet">Shadow</RoleBadge>
          <RoleBadge tone={model.is_active ? "red" : "slate"}>is_active={String(model.is_active)}</RoleBadge>
        </>
      }
      subtitle="Challenger · predice ciclos del champion, nunca hace submissions"
      params={SHADOW_PARAMS}
    >
      <div className="rounded-md border border-violet-300/15 bg-violet-300/[0.04] p-3">
        <SectionLabel>SNR gate</SectionLabel>
        <p className="mt-2 text-sm leading-6 text-slate-300">
          Base Candidate F × corrección de nivel aplicada solo cuando el cambio residual es significativo.{" "}
          <span className="font-mono text-[12px] text-slate-200">z</span> = |mean log(actual/base)| over the last {text(layer.level_window)} resolved slots ÷ its noise
          (sd over the last {text(layer.noise_window)});{" "}
          <span className="font-mono text-[12px] text-slate-200">gate = z² / (z² + {text(layer.kappa)})</span>; factor ={" "}
          <span className="font-mono text-[12px] text-slate-200">
            clip(exp(gate · log level), {text(clip[0])}, {text(clip[1])})
          </span>
          . Los cambios persistentes se corrigen; el ruido deja el factor ≈ 1.
        </p>
        {typeof sealedTest === "string" ? (
          <p className="mt-2 break-words text-xs leading-5 text-slate-500">
            Selección offline (registro, no live): {sealedTest}
          </p>
        ) : null}
      </div>
    </ModelCard>
  );
}

function ModelCard({
  model,
  badges,
  subtitle,
  status,
  params,
  children,
}: {
  model: ModelVersion;
  badges: React.ReactNode;
  subtitle: string;
  status?: string;
  params: ParamSpec[];
  children?: React.ReactNode;
}) {
  const version = [at(model.feature_config, "version"), at(model.feature_config, "mode")].filter(Boolean).join(" · ");
  const commit = at(model.feature_config, "git_commit") ?? at(model.feature_config, "artifact_git_commit");
  const paramValues = resolveParams(model, params);
  return (
    <article className={`${panel} flex h-full min-w-0 flex-col gap-5`}>
      <header className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap gap-1.5">{badges}</div>
          <h3 className="mt-3 truncate text-xl font-semibold text-white" title={model.model_name}>
            {model.model_name}
          </h3>
          <p className="mt-1 text-sm text-slate-400">{subtitle}</p>
        </div>
        {status ? <StatusPill status={status} /> : null}
      </header>

      <dl className="grid grid-cols-1 gap-x-4 gap-y-3 sm:grid-cols-2">
        <Field label="Model type" value={model.model_type} mono />
        <Field label="Entrenado hasta" value={utcTimestamp(model.trained_until)} />
        <Field label="Artifact URI" value={text(model.artifact_uri)} mono wide />
        <Field label="Model ID" value={model.model_id} mono />
        <Field label="Version" value={text(version)} />
        <Field label="Git commit" value={typeof commit === "string" ? commit : "n/a"} display={typeof commit === "string" ? commit.slice(0, 10) : "n/a"} mono />
      </dl>

      <div>
        <SectionLabel>Parámetros</SectionLabel>
        <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3">
          {paramValues.map((param) => (
            <div key={param.label} className="min-w-0 rounded-md border border-white/10 bg-white/[0.035] px-3 py-2.5">
              <p className="truncate text-[11px] uppercase tracking-[0.12em] text-slate-500">{param.label}</p>
              <p className="mt-1 line-clamp-2 break-all text-sm font-semibold text-white" title={param.value}>
                {param.value}
              </p>
            </div>
          ))}
        </div>
      </div>

      {children ? <div className="mt-auto">{children}</div> : null}
    </article>
  );
}

function EmptyCard({ badges, title, status, checks }: { badges: React.ReactNode; title: string; status: string; checks: Check[] }) {
  return (
    <article className={`${panel} flex h-full min-w-0 flex-col gap-4`}>
      <header className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap gap-1.5">{badges}</div>
          <h3 className="mt-3 text-xl font-semibold text-white">{title}</h3>
        </div>
        <StatusPill status={status} />
      </header>
      <CheckList checks={checks} />
    </article>
  );
}

function ComparisonSection({ comparison }: { comparison: CycleComparison | null }) {
  if (!comparison) {
    return (
      <section className={panel}>
        <SectionHeader icon={GitCompareArrows} eyebrow="Champion vs shadow" title="Último ciclo compartido" />
        <EmptyState title="Aún no hay ciclo compartido" detail="No encontré un forecast cycle con predictions tanto del champion activo como del shadow challenger." />
      </section>
    );
  }
  const { cycle, models } = comparison;
  const expected = cycle.expected_predictions;
  const resolved = comparison.resolved_targets;
  const fullyResolved = resolved !== null && resolved >= comparison.cycle_targets && comparison.cycle_targets > 0;
  return (
    <section className={panel}>
      <SectionHeader
        icon={GitCompareArrows}
        eyebrow="Champion vs shadow"
        title="Último ciclo compartido"
        right={
          <span className="max-w-full truncate rounded-md border border-white/10 bg-white/[0.04] px-2.5 py-1 font-mono text-xs text-slate-200" title={cycle.cycle_id}>
            {cycle.cycle_id}
          </span>
        }
      />
      <p className="mt-2 text-sm text-slate-400">
        Corte de datos {utcTimestamp(cycle.data_cutoff)} · cierra {utcTimestamp(cycle.closes_at)} · {expected} predictions esperadas
      </p>

      <div className="mt-4 overflow-x-auto rounded-lg border border-white/10">
        <table className="w-full min-w-[720px] border-collapse text-left text-sm">
          <thead className="bg-white/[0.04] text-[11px] uppercase tracking-[0.14em] text-slate-400">
            <tr>
              {["Modelo", "Rol", "Predictions", "Cobertura", "Targets", "Persistido en"].map((column) => (
                <th key={column} className="border-b border-white/10 px-4 py-3 font-semibold">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-white/8">
            {models.map((model) => {
              const coverage = expected > 0 ? Math.min(100, (model.predictions / expected) * 100) : 0;
              return (
                <tr key={model.model_id}>
                  <td className="max-w-[280px] truncate px-4 py-3 font-medium text-white" title={model.model_name}>
                    {model.model_name}
                  </td>
                  <td className="px-4 py-3">{model.role === "champion" ? <RoleBadge tone="cyan">Champion</RoleBadge> : <RoleBadge tone="violet">Shadow</RoleBadge>}</td>
                  <td className="px-4 py-3 tabular-nums text-slate-200">
                    {model.predictions} / {expected}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-white/10">
                        <div className={`h-full rounded-full ${coverage === 100 ? "bg-emerald-400" : "bg-amber-300"}`} style={{ width: `${coverage}%` }} />
                      </div>
                      <span className="tabular-nums text-slate-300">{coverage.toFixed(0)}%</span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <StatusPill status={model.aligned ? "OK" : "WATCH"} compact />
                  </td>
                  <td className="px-4 py-3 text-slate-300">{utcTimestamp(model.last_created_at)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        <Tile label="Targets compartidos" value={`${comparison.shared_targets} / ${comparison.cycle_targets}`} detail="Targets del shadow que también predijo el champion." />
        <Tile
          label="Targets resueltos"
          value={resolved === null ? "n/a" : `${resolved} / ${comparison.cycle_targets}`}
          detail={resolved === null ? "No se pudieron leer observations." : "Targets con observation real en Supabase."}
        />
        <Tile
          label="Accuracy G vs I"
          value="No disponible"
          detail={
            fullyResolved
              ? "Los targets están resueltos, pero Supabase no persiste accuracy shadow; el dashboard no la inventa ni la recalcula."
              : "Pendiente: no muestro accuracy hasta que todos los targets del ciclo tengan observation real."
          }
        />
      </div>
    </section>
  );
}

function ShadowStatusSection({ health }: { health: ShadowHealth }) {
  const byLabel = new Map(health.checks.map((check) => [check.label, check]));
  const pill = (label: string) => CHECK_PILL[byLabel.get(label)?.state ?? "unknown"];
  const lag = health.lag_cycles === null ? "n/a" : `${health.lag_is_lower_bound ? "≥" : ""}${health.lag_cycles}`;
  return (
    <section className={panel}>
      <SectionHeader icon={ShieldCheck} eyebrow="Shadow status" title="Ejecución shadow de Candidate I" right={<StatusPill status={health.status} />} />

      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard icon={Layers} label="Registro" value={byLabel.get("Registered")?.state === "pass" ? "Registrado" : "Faltante"} status={pill("Registered")} detail="model_versions, resuelto por model_name" />
        <MetricCard
          icon={CheckCircle2}
          label="Shadow predictions"
          value={health.predictions === null ? "n/a" : `${health.predictions} / ${text(health.expected_predictions)}`}
          status={pill("Shadow coverage")}
          detail="Último ciclo compartido con el champion"
        />
        <MetricCard
          icon={Send}
          label="Shadow receipts"
          value={health.receipts_total === null ? "n/a" : String(health.receipts_total)}
          status={pill("Official submissions")}
          detail={`${text(health.receipts_official)} submissions oficiales`}
        />
        <MetricCard icon={Timer} label="Ciclos detrás del champion" value={lag} status={pill("Freshness")} detail="El shadow corre por hora después del oficial" />
      </div>

      <CheckList checks={health.checks} className="mt-4 grid gap-2 md:grid-cols-2" />

      <p className="mt-4 text-xs leading-5 text-slate-500">
        HEALTHY: todos los checks pasan. WATCH: cobertura, alineación de targets o freshness degradada. FAILED: se rompió una invariante shadow (no registrado,
        activo, o con submission receipts). INSUFFICIENT DATA: aún no hay shadow predictions.
      </p>
    </section>
  );
}

function AdaptationSection({ adaptation }: { adaptation: Adaptation }) {
  return (
    <section className={panel}>
      <SectionHeader
        icon={Database}
        eyebrow="Adaptación reciente"
        title="Corrección de nivel por estación"
        right={adaptation.state === "available" ? <span className="max-w-full truncate font-mono text-xs text-slate-400" title={adaptation.cycle_id}>{adaptation.cycle_id}</span> : null}
      />
      {adaptation.state === "available" ? (
        <div className="mt-4 overflow-x-auto rounded-lg border border-white/10">
          <table className="w-full min-w-[640px] border-collapse text-left text-sm">
            <thead className="bg-white/[0.04] text-[11px] uppercase tracking-[0.14em] text-slate-400">
              <tr>
                {["Estación", "Factor I", "Gate I", "Z I", "Slots históricos", "Factor G"].map((column) => (
                  <th key={column} className="border-b border-white/10 px-4 py-3 font-semibold">
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/8">
              {adaptation.rows.map((row) => (
                <tr key={row.station_id} className="tabular-nums hover:bg-white/[0.03]">
                  <td className="px-4 py-3 font-mono text-white">{row.station_id}</td>
                  <td className="px-4 py-3 text-slate-200">{fixed(row.shadow_factor, 3)}</td>
                  <td className="px-4 py-3 text-slate-200">{adaptation.has_gate ? fixed(row.gate, 3) : <NotPersisted />}</td>
                  <td className="px-4 py-3 text-slate-200">{adaptation.has_z ? fixed(row.z, 2) : <NotPersisted />}</td>
                  <td className="px-4 py-3 text-slate-200">{fixed(row.history_slots, 0)}</td>
                  <td className="px-4 py-3 text-slate-200">{adaptation.has_champion ? fixed(row.champion_factor, 3) : <NotPersisted />}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyState
          title={adaptation.title}
          detail={adaptation.detail}
          note="Cuando existan filas en prediction_traces, este panel listará el factor de Candidate I por estación junto al de Candidate G para el último ciclo compartido. El schema actual de traces guarda base prediction, factor y slots históricos; gate y z solo salen en el log del workflow shadow."
        />
      )}
    </section>
  );
}

function SectionHeader({ icon: Icon, eyebrow, title, right }: { icon: LucideIcon; eyebrow: string; title: string; right?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="flex min-w-0 items-start gap-3">
        <div className="rounded-md border border-white/10 bg-white/[0.04] p-2 text-cyan-100">
          <Icon className="size-4" />
        </div>
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-100">{eyebrow}</p>
          <h2 className="mt-1 text-lg font-semibold text-white">{title}</h2>
        </div>
      </div>
      {right ? <div className="flex min-w-0 max-w-full items-center">{right}</div> : null}
    </div>
  );
}

function CheckList({ checks, className = "space-y-2" }: { checks: Check[]; className?: string }) {
  return (
    <ul className={className}>
      {checks.map((check) => {
        const { Icon, color } = checkIcon(check.state);
        return (
          <li key={check.label} className="flex min-w-0 items-start gap-2.5 rounded-md border border-white/[0.06] bg-white/[0.02] px-3 py-2">
            <Icon className={`mt-0.5 size-4 shrink-0 ${color}`} />
            <div className="min-w-0">
              <p className="text-sm font-medium text-white">{checkLabel(check.label)}</p>
              <p className="break-words text-xs leading-5 text-slate-400">{check.detail}</p>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

function Field({ label, value, display, mono = false, wide = false }: { label: string; value: string; display?: string; mono?: boolean; wide?: boolean }) {
  return (
    <div className={`min-w-0 ${wide ? "sm:col-span-2" : ""}`}>
      <dt className="text-[11px] uppercase tracking-[0.14em] text-slate-500">{label}</dt>
      <dd className={`mt-1 truncate text-slate-100 ${mono ? "font-mono text-[13px]" : "text-sm"}`} title={value}>
        {display ?? value}
      </dd>
    </div>
  );
}

function Tile({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="flex min-w-0 flex-col rounded-md border border-white/10 bg-white/[0.035] p-3">
      <p className="text-[11px] uppercase tracking-[0.14em] text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-semibold tabular-nums text-white">{value}</p>
      <p className="mt-1 text-xs leading-5 text-slate-400">{detail}</p>
    </div>
  );
}

function EmptyState({ title, detail, note }: { title: string; detail: string; note?: string }) {
  return (
    <div className="mt-4 flex items-start gap-3 rounded-lg border border-dashed border-white/15 bg-white/[0.02] p-4">
      <CircleSlash className="mt-0.5 size-4 shrink-0 text-slate-400" />
      <div className="min-w-0">
        <p className="font-medium text-white">{title}</p>
        <p className="mt-1 break-words text-sm text-slate-400">{detail}</p>
        {note ? <p className="mt-2 text-xs leading-5 text-slate-500">{note}</p> : null}
      </div>
    </div>
  );
}

function Notice({ tone, title, children }: { tone: "amber" | "slate"; title: string; children: React.ReactNode }) {
  const styles = tone === "amber" ? "border-amber-300/25 bg-amber-300/[0.06] text-amber-50" : "border-white/10 bg-white/[0.03] text-slate-200";
  return (
    <div className={`flex items-start gap-3 rounded-lg border p-4 text-sm ${styles}`}>
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      <div className="min-w-0">
        <p className="font-semibold">{title}</p>
        <div className="mt-1 text-slate-300">{children}</div>
      </div>
    </div>
  );
}

const roleTones = {
  emerald: "border-emerald-400/30 bg-emerald-400/12 text-emerald-200",
  cyan: "border-cyan-300/30 bg-cyan-300/12 text-cyan-100",
  violet: "border-violet-300/30 bg-violet-300/12 text-violet-100",
  slate: "border-slate-300/20 bg-slate-300/10 text-slate-300",
  red: "border-red-400/35 bg-red-400/12 text-red-100",
};

function RoleBadge({ tone, children }: { tone: keyof typeof roleTones; children: React.ReactNode }) {
  return <span className={`${roleTones[tone]} inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-[0.08em]`}>{children}</span>;
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">{children}</p>;
}

function NotPersisted() {
  return <span className="text-xs text-slate-500">no persistido</span>;
}

function checkIcon(state: CheckState): { Icon: LucideIcon; color: string } {
  if (state === "pass") return { Icon: CheckCircle2, color: "text-emerald-300" };
  if (state === "watch") return { Icon: AlertTriangle, color: "text-amber-200" };
  if (state === "fail") return { Icon: XCircle, color: "text-red-300" };
  return { Icon: HelpCircle, color: "text-slate-400" };
}

function checkLabel(label: string): string {
  const labels: Record<string, string> = {
    Registry: "Registro",
    "Latest cycle": "Último ciclo",
    "Official submission": "Submission oficial",
    Registered: "Registrado",
    "Shadow mode": "Modo shadow",
    "Champion unaffected": "Champion intacto",
    "Official submissions": "Submissions oficiales",
    "Shadow coverage": "Cobertura shadow",
    "Target alignment": "Alineación de targets",
    Freshness: "Freshness",
  };
  return labels[label] ?? label;
}

// Known parameters first; a champion with a different layout falls back to its top-level scalars.
function resolveParams(model: ModelVersion, specs: ParamSpec[]): Array<{ label: string; value: string }> {
  const known = specs
    .map((spec) => ({ label: spec.label, raw: firstValue(spec.source === "params" ? model.params : model.feature_config, spec.paths) }))
    .filter((param) => param.raw !== undefined && param.raw !== null)
    .map((param) => ({ label: param.label, value: paramValue(param.raw) }));
  if (known.length) return known;
  return Object.entries(model.params ?? {})
    .filter(([, value]) => value === null || typeof value !== "object" || Array.isArray(value))
    .slice(0, 6)
    .map(([key, value]) => ({ label: key.replaceAll("_", " "), value: paramValue(value) }));
}

function paramValue(value: unknown): string {
  if (Array.isArray(value)) return value.map((item) => text(item)).join(" – ");
  if (value !== null && typeof value === "object") return JSON.stringify(value);
  return text(value);
}

function at(source: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((value, key) => (value !== null && typeof value === "object" ? (value as Record<string, unknown>)[key] : undefined), source);
}

function firstValue(source: unknown, paths: string[]): unknown {
  for (const path of paths) {
    const value = at(source, path);
    if (value !== undefined && value !== null) return value;
  }
  return undefined;
}

function adaptiveLayer(model: ModelVersion): Record<string, unknown> {
  const nested = at(model.params, "adaptive_layer");
  if (nested !== null && typeof nested === "object") return nested as Record<string, unknown>;
  return model.params ?? {};
}

function fixed(value: number | null, digits: number): string {
  return value === null ? "n/a" : value.toFixed(digits);
}
