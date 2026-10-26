export function DataUnavailable({ message }: { message: string }) {
  return (
    <section role="alert" className="rounded-lg border border-amber-300/25 bg-amber-300/[0.06] p-5 text-sm text-slate-300">
      <h2 className="text-lg font-semibold text-amber-100">Información no disponible</h2>
      <p className="mt-2">{message}</p>
      <p className="mt-2">No se muestran métricas ni modelos de demostración.</p>
    </section>
  );
}
