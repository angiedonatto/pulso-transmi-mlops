import "server-only";
import { cache } from "react";
import { getSupabaseReader } from "@/lib/supabase";
import { parseSnapshot } from "@/lib/snapshot";
import type { SnapshotResult } from "@/lib/types";

// One snapshot per server request; layout and page share the same historical state.
export const getMonitoringSnapshot = cache(async (): Promise<SnapshotResult> => {
  try {
    const db = getSupabaseReader();
    if (!db) return { source: "unavailable", snapshot: null, reason: "not_configured", message: "Supabase no está configurado o su configuración es inválida." };
    const { data, error } = await db
      .select("monitoring_snapshots", "generated_at,model_id,overall_status,drift_status,retrain_recommended,report")
      .order("generated_at", { ascending: false }).limit(1).maybeSingle();
    if (error) return { source: "unavailable", snapshot: null, reason: "query_error", message: "No se pudo leer monitoring_snapshots. Revisa conexión, credenciales y permisos del servidor." };
    if (!data) return { source: "unavailable", snapshot: null, reason: "empty", message: "No hay snapshots de monitoreo disponibles." };
    const snapshot = parseSnapshot(data);
    if (!snapshot) return { source: "unavailable", snapshot: null, reason: "invalid_snapshot", message: "El último snapshot tiene un formato incompatible. No se pueden mostrar sus métricas con seguridad." };
    return { source: "supabase", snapshot };
  } catch {
    return { source: "unavailable", snapshot: null, reason: "query_error", message: "No se pudo conectar con Supabase. No se muestran datos de muestra." };
  }
});
