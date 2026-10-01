import { demoSnapshot } from "@/lib/demo-data";
import { getSupabaseReader } from "@/lib/supabase";
import type { MonitoringSnapshot } from "@/lib/types";

export const revalidate = 60;

export async function getMonitoringSnapshot(): Promise<MonitoringSnapshot & { source: "supabase" | "demo" }> {
  const supabase = getSupabaseReader();
  if (!supabase) {
    return { ...demoSnapshot, source: "demo" };
  }

  const { data, error } = await supabase
    .select("monitoring_snapshots", "generated_at,model_id,overall_status,drift_status,retrain_recommended,report")
    .order("generated_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error || !data) {
    return { ...demoSnapshot, source: "demo" };
  }

  return { ...(data as MonitoringSnapshot), source: "supabase" };
}
