import { createClient } from "@supabase/supabase-js";

// Server-side only: the service key never reaches the browser (no NEXT_PUBLIC_* key).
// The dashboard is read-only by construction: the reader exposes SELECT builders only,
// so insert / update / upsert / delete / rpc are not reachable from dashboard code.
export function getSupabaseReader() {
  const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY;
  if (!url || !serviceKey) return null;

  const client = createClient(url, serviceKey, {
    auth: { persistSession: false, autoRefreshToken: false },
    global: { headers: { "X-Client-Info": "pulso-transmi-dashboard" } },
  });

  return {
    select: (table: string, columns: string, options?: { count?: "exact"; head?: boolean }) => client.from(table).select(columns, options) as unknown as ReadQuery,
  };
}

export type SupabaseReader = NonNullable<ReturnType<typeof getSupabaseReader>>;

type QueryResponse = { data: unknown; error: { code?: string; message: string } | null; count?: number | null };

// Filter-only view of a PostgREST SELECT builder. Row types are cast by each caller: the
// schema is not generated, and the builder's inferred types are not worth their compile cost.
export type ReadQuery = PromiseLike<QueryResponse> & {
  eq(column: string, value: unknown): ReadQuery;
  in(column: string, values: readonly unknown[]): ReadQuery;
  not(column: string, operator: string, value: unknown): ReadQuery;
  order(column: string, options?: { ascending?: boolean }): ReadQuery;
  limit(count: number): ReadQuery;
  maybeSingle(): PromiseLike<QueryResponse>;
};

export type ReadIssue = {
  kind: "missing_table" | "missing_column" | "query_error";
  table: string;
  message: string;
};

export type ReadResult<T> = { ok: true; rows: T[]; count: number | null } | { ok: false; issue: ReadIssue };

export async function readRows<T>(table: string, query: PromiseLike<QueryResponse>): Promise<ReadResult<T>> {
  try {
    const { data, error, count } = await query;
    if (error) return { ok: false, issue: classifyReadError(table, error) };
    return { ok: true, rows: Array.isArray(data) ? (data as T[]) : [], count: count ?? null };
  } catch (exc) {
    return { ok: false, issue: { kind: "query_error", table, message: exc instanceof Error ? exc.message : String(exc) } };
  }
}

function classifyReadError(table: string, error: { code?: string; message: string }): ReadIssue {
  if (error.code === "PGRST205" || error.code === "42P01") {
    return { kind: "missing_table", table, message: `Table public.${table} does not exist in Supabase.` };
  }
  if (error.code === "42703" || error.code === "PGRST204") {
    return { kind: "missing_column", table, message: error.message };
  }
  return { kind: "query_error", table, message: error.message };
}
