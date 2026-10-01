export function pct(value: unknown, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)}%` : "n/a";
}

export function numberValue(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("es-CO") : "n/a";
}

export function shortDate(value: unknown): string {
  if (typeof value !== "string" || !value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("es-CO", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function utcTimestamp(value: unknown): string {
  if (typeof value !== "string" || !value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return `${date.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

export function text(value: unknown, fallback = "n/a"): string {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}
