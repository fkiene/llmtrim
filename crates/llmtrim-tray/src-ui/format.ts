// Locale-aware formatting. Per CLAUDE.md §5 we never hardcode English-only number
// or date formats — `Intl` reads the system locale (undefined = runtime default).

const EM_DASH = "—";

/** Compact percentage, e.g. `42%`. `has_savings_data === false` renders an em dash. */
export function formatPct(pct: number, hasData: boolean): string {
  if (!hasData) return EM_DASH;
  return new Intl.NumberFormat(undefined, {
    style: "percent",
    maximumFractionDigits: 0,
  }).format(pct / 100);
}

/** Micro-USD (1e-6 USD) to a currency string. Sub-cent amounts keep more digits. */
export function formatBill(billMicros: number): string {
  const usd = billMicros / 1_000_000;
  const fractionDigits = usd > 0 && usd < 0.01 ? 4 : 2;
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: fractionDigits,
  }).format(usd);
}

/** Compact token count, e.g. `12.3K`. */
export function formatTokens(n: number): string {
  return new Intl.NumberFormat(undefined, {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(n);
}

const RELATIVE = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

const UNITS: Array<[Intl.RelativeTimeFormatUnit, number]> = [
  ["year", 365 * 24 * 3600],
  ["month", 30 * 24 * 3600],
  ["day", 24 * 3600],
  ["hour", 3600],
  ["minute", 60],
  ["second", 1],
];

/** RFC-3339 timestamp to a relative phrase like "5 minutes ago". `null` -> em dash. */
export function formatRelative(ts: string | null): string {
  if (!ts) return EM_DASH;
  const then = Date.parse(ts);
  if (Number.isNaN(then)) return EM_DASH;
  const diffSecs = Math.round((then - Date.now()) / 1000);
  const abs = Math.abs(diffSecs);
  if (abs < 5) return RELATIVE.format(0, "second");
  for (const [unit, secs] of UNITS) {
    if (abs >= secs) {
      return RELATIVE.format(Math.round(diffSecs / secs), unit);
    }
  }
  return RELATIVE.format(diffSecs, "second");
}
