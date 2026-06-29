import type { AgentCard } from "./types.js";
import { el } from "./dom.js";
import { sparkline } from "./sparkline.js";
import { formatBill, formatPct, formatRelative, formatTokens } from "./format.js";

// A single agent row. The savings bar gradient is the one decorative gradient
// (a meter fill, not gradient text); width encodes saved_pct directly.
export function agentCard(card: AgentCard): HTMLElement {
  const hasData = card.has_savings_data;
  const pct = Math.max(0, Math.min(100, card.saved_pct));

  // --- header: name + last-active ---
  const name = el("span", { class: "card-name" }, [card.display_name]);
  const when = el("span", { class: "card-when" }, [formatRelative(card.last_event_ts)]);
  const head = el("div", { class: "card-head" }, [name, when]);

  // --- savings meter ---
  const fill = el("div", { class: "meter-fill" });
  // Drive the bar with a 0..1 scaleX (compositor-only) rather than width.
  fill.style.setProperty("--fill", String(pct / 100));
  if (!hasData) fill.classList.add("meter-empty");
  const meter = el(
    "div",
    {
      class: "meter",
      role: "progressbar",
      "aria-valuemin": "0",
      "aria-valuemax": "100",
      ...(hasData ? { "aria-valuenow": String(Math.round(pct)) } : {}),
      "aria-label": "input saved",
    },
    [fill],
  );

  const pctLabel = el(
    "span",
    { class: hasData ? "card-pct" : "card-pct card-pct-empty" },
    [formatPct(card.saved_pct, hasData)],
  );
  const savedTag = el("span", { class: "card-tag" }, ["saved"]);
  const pctBlock = el("div", { class: "card-pct-block" }, [pctLabel, savedTag]);

  const meterRow = el("div", { class: "card-meter-row" }, [meter, pctBlock]);

  // --- stats: bill, cache, sparkline ---
  const bill = stat(formatBill(card.bill_micros), "billed");
  const cache = stat(formatTokens(card.cache_read_tokens), "cache reads");
  // Gradient id keyed on the agent (slugified), not the array index, so it
  // stays unique and stable across re-renders that reorder the cards.
  const gradId = `spark-grad-${card.agent.replace(/[^a-z0-9]/gi, "-")}`;
  const spark = el("div", { class: "card-spark" }, [sparkline(card.trend, gradId)]);
  const stats = el("div", { class: "card-stats" }, [bill, cache, spark]);

  return el("article", { class: "card" }, [head, meterRow, stats]);
}

function stat(value: string, label: string): HTMLElement {
  return el("div", { class: "stat" }, [
    el("span", { class: "stat-value" }, [value]),
    el("span", { class: "stat-label" }, [label]),
  ]);
}
