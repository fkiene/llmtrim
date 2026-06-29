// SVG sparkline drawn from a `trend: number[]` of per-bucket saved_pct values.
// Built with createElementNS (no innerHTML), so it stays CSP-safe and inert.

const NS = "http://www.w3.org/2000/svg";
const W = 96;
const H = 28;
const PAD = 2;

/** Returns an SVG sparkline, or a flat baseline placeholder when data is thin. */
export function sparkline(trend: number[], gradientId: string): SVGSVGElement {
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", String(W));
  svg.setAttribute("height", String(H));
  svg.setAttribute("class", "spark");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("preserveAspectRatio", "none");

  const min = Math.min(...trend);
  const max = Math.max(...trend);
  const span = max - min;

  // Thin (0/1 point) or perfectly flat data has no shape to plot — draw a
  // centered dashed baseline rather than a line pinned to an arbitrary edge.
  if (trend.length < 2 || span === 0) {
    const base = document.createElementNS(NS, "line");
    base.setAttribute("x1", String(PAD));
    base.setAttribute("y1", String(H / 2));
    base.setAttribute("x2", String(W - PAD));
    base.setAttribute("y2", String(H / 2));
    base.setAttribute("class", "spark-flat");
    svg.appendChild(base);
    return svg;
  }

  const stepX = (W - PAD * 2) / (trend.length - 1);

  const points = trend.map((v, i) => {
    const x = PAD + i * stepX;
    // Higher savings sit higher on the chart (smaller y).
    const y = PAD + (H - PAD * 2) * (1 - (v - min) / span);
    return [x, y] as const;
  });

  const line = points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");

  // Gradient fill under the curve.
  const defs = document.createElementNS(NS, "defs");
  const grad = document.createElementNS(NS, "linearGradient");
  grad.setAttribute("id", gradientId);
  grad.setAttribute("x1", "0");
  grad.setAttribute("y1", "0");
  grad.setAttribute("x2", "0");
  grad.setAttribute("y2", "1");
  for (const [offset, cls] of [
    ["0%", "spark-grad-top"],
    ["100%", "spark-grad-bottom"],
  ] as const) {
    const stop = document.createElementNS(NS, "stop");
    stop.setAttribute("offset", offset);
    stop.setAttribute("class", cls);
    grad.appendChild(stop);
  }
  defs.appendChild(grad);
  svg.appendChild(defs);

  const area = document.createElementNS(NS, "polygon");
  area.setAttribute(
    "points",
    `${PAD},${H - PAD} ${line} ${W - PAD},${H - PAD}`,
  );
  area.setAttribute("fill", `url(#${gradientId})`);
  svg.appendChild(area);

  const poly = document.createElementNS(NS, "polyline");
  poly.setAttribute("points", line);
  poly.setAttribute("class", "spark-line");
  svg.appendChild(poly);

  // Endpoint dot.
  const last = points[points.length - 1];
  const dot = document.createElementNS(NS, "circle");
  dot.setAttribute("cx", last[0].toFixed(2));
  dot.setAttribute("cy", last[1].toFixed(2));
  dot.setAttribute("r", "1.8");
  dot.setAttribute("class", "spark-dot");
  svg.appendChild(dot);

  return svg;
}
