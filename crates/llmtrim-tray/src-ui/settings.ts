import { invoke } from "@tauri-apps/api/core";

import { el } from "./dom.js";

// Settings view. Built once and swapped in over the savings content (see
// `setSettingsOpen` in main.ts). All controls drive Tauri commands that already
// exist on the Rust side; every rejection arrives as a sanitised string.

export interface SettingsView {
  /** Root node, appended to #app and shown via the `settings-open` class. */
  root: HTMLElement;
  /** Re-read backend state. Called each time the view opens. */
  refresh(): Promise<void>;
}

/** Controls built in main.ts (so the dashboard can keep them in sync) but shown here. */
export interface SettingsExtras {
  /** Refresh-interval selector; its change handler lives in main.ts. */
  intervalSelect: HTMLSelectElement;
  /** Quit button; its click handler lives in main.ts. */
  quitBtn: HTMLElement;
  /** Hide-window control; main header is hidden while settings is open. */
  closeBtn: HTMLElement;
}

const SVG_NS = "http://www.w3.org/2000/svg";

function backIcon(): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", "16");
  svg.setAttribute("height", "16");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  for (const d of ["M15 18l-6-6 6-6"]) {
    const p = document.createElementNS(SVG_NS, "path");
    p.setAttribute("d", d);
    svg.appendChild(p);
  }
  return svg;
}

function errorMessage(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  return "Unexpected error.";
}

/** One labelled on/off switch wired to a get/set Tauri command pair. */
interface Toggle {
  group: HTMLElement;
  /** Re-read the backend state into the switch. */
  refresh(): Promise<void>;
}

function makeToggle(
  id: string,
  title: string,
  hint: string,
  getCmd: string,
  setCmd: string,
): Toggle {
  const input = el("input", {
    class: "switch-input",
    type: "checkbox",
    id,
    role: "switch",
  }) as HTMLInputElement;

  const knob = el("span", { class: "switch-track", "aria-hidden": "true" }, [
    el("span", { class: "switch-thumb" }),
  ]);

  const label = el("label", { class: "switch", for: id }, [
    el("span", { class: "row-text" }, [
      el("span", { class: "row-title" }, [title]),
      el("span", { class: "row-hint" }, [hint]),
    ]),
    input,
    knob,
  ]);

  const error = el("p", { class: "row-error", role: "alert" });
  error.hidden = true;

  input.addEventListener("change", () => {
    const enable = input.checked;
    input.disabled = true;
    error.hidden = true;
    void invoke(setCmd, { enable })
      .catch((e: unknown) => {
        input.checked = !enable; // revert
        error.textContent = errorMessage(e);
        error.hidden = false;
      })
      .finally(() => {
        input.disabled = false;
      });
  });

  const group = el("section", { class: "set-group" }, [label, error]);

  async function refresh(): Promise<void> {
    error.hidden = true;
    try {
      input.checked = await invoke<boolean>(getCmd);
    } catch (e) {
      error.textContent = errorMessage(e);
      error.hidden = false;
    }
  }

  return { group, refresh };
}

/**
 * Build the settings view.
 *
 * @param onClose Called when the user dismisses the view (Back / Escape).
 * @param extras Controls owned by main.ts but rendered here (interval, Quit).
 */
export function createSettingsView(
  onClose: () => void,
  extras: SettingsExtras,
): SettingsView {
  // --- header: back + title + hide (main header/X is display:none while open) ---
  const back = el(
    "button",
    { class: "set-back", type: "button", "aria-label": "Back to savings" },
    [backIcon(), el("span", {}, ["Back"])],
  );
  back.addEventListener("click", onClose);

  const head = el("header", { class: "set-head" }, [
    back,
    el("span", { class: "set-title" }, ["Settings"]),
    extras.closeBtn,
  ]);

  // --- autostart: the proxy and the tray are independent login items ---
  const proxyAutostart = makeToggle(
    "set-proxy-autostart",
    "Start proxy at login",
    "Run the llmtrim proxy when you sign in.",
    "get_proxy_autostart",
    "set_proxy_autostart",
  );
  const trayAutostart = makeToggle(
    "set-tray-autostart",
    "Open tray at login",
    "Show this tray app when you sign in.",
    "get_tray_autostart",
    "set_tray_autostart",
  );

  // --- proxy control: one button that flips Start/Stop to match live state ---
  const proxyStatus = el("p", { class: "row-status", "aria-live": "polite" });
  proxyStatus.hidden = true;

  // `running` is refreshed from `get_proxy_running` each time the view opens and
  // after every start/stop, so the button always offers the opposite action.
  let running = false;
  const proxyBtn = el("button", { class: "set-btn set-btn-accent", type: "button" }, [
    "Start proxy",
  ]) as HTMLButtonElement;

  function renderProxyBtn(): void {
    proxyBtn.textContent = running ? "Stop proxy" : "Start proxy";
    proxyBtn.classList.toggle("set-btn-accent", !running);
  }

  function flash(message: string, isError: boolean): void {
    proxyStatus.textContent = message;
    proxyStatus.classList.toggle("row-status-error", isError);
    proxyStatus.hidden = false;
  }

  // Lock the button while a start/stop is in flight so a double-click can't race.
  // `aria-busy` announces the pending state to assistive tech.
  async function toggleProxy(): Promise<void> {
    if (proxyBtn.disabled) return;
    const cmd = running ? "stop_proxy" : "start_proxy";
    const ok = running ? "Proxy stopped." : "Proxy started.";
    proxyBtn.disabled = true;
    proxyBtn.classList.add("is-busy");
    proxyBtn.setAttribute("aria-busy", "true");
    try {
      await invoke(cmd);
      running = !running;
      flash(ok, false);
    } catch (e) {
      flash(errorMessage(e), true);
    } finally {
      proxyBtn.classList.remove("is-busy");
      proxyBtn.removeAttribute("aria-busy");
      proxyBtn.disabled = false;
      renderProxyBtn();
    }
  }

  proxyBtn.addEventListener("click", () => void toggleProxy());

  async function refreshProxy(): Promise<void> {
    try {
      running = await invoke<boolean>("get_proxy_running");
    } catch {
      running = false; // safe default: offer Start
    }
    renderProxyBtn();
  }

  const proxyGroup = el("section", { class: "set-group" }, [
    el("span", { class: "row-title" }, ["Proxy"]),
    el("span", { class: "row-hint" }, [
      "Run the local llmtrim proxy on this machine.",
    ]),
    el("div", { class: "set-btn-row" }, [proxyBtn]),
    proxyStatus,
  ]);

  // --- general: refresh interval (selector built in main.ts) ---
  const intervalGroup = el("section", { class: "set-group" }, [
    el("div", { class: "set-row" }, [
      el("span", { class: "row-text" }, [
        el("span", { class: "row-title" }, ["Refresh interval"]),
        el("span", { class: "row-hint" }, ["How often the savings update."]),
      ]),
      extras.intervalSelect,
    ]),
  ]);

  // --- quit (button built in main.ts) ---
  const quitGroup = el("section", { class: "set-group set-group-quit" }, [
    extras.quitBtn,
  ]);

  const body = el("div", { class: "set-body" }, [
    proxyGroup,
    proxyAutostart.group,
    trayAutostart.group,
    intervalGroup,
    quitGroup,
  ]);

  const root = el("section", { class: "settings", "aria-label": "Settings" }, [
    head,
    body,
  ]);

  // Escape is handled document-wide in main.ts (settings → back, else hide).

  async function refresh(): Promise<void> {
    proxyStatus.hidden = true;
    await Promise.all([
      refreshProxy(),
      proxyAutostart.refresh(),
      trayAutostart.refresh(),
    ]);
  }

  return { root, refresh };
}
