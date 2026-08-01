/* Cost estimator page behaviour.
 *
 * The estimate lives entirely in this module and in the URL hash — the server
 * is stateless and does all of the arithmetic. Every mutation POSTs the whole
 * state to /api/estimator/render and swaps the returned fragment, so the math
 * has exactly one home. Nothing here writes markup from data: fragments come
 * server-rendered, everything else is textContent.
 */

import { swap, writeClipboard } from "/static/console.js";

const target = document.getElementById("estimate-target");
const kindSelect = document.getElementById("est-kind");

const state = {
  region: "",
  discount_dbu_pct: 0,
  discount_vm_pct: 0,
  lines: [],
};

let renderAbort = null;

function problemHost() {
  return document.getElementById("estimator-problem");
}

function showProblem(message) {
  const host = problemHost();
  if (!host) return;
  host.querySelector("#estimator-problem-detail").textContent = message;
  host.hidden = false;
}

function clearProblem() {
  const host = problemHost();
  if (host) host.hidden = true;
}

async function problemMessage(response) {
  const fallback = `The console returned HTTP ${response.status}.`;
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("json")) return fallback;
  try {
    const problem = await response.json();
    const first = problem.errors?.[0];
    if (first?.path && first?.message) return `${first.path}: ${first.message}`;
    return problem.detail || problem.title || fallback;
  } catch {
    return fallback;
  }
}

/* -------------------------------------------------------------- payload */

function fieldValue(input) {
  if (input.type === "checkbox") return input.checked;
  const raw = input.value.trim();
  if (raw === "") return null;
  if (input.type === "number") return Number(raw);
  return raw;
}

function collectUsage(fieldset) {
  const block = fieldset.querySelector(".usage-block");
  if (!block) return null;
  const mode = block.querySelector("input[type=radio]:checked")?.value;
  const active = block.querySelector(`.usage-fields[data-usage-mode="${mode}"]`);
  if (!active) return null;
  const usage = {};
  for (const input of active.querySelectorAll("[data-usage]")) {
    const raw = input.value.trim();
    if (raw !== "") usage[input.dataset.usage] = Number(raw);
  }
  return Object.keys(usage).length ? usage : null;
}

function collectLine(fieldset, label) {
  const line = { kind: fieldset.dataset.kind, label };
  for (const input of fieldset.querySelectorAll("[data-field]")) {
    const value = fieldValue(input);
    if (value !== null) line[input.dataset.field] = value;
  }
  const usage = collectUsage(fieldset);
  if (usage) line.usage = usage;
  return line;
}

function readSettings() {
  state.region = document.getElementById("est-region").value;
  state.discount_dbu_pct =
    Number(document.getElementById("est-discount-dbu").value) || 0;
  state.discount_vm_pct =
    Number(document.getElementById("est-discount-vm").value) || 0;
}

/* ------------------------------------------------------------ share hash */

function b64urlEncode(text) {
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

function b64urlDecode(encoded) {
  const padded = encoded.replaceAll("-", "+").replaceAll("_", "/");
  const binary = atob(padded + "=".repeat((4 - (padded.length % 4)) % 4));
  return new TextDecoder().decode(Uint8Array.from(binary, (c) => c.charCodeAt(0)));
}

function syncHash() {
  if (!state.lines.length) {
    history.replaceState(null, "", window.location.pathname);
    return;
  }
  const encoded = b64urlEncode(JSON.stringify({ v: 1, ...state }));
  history.replaceState(null, "", `${window.location.pathname}#e=${encoded}`);
}

function restoreFromHash() {
  const match = window.location.hash.match(/^#e=([A-Za-z0-9_-]+)$/);
  if (!match) return false;
  try {
    const decoded = JSON.parse(b64urlDecode(match[1]));
    if (decoded?.v !== 1 || !Array.isArray(decoded.lines)) throw new Error("shape");
    state.region = String(decoded.region || state.region);
    state.discount_dbu_pct = Number(decoded.discount_dbu_pct) || 0;
    state.discount_vm_pct = Number(decoded.discount_vm_pct) || 0;
    state.lines = decoded.lines;
  } catch {
    history.replaceState(null, "", window.location.pathname);
    showProblem("The shared estimate link could not be read, so it was ignored.");
    return false;
  }
  const region = document.getElementById("est-region");
  if ([...region.options].some((option) => option.value === state.region)) {
    region.value = state.region;
  }
  document.getElementById("est-discount-dbu").value = String(state.discount_dbu_pct);
  document.getElementById("est-discount-vm").value = String(state.discount_vm_pct);
  return true;
}

/* --------------------------------------------------------------- render */

function syncToolbar() {
  for (const selector of ["[data-export-csv]", "[data-copy-link]"]) {
    const button = document.querySelector(selector);
    if (button) button.disabled = !state.lines.length;
  }
}

/**
 * POST the whole state and swap the fragment.
 *
 * "aborted" means a newer render superseded this one; the newer render posts
 * the full current state, so the caller must NOT mutate state in response —
 * only "rejected" reports on the payload itself.
 */
async function render() {
  if (!target) return "rejected";
  renderAbort?.abort();
  renderAbort = new AbortController();
  const signal = renderAbort.signal;
  target.setAttribute("aria-busy", "true");
  try {
    const response = await fetch("/api/estimator/render", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(state),
      signal,
    });
    if (!response.ok) {
      showProblem(await problemMessage(response));
      return "rejected";
    }
    const html = await response.text();
    if (signal.aborted) return "aborted";
    clearProblem();
    swap(target, html);
    syncHash();
    syncToolbar();
    return "ok";
  } catch (error) {
    if (error.name === "AbortError") return "aborted";
    showProblem("The console could not be reached. The estimate was not updated.");
    return "rejected";
  } finally {
    target.removeAttribute("aria-busy");
  }
}

async function addLine(button) {
  const labelInput = document.getElementById("est-label");
  const label = labelInput.value.trim();
  if (!label) {
    showProblem("Give the line item a label first.");
    labelInput.focus();
    return;
  }
  const fieldset = document.querySelector(
    `.est-kind[data-kind="${kindSelect.value}"]`,
  );
  if (!fieldset) return;
  readSettings();
  const line = collectLine(fieldset, label);
  state.lines.push(line);
  button.disabled = true;
  try {
    const outcome = await render();
    if (outcome === "ok") {
      labelInput.value = "";
    } else if (outcome === "rejected") {
      // Roll back this specific line, never whatever happens to be last: a
      // concurrent action may have appended since. The screen still shows the
      // last successful fragment, which equals the rolled-back state, so no
      // recovery render is needed (and none can cascade-abort other work).
      const index = state.lines.indexOf(line);
      if (index !== -1) state.lines.splice(index, 1);
    }
    // "aborted": the superseding render owns the final paint; nothing to do.
  } finally {
    button.disabled = false;
  }
}

async function downloadCsv() {
  readSettings();
  try {
    const response = await fetch("/api/estimator/export.csv", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(state),
    });
    if (!response.ok) {
      showProblem(await problemMessage(response));
      return;
    }
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "databricks-cost-estimate.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  } catch {
    showProblem("The console could not be reached. No file was downloaded.");
  }
}

/* -------------------------------------------------------------- wiring */

if (target && kindSelect) {
  document.addEventListener("click", (event) => {
    const add = event.target.closest("[data-add-line]");
    if (add) {
      if (!add.disabled) addLine(add);
      return;
    }
    const remove = event.target.closest("[data-remove-line]");
    if (remove) {
      const index = Number(remove.dataset.removeLine);
      if (Number.isInteger(index) && index >= 0 && index < state.lines.length) {
        state.lines.splice(index, 1);
        readSettings();
        render();
      }
      return;
    }
    if (event.target.closest("[data-export-csv]")) {
      downloadCsv();
      return;
    }
    const share = event.target.closest("[data-copy-link]");
    if (share) {
      syncHash();
      writeClipboard(window.location.href).then(
        () => {
          share.textContent = "Link copied";
          setTimeout(() => {
            share.textContent = "Copy share link";
          }, 1600);
        },
        () => showProblem("The link could not be copied — copy the address bar."),
      );
    }
  });

  kindSelect.addEventListener("change", () => {
    for (const fieldset of document.querySelectorAll(".est-kind")) {
      fieldset.hidden = fieldset.dataset.kind !== kindSelect.value;
    }
  });

  document.addEventListener("change", (event) => {
    const radio = event.target.closest(".usage-block input[type=radio]");
    if (radio) {
      const block = radio.closest(".usage-block");
      for (const fields of block.querySelectorAll(".usage-fields")) {
        fields.hidden = fields.dataset.usageMode !== radio.value;
      }
      return;
    }
    if (
      event.target.closest("#est-region, #est-discount-dbu, #est-discount-vm")
    ) {
      readSettings();
      if (state.lines.length) render();
    }
  });

  readSettings();
  if (restoreFromHash() && state.lines.length) {
    render();
  } else {
    syncToolbar();
  }
}
