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

// Serialization of the last state the server confirmed. `state` is the draft;
// a rejected render reverts the draft to this, so state, fragment, hash, and
// controls can never drift apart no matter which request lost the race.
let committed = "";

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

/** Revert the draft to the last server-confirmed state, controls included. */
function restoreCommitted() {
  const parsed = JSON.parse(committed);
  state.region = parsed.region;
  state.discount_dbu_pct = parsed.discount_dbu_pct;
  state.discount_vm_pct = parsed.discount_vm_pct;
  state.lines = parsed.lines;
  // The committed fragment on screen is authoritative again: its remove
  // buttons (disabled while a removal was in flight) become safe to use.
  for (const button of document.querySelectorAll("[data-remove-line]")) {
    button.disabled = false;
  }
  const region = document.getElementById("est-region");
  if ([...region.options].some((option) => option.value === state.region)) {
    region.value = state.region;
  }
  document.getElementById("est-discount-dbu").value = String(state.discount_dbu_pct);
  document.getElementById("est-discount-vm").value = String(state.discount_vm_pct);
  // The screen still shows the committed fragment (a rejected render never
  // swaps), so re-syncing the hash and toolbar realigns everything without a
  // recovery render.
  syncHash();
  syncToolbar();
}

/**
 * POST the whole draft state and swap the fragment.
 *
 * Outcomes: "ok" commits the exact payload that rendered; "rejected" (the
 * server refused the payload, or it never arrived) reverts the draft to the
 * last committed state; "aborted" means a newer render superseded this one and
 * will itself either commit or revert — so callers never mutate state in
 * response to an outcome.
 */
async function render() {
  if (!target) return "rejected";
  renderAbort?.abort();
  renderAbort = new AbortController();
  const signal = renderAbort.signal;
  const payload = JSON.stringify(state);
  target.setAttribute("aria-busy", "true");
  try {
    const response = await fetch("/api/estimator/render", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: payload,
      signal,
    });
    if (!response.ok) {
      const message = await problemMessage(response);
      // Superseded while reading the error body: the abort surfaces inside
      // problemMessage (whose catch returns the fallback), so re-check here —
      // only the newest render may show a problem or touch state.
      if (signal.aborted) return "aborted";
      showProblem(message);
      restoreCommitted();
      return "rejected";
    }
    const html = await response.text();
    if (signal.aborted) return "aborted";
    committed = payload;
    clearProblem();
    swap(target, html);
    syncHash();
    syncToolbar();
    return "ok";
  } catch (error) {
    if (error.name === "AbortError" || signal.aborted) return "aborted";
    showProblem("The console could not be reached. The estimate was not updated.");
    restoreCommitted();
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
  state.lines.push(collectLine(fieldset, label));
  button.disabled = true;
  try {
    if ((await render()) === "ok") labelInput.value = "";
    // "rejected" already reverted the draft; "aborted" hands off to the
    // superseding render, which commits or reverts the whole draft itself.
  } finally {
    button.disabled = false;
  }
}

async function downloadCsv() {
  try {
    // Export the committed state: it is exactly what the fragment shows.
    const response = await fetch("/api/estimator/export.csv", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: committed,
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
      if (remove.disabled) return;
      const index = Number(remove.dataset.removeLine);
      if (Number.isInteger(index) && index >= 0 && index < state.lines.length) {
        // Serialize removals: a second click before the render lands would
        // read this fragment's indexes against the already-shifted lines and
        // delete the wrong workload. A successful render swaps in fresh
        // buttons; a rejected one re-enables these via restoreCommitted().
        for (const button of document.querySelectorAll("[data-remove-line]")) {
          button.disabled = true;
        }
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
  // Baseline before adopting any hash: a shared link that decodes but cannot
  // be priced reverts to a clean page, not to the unpriceable state itself.
  committed = JSON.stringify(state);
  if (restoreFromHash() && state.lines.length) {
    render();
  } else {
    syncToolbar();
  }
}
