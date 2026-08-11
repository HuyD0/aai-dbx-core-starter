/* AAI platform console — client behaviour.
 *
 * No framework and no build step. Fragments are parsed with <template>.content rather
 * than DOMParser: DOMParser runs full HTML tree construction and silently foster-parents
 * <tr>/<td>/<option> out of a fragment, and the platform-state list is table-shaped.
 */

/** Parse an HTML fragment without losing table rows. */
export function parseFragment(html) {
  const holder = document.createElement("template");
  holder.innerHTML = html;
  return holder.content;
}

/** Replace a container's children with a parsed fragment. */
export function swap(target, html) {
  target.replaceChildren(parseFragment(html));
}

/* ------------------------------------------------------------ clipboard */

export async function writeClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  // Fallback for non-secure contexts, where navigator.clipboard is undefined.
  const scratch = document.createElement("textarea");
  scratch.value = text;
  scratch.setAttribute("readonly", "");
  scratch.style.position = "fixed";
  scratch.style.opacity = "0";
  document.body.appendChild(scratch);
  scratch.select();
  try {
    if (!document.execCommand("copy")) throw new Error("copy rejected");
  } finally {
    scratch.remove();
  }
}

const copyTimers = new WeakMap();

function flashCopy(button, state, label) {
  const text = button.querySelector(".code__copytext");
  button.dataset.state = state;
  if (text) text.textContent = label;
  clearTimeout(copyTimers.get(button));
  copyTimers.set(
    button,
    setTimeout(() => {
      delete button.dataset.state;
      if (text) text.textContent = "Copy";
    }, 1600),
  );
}

document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-copy]");
  if (!button) return;
  const code = button.closest(".code")?.querySelector("code");
  if (!code) return;
  try {
    await writeClipboard(code.textContent);
    flashCopy(button, "done", "Copied");
  } catch {
    flashCopy(button, "error", "Press ⌘C");
  }
});

/* --------------------------------------------------------------- checks */

let checksAbort = null;

async function runChecks() {
  const target = document.getElementById("checks-target");
  if (!target) return;
  // Cancel any in-flight run so two clicks cannot land out of order.
  checksAbort?.abort();
  checksAbort = new AbortController();
  target.setAttribute("aria-busy", "true");
  try {
    const response = await fetch("/api/checks/run", {
      method: "POST",
      signal: checksAbort.signal,
    });
    if (!response.ok) throw new Error(String(response.status));
    swap(target, await response.text());
  } catch (error) {
    if (error.name === "AbortError") return;
    target.textContent = "Could not reach the console API. Retry in a moment.";
  } finally {
    target.removeAttribute("aria-busy");
  }
}

document.addEventListener("click", (event) => {
  if (event.target.closest("[data-run-checks], #checks-run")) runChecks();
});

/* ------------------------------------------------------------- generate */

let generateAbort = null;

document.addEventListener("click", async (event) => {
  const choice = event.target.closest("[data-template]");
  if (!choice) return;

  for (const other of choice.closest("[data-choices]").querySelectorAll("[data-template]")) {
    other.setAttribute("aria-pressed", String(other === choice));
  }

  const target = document.getElementById("generate-target");
  if (!target) return;

  generateAbort?.abort();
  generateAbort = new AbortController();
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ template: choice.dataset.template }),
      signal: generateAbort.signal,
    });
    if (!response.ok) throw new Error(String(response.status));
    swap(target, await response.text());
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    if (error.name === "AbortError") return;
    target.textContent = "Could not build the command. Retry in a moment.";
  }
});

/* -------------------------------------------------------------- palette */

const palette = document.getElementById("palette");
const paletteInput = document.getElementById("palette-input");
const paletteList = document.getElementById("palette-list");
let paletteAbort = null;

function openPalette(seed = "") {
  if (!palette || palette.open) return;
  paletteInput.value = seed;
  palette.showModal();
  paletteInput.focus();
  searchPalette();
}

async function searchPalette() {
  if (!paletteList) return;
  paletteAbort?.abort();
  paletteAbort = new AbortController();
  try {
    const response = await fetch(
      `/api/palette?q=${encodeURIComponent(paletteInput.value)}`,
      { signal: paletteAbort.signal },
    );
    const data = await response.json();
    paletteList.replaceChildren();
    for (const hit of data.results) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = `/track/${hit.track}#step-${hit.step}`;
      // textContent, never innerHTML: these strings come from content the server renders.
      const crumb = document.createElement("span");
      crumb.className = "palette__crumb";
      crumb.textContent = hit.track_title;
      const title = document.createElement("span");
      title.textContent = hit.title;
      link.append(crumb, title);
      item.append(link);
      paletteList.append(item);
    }
  } catch (error) {
    if (error.name !== "AbortError") paletteList.replaceChildren();
  }
}

paletteInput?.addEventListener("input", searchPalette);
document.getElementById("palette-open")?.addEventListener("click", () => openPalette());

document.addEventListener("keydown", (event) => {
  const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName ?? "");
  if (event.key === "/" && !typing) {
    event.preventDefault();
    openPalette();
  }
});

/* ------------------------------------------------------------- composer */

const composer = document.getElementById("composer-input");

function autosize() {
  composer.style.height = "auto";
  composer.style.height = `${composer.scrollHeight}px`;
}

if (composer) {
  composer.addEventListener("input", autosize);
  composer.addEventListener("keydown", (event) => {
    // Enter submits; Shift+Enter inserts a newline. isComposing guards IME input, where
    // Enter commits a candidate and must not be treated as submit.
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      const query = composer.value.trim();
      composer.value = "";
      autosize();
      openPalette(query);
    }
  });
  document.getElementById("composer-send")?.addEventListener("click", () => {
    const query = composer.value.trim();
    composer.value = "";
    autosize();
    openPalette(query);
  });
}

/* ----------------------------------------------------------- detail tabs */

const tabButtons = [...document.querySelectorAll("[role='tab'][data-tab]")];

function activateTab(selected, { updateUrl = true } = {}) {
  if (!selected) return;
  for (const button of tabButtons) {
    const active = button === selected;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    const panel = document.getElementById(button.getAttribute("aria-controls"));
    if (panel) panel.hidden = !active;
  }
  if (updateUrl) {
    const url = new URL(window.location.href);
    url.searchParams.set("tab", selected.dataset.tab);
    window.history.replaceState({}, "", url);
  }
}

for (const button of tabButtons) {
  button.addEventListener("click", () => activateTab(button));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    let index = tabButtons.indexOf(button);
    if (event.key === "Home") index = 0;
    if (event.key === "End") index = tabButtons.length - 1;
    if (event.key === "ArrowLeft") index = (index - 1 + tabButtons.length) % tabButtons.length;
    if (event.key === "ArrowRight") index = (index + 1) % tabButtons.length;
    tabButtons[index].focus();
    activateTab(tabButtons[index]);
  });
}

if (tabButtons.length) {
  const requestedTab = new URL(window.location.href).searchParams.get("tab");
  const initial = tabButtons.find((button) => button.dataset.tab === requestedTab);
  activateTab(initial || tabButtons[0], { updateUrl: false });
}

/* ------------------------------------------------------ governed actions */

function actionTarget() {
  return document.getElementById("action-result");
}

async function problemDetail(response) {
  const fallback = `The Hub returned HTTP ${response.status}.`;
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("json")) return fallback;
  try {
    const problem = await response.json();
    return problem.detail || problem.title || fallback;
  } catch {
    return fallback;
  }
}

async function submitAction(button, url, payload) {
  const target = actionTarget();
  if (!target) return;
  button.disabled = true;
  target.textContent = "Submitting…";
  target.dataset.tone = "pending";
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-requested-with": "AI-Platform-Hub",
      },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      target.textContent = await problemDetail(response);
      target.dataset.tone = "error";
      return;
    }
    const result = await response.json();
    target.textContent = result.message || `Accepted in state ${result.status}.`;
    target.dataset.tone = "success";
    // Keep the control disabled while a workflow is active. A page refresh obtains the
    // authoritative state; no optimistic success is shown before the transaction.
    if (!["REQUESTED", "QUEUED", "RUNNING", "PENDING_REVIEW"].includes(result.status)) {
      button.disabled = false;
    }
  } catch {
    target.textContent = "The Hub could not be reached. Nothing was submitted.";
    target.dataset.tone = "error";
    button.disabled = false;
  }
}

document.addEventListener("click", (event) => {
  const adminDecision = event.target.closest("[data-admin-decision]");
  if (adminDecision && !adminDecision.disabled) {
    const decision = adminDecision.dataset.adminDecision;
    const promotionId = adminDecision.dataset.promotionId;
    const applicationId = adminDecision.dataset.applicationId;
    const version = adminDecision.dataset.version;
    const source = adminDecision.dataset.source;
    const targetEnvironment = adminDecision.dataset.target;
    const rowVersion = Number(adminDecision.dataset.rowVersion);
    const comment = document
      .getElementById(adminDecision.dataset.commentInput)
      ?.value.trim();
    if (decision !== "approve" && !comment) {
      const target = actionTarget();
      if (target) {
        target.textContent =
          "A review comment is required to reject or request changes.";
        target.dataset.tone = "error";
      }
      return;
    }
    const confirmed = window.confirm(
      `${decision.replace("-", " ")} promotion request?\n\nApplication: ${applicationId}\nVersion: ${version}\nEnvironment: ${source} → ${targetEnvironment}\n\nThe decision is recorded as an append-only workflow event.`,
    );
    if (!confirmed) return;
    submitAction(
      adminDecision,
      `/api/v1/admin/promotion-requests/${encodeURIComponent(promotionId)}/${decision}`,
      {
        rowVersion,
        comment: comment || null,
      },
    );
    return;
  }

  const evaluation = event.target.closest("[data-run-evaluation]");
  if (evaluation && !evaluation.disabled) {
    const applicationId = evaluation.dataset.runEvaluation;
    const environment = evaluation.dataset.environment;
    const datasetVersion = document
      .getElementById("evaluation-dataset-version")
      ?.value.trim();
    if (!datasetVersion) {
      const target = actionTarget();
      if (target) {
        target.textContent =
          "Enter an immutable governed dataset version before running evaluation.";
        target.dataset.tone = "error";
      }
      return;
    }
    submitAction(
      evaluation,
      `/api/v1/applications/${encodeURIComponent(applicationId)}/evaluations`,
      {
        environment,
        datasetVersion,
      },
    );
    return;
  }

  const promotion = event.target.closest("[data-request-promotion]");
  if (!promotion || promotion.disabled) return;
  const applicationId = promotion.dataset.requestPromotion;
  const source = promotion.dataset.source;
  const target = promotion.dataset.target;
  const version = promotion.dataset.version;
  const confirmed = window.confirm(
    `Request UAT promotion?\n\nApplication: ${applicationId}\nVersion: ${version}\nEnvironment: ${source} → ${target}\n\nSubmission does not deploy. An administrator must review current readiness evidence.`,
  );
  if (!confirmed) return;
  submitAction(
    promotion,
    `/api/v1/applications/${encodeURIComponent(applicationId)}/promotion-requests`,
    {
      sourceEnvironment: source,
      targetEnvironment: target,
    },
  );
});
