/* AAI platform console — client behaviour.
 *
 * No framework and no build step. Fragments are parsed with <template>.content rather
 * than DOMParser: DOMParser runs full HTML tree construction and silently foster-parents
 * <tr>/<td>/<option> out of a fragment, and the platform-state list is table-shaped.
 */

/** Parse an HTML fragment without losing table rows. */
function parseFragment(html) {
  const holder = document.createElement("template");
  holder.innerHTML = html;
  return holder.content;
}

/** Replace a container's children with a parsed fragment. */
function swap(target, html) {
  target.replaceChildren(parseFragment(html));
}

/* ------------------------------------------------------------ clipboard */

async function writeClipboard(text) {
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
let selectedTemplate = null;

async function generateCommand(template, scroll = true) {
  const target = document.getElementById("generate-target");
  if (!target) return;
  const projectName = document.getElementById("project-name");
  if (projectName && !projectName.reportValidity()) return;

  generateAbort?.abort();
  generateAbort = new AbortController();
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        template,
        project_name: projectName?.value || "my-project",
      }),
      signal: generateAbort.signal,
    });
    if (!response.ok) throw new Error(String(response.status));
    swap(target, await response.text());
    if (scroll) target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    if (error.name === "AbortError") return;
    target.textContent = "Could not build the command. Retry in a moment.";
  }
}

document.addEventListener("click", (event) => {
  const choice = event.target.closest("[data-template]");
  if (!choice) return;
  selectedTemplate = choice.dataset.template;
  for (const other of choice.closest("[data-choices]").querySelectorAll("[data-template]")) {
    other.setAttribute("aria-pressed", String(other === choice));
  }
  generateCommand(selectedTemplate);
});

document.getElementById("project-name")?.addEventListener("change", () => {
  if (selectedTemplate) generateCommand(selectedTemplate, false);
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

/* --------------------------------------------------------- guide search */

const guideSearch = document.getElementById("guide-search-input");

function submitGuideSearch() {
  openPalette(guideSearch?.value.trim() || "");
}

guideSearch?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.isComposing) {
    event.preventDefault();
    submitGuideSearch();
  }
});
document.getElementById("guide-search-submit")?.addEventListener("click", submitGuideSearch);

/* ------------------------------------------------------ local progress */

const progressKey = "aai-console-completed-steps-v1";

function readProgress() {
  try {
    return new Set(JSON.parse(localStorage.getItem(progressKey) || "[]"));
  } catch {
    return new Set();
  }
}

function writeProgress(completed) {
  try {
    localStorage.setItem(progressKey, JSON.stringify([...completed]));
  } catch {
    // Progress is optional; private browsing and storage policy must not block guidance.
  }
}

function renderProgress(completed) {
  const steps = [...document.querySelectorAll("[data-step]")];
  for (const step of steps) {
    const done = completed.has(step.dataset.step);
    step.classList.toggle("step--complete", done);
    const button = step.querySelector("[data-step-complete]");
    button?.setAttribute("aria-pressed", String(done));
    if (button) button.textContent = done ? "Completed" : "Mark complete";
  }
  const label = document.querySelector("[data-track-progress]");
  if (label) label.textContent = `${steps.filter((step) => completed.has(step.dataset.step)).length} of ${steps.length} complete`;
}

const completedSteps = readProgress();
renderProgress(completedSteps);
document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-step-complete]");
  if (!button) return;
  const id = button.closest("[data-step]")?.dataset.step;
  if (!id) return;
  if (completedSteps.has(id)) completedSteps.delete(id);
  else completedSteps.add(id);
  writeProgress(completedSteps);
  renderProgress(completedSteps);
});
