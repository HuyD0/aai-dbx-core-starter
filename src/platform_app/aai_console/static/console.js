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
