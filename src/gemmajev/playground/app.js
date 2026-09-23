"use strict";

// The visible form is the only editable representation of a request.
const $ = (id) => document.getElementById(id);
const fields = ["task", "state", "query"];
let examples = [];
let selectedExample = null;
let currentModel = "12b";
let busy = true;
let lastResult = null;

function setStatus(message, kind = "ready") {
  const status = $("run-status");
  status.textContent = message;
  status.title = message;
  status.dataset.kind = kind;
}

function syncControls() {
  document.querySelectorAll("button, input, textarea, select").forEach((control) => {
    control.disabled = busy;
  });
  const count = $("option-rows").children.length;
  $("add-option").disabled = busy || count >= 26;
  document.querySelectorAll(".remove-option").forEach((button) => {
    button.disabled = busy || count <= 2;
  });
  $("restore-example").disabled = busy || selectedExample === null;
  $("input-panel").setAttribute("aria-busy", String(busy));
}

function setBusy(value) {
  busy = value;
  syncControls();
}

function getRequest() {
  const request = Object.fromEntries(fields.map((name) => [name, $(name).value]));
  for (const name of fields) {
    if (!request[name].trim()) throw new Error(`${name[0].toUpperCase() + name.slice(1)} cannot be empty.`);
  }
  const rows = [...$("option-rows").children];
  if (rows.length < 2 || rows.length > 26) throw new Error("Provide between 2 and 26 options.");
  // Null-prototype storage keeps IDs such as "__proto__" ordinary option IDs.
  request.options = Object.create(null);
  for (const row of rows) {
    const id = row.querySelector(".option-id").value;
    const description = row.querySelector(".option-description").value;
    if (!id.trim() || !description.trim()) throw new Error("Each option needs an ID and a description.");
    const numericId = Number(id);
    if (Number.isInteger(numericId) && numericId >= 0 && numericId < 2 ** 32 - 1 && String(numericId) === id) {
      throw new Error("Use a meaningful option ID such as score_2 instead of a number, to preserve option order.");
    }
    if (Object.hasOwn(request.options, id)) throw new Error(`Duplicate option ID: ${id}`);
    request.options[id] = description;
  }
  return request;
}

function edited() {
  if (busy) return;
  if (!lastResult) {
    setStatus("Ready to run.");
    return;
  }
  let matches = false;
  try {
    matches = lastResult.model === currentModel && JSON.stringify(getRequest()) === JSON.stringify(lastResult.request);
  } catch {
    // An incomplete edit still makes the old result stale; validate on Run.
  }
  setStatus(matches ? "Ready · result matches this input." : "Input changed · run again to update.");
}

function addOption(id = "", description = "", focus = false) {
  const row = document.createElement("tr");
  const idCell = document.createElement("td");
  const idEditor = document.createElement("div");
  idEditor.className = "option-id-editor";
  const descriptionCell = document.createElement("td");
  const idInput = document.createElement("input");
  idInput.type = "text";
  idInput.className = "option-id";
  idInput.value = id;
  idInput.spellcheck = false;
  idInput.autocomplete = "off";
  const descriptionInput = document.createElement("textarea");
  descriptionInput.className = "option-description";
  descriptionInput.value = description;
  descriptionInput.spellcheck = false;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "remove-option";
  remove.textContent = "×";
  remove.addEventListener("click", () => {
    if (busy || $("option-rows").children.length <= 2) return;
    const next = row.nextElementSibling || row.previousElementSibling;
    row.remove();
    labelOptions();
    syncControls();
    edited();
    next?.querySelector(".option-id").focus();
  });
  idEditor.append(idInput, remove);
  idCell.append(idEditor);
  descriptionCell.append(descriptionInput);
  row.append(idCell, descriptionCell);
  $("option-rows").append(row);
  labelOptions();
  syncControls();
  if (focus) {
    idInput.focus();
    row.scrollIntoView({block: "nearest"});
  }
}

function labelOptions() {
  [...$("option-rows").children].forEach((row, index) => {
    row.querySelector(".option-id").setAttribute("aria-label", `Option ${index + 1} ID`);
    row.querySelector(".option-description").setAttribute("aria-label", `Option ${index + 1} description`);
    row.querySelector(".remove-option").setAttribute("aria-label", `Remove option ${index + 1}`);
    row.querySelector(".remove-option").title = `Remove option ${index + 1}`;
  });
}

function selectExample(example) {
  selectedExample = example;
  for (const name of fields) {
    $(name).value = example.request[name];
    $(name).scrollTop = 0;
  }
  $("option-rows").replaceChildren();
  for (const [id, description] of Object.entries(example.request.options)) addOption(id, description);
  $("options-scroll").scrollTop = 0;
  lastResult = null;
  renderResult(null);
  document.querySelectorAll(".example-button").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.example === example.id));
  });
  syncControls();
  setStatus("Ready to run.");
}

function sameOptions(first, second) {
  const entries = Object.entries(first);
  return entries.length === Object.keys(second).length && entries.every(([id, description]) => Object.hasOwn(second, id) && second[id] === description);
}

function legendItem(label, className) {
  const item = document.createElement("span");
  const swatch = document.createElement("i");
  swatch.className = className;
  swatch.setAttribute("aria-hidden", "true");
  item.append(swatch, document.createTextNode(label));
  return item;
}

function renderResult(result, previous = null) {
  const winner = $("result-winner");
  const list = $("probability-list");
  const legend = $("result-legend");
  list.replaceChildren();
  list.scrollTop = 0;
  legend.replaceChildren();
  winner.classList.toggle("empty-winner", result === null);
  if (!result) {
    winner.textContent = "No result yet";
    winner.removeAttribute("title");
    $("result-meta").textContent = "Click Run decision to see the probabilities.";
    return;
  }
  const probabilities = Object.entries(result.probabilities);
  const selected = probabilities.reduce((best, next) => next[1] > best[1] ? next : best);
  winner.textContent = selected[0];
  winner.title = selected[0];
  $("result-meta").textContent = `Gemma 4 ${result.model.toUpperCase()} · ${(result.request_ms / 1000).toFixed(2)} s`;
  const comparable = previous && sameOptions(result.request.options, previous.request.options);
  for (const [id, probability] of probabilities) {
    const row = document.createElement("div");
    row.className = "probability-row";
    const label = document.createElement("div");
    label.className = "probability-label";
    const name = document.createElement("span");
    name.textContent = id;
    name.title = id;
    const value = document.createElement("span");
    value.className = "probability-value";
    value.textContent = `${(100 * probability).toFixed(1)}%`;
    label.append(name, value);
    const track = document.createElement("div");
    track.className = "probability-track";
    track.setAttribute("aria-hidden", "true");
    if (comparable) {
      const old = document.createElement("div");
      old.className = "previous-bar";
      old.style.width = `${100 * previous.probabilities[id]}%`;
      track.append(old);
    }
    const current = document.createElement("div");
    current.className = "current-bar";
    current.style.width = `${100 * probability}%`;
    track.append(current);
    row.append(label, track);
    list.append(row);
  }
  legend.append(legendItem("Current", "legend-current"));
  if (comparable) legend.append(legendItem("Previous", "legend-previous"));
}

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {cache: "no-store"} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request could not be completed.");
  return data;
}

$("input-panel").addEventListener("input", edited);
$("input-panel").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  let request;
  try {
    request = getRequest();
  } catch (error) {
    setStatus(error.message + (lastResult ? " Previous result shown." : ""), "error");
    return;
  }
  setBusy(true);
  setStatus("Running…", "running");
  try {
    const result = await api("/api/decide", {model: currentModel, request});
    renderResult(result, lastResult);
    lastResult = result;
    setStatus("Ready · result matches this input.");
  } catch (error) {
    setStatus(error.message + (lastResult ? " Previous result shown." : ""), "error");
  } finally {
    setBusy(false);
  }
});

$("add-option").addEventListener("click", () => {
  if (busy || $("option-rows").children.length >= 26) return;
  addOption("", "", true);
  edited();
});
$("restore-example").addEventListener("click", () => {
  if (!busy && selectedExample) selectExample(selectedExample);
});
$("model-select").addEventListener("change", async () => {
  if (busy) return;
  const model = $("model-select").value;
  lastResult = null;
  renderResult(null);
  setBusy(true);
  setStatus(`Loading Gemma 4 ${model.toUpperCase()}…`, "running");
  try {
    await api("/api/model", {model});
    currentModel = model;
    setStatus("Ready to run.");
  } catch (error) {
    $("model-select").value = currentModel;
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

async function initialize() {
  try {
    const config = await api("/api/config");
    currentModel = config.model;
    examples = config.examples;
    $("model-select").value = currentModel;
    for (const example of examples) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example-button";
      button.dataset.example = example.id;
      button.textContent = example.title;
      button.addEventListener("click", () => { if (!busy) selectExample(example); });
      $("example-buttons").append(button);
    }
    selectExample(examples[0]);
    setBusy(false);
  } catch (error) {
    setStatus(`Could not connect: ${error.message} Refresh to retry.`, "error");
  }
}

initialize();
