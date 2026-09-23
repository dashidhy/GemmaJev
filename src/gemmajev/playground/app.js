"use strict";

// The visible form is the only editable representation of a request.
const $ = (id) => document.getElementById(id);
const fields = ["task", "state", "query"];
let examples = [];
let selectedExample = null;
let currentModel = "12b";
let busy = true;
let lastResult = null;
let imageSamples = [];
let currentImage = null;
let imageRevision = 0;

function exampleHasImage(example = selectedExample) {
  return Boolean(example?.image_id);
}

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
  const imageEnabled = exampleHasImage();
  $("image-panel").querySelectorAll("button, input").forEach((control) => {
    control.disabled = busy || !imageEnabled;
  });
  $("clear-image").disabled = busy || !imageEnabled || currentImage === null;
  $("image-panel").setAttribute("aria-disabled", String(!imageEnabled));
  $("input-panel").setAttribute("aria-busy", String(busy));
  $("image-panel").setAttribute("aria-busy", String(busy));
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
    matches = lastResult.model === currentModel && (lastResult.image_id || null) === (currentImage?.id || null) && JSON.stringify(getRequest()) === JSON.stringify(lastResult.request);
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

function clearResult() {
  lastResult = null;
  renderResult(null);
}

function clearImage(message = exampleHasImage()
  ? "No image attached"
  : "This example uses text only. Select Image style to add an image.") {
  imageRevision += 1;
  currentImage = null;
  $("image-preview").removeAttribute("src");
  $("image-preview").hidden = true;
  $("image-placeholder").textContent = message;
  $("image-placeholder").hidden = false;
  document.querySelectorAll(".image-sample").forEach((button) => button.setAttribute("aria-pressed", "false"));
  return imageRevision;
}

function localImageURL(value) {
  const url = new URL(value, window.location.origin);
  if (url.origin !== window.location.origin || !url.pathname.startsWith("/api/images/")) {
    throw new Error("The image must come from this Playground.");
  }
  return url.href;
}

async function displayImage(selection, revision) {
  const preview = new Image();
  preview.id = "image-preview";
  preview.alt = selection.title || "Uploaded image";
  preview.src = localImageURL(selection.url);
  try {
    await preview.decode();
  } catch {
    throw new Error("Could not load the image. Check that the Playground server is running, then try again.");
  }
  if (revision !== imageRevision) return;
  $("image-preview").replaceWith(preview);
  $("image-placeholder").hidden = true;
  currentImage = {id: selection.id || selection.image_id, url: selection.url};
  document.querySelectorAll(".image-sample").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.image === currentImage.id));
  });
}

async function chooseImage(selection) {
  if (!exampleHasImage()) return;
  clearResult();
  const revision = clearImage("Loading image…");
  setBusy(true);
  setStatus("Loading image…", "running");
  try {
    await displayImage(selection, revision);
    if (revision === imageRevision) setStatus("Ready to run.");
  } catch (error) {
    if (revision === imageRevision) {
      $("image-placeholder").textContent = error.message;
      setStatus(error.message, "error");
    }
  } finally {
    if (revision === imageRevision) setBusy(false);
  }
}

async function selectExample(example) {
  selectedExample = example;
  for (const name of fields) {
    $(name).value = example.request[name];
    $(name).scrollTop = 0;
  }
  $("option-rows").replaceChildren();
  for (const [id, description] of Object.entries(example.request.options)) addOption(id, description);
  $("options-scroll").scrollTop = 0;
  clearResult();
  clearImage();
  document.querySelectorAll(".example-button").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.example === example.id));
  });
  if (example.image_id) {
    const sample = imageSamples.find((item) => item.id === example.image_id);
    if (sample) {
      await chooseImage(sample);
      return;
    }
    setStatus("The example image is unavailable. Upload an image to continue.", "error");
  } else {
    setStatus("Ready to run.");
  }
  syncControls();
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
  const comparable = previous && (result.image_id || null) === (previous.image_id || null) && sameOptions(result.request.options, previous.request.options);
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
  return readResponse(response);
}

async function readResponse(response) {
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
    if (exampleHasImage() && !currentImage) {
      throw new Error("Choose a sample or upload an image before running Image style.");
    }
    request = getRequest();
  } catch (error) {
    setStatus(error.message + (lastResult ? " Previous result shown." : ""), "error");
    return;
  }
  setBusy(true);
  setStatus("Running…", "running");
  try {
    const result = await api("/api/decide", {model: currentModel, request, ...(currentImage ? {image_id: currentImage.id} : {})});
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
  if (!busy && selectedExample) void selectExample(selectedExample);
});

async function imageDimensions(file) {
  // Decode only bounded still images. A small compressed file can otherwise
  // allocate a very large bitmap before the canvas resize happens.
  const bytes = new Uint8Array(await file.arrayBuffer());
  const view = new DataView(bytes.buffer);
  let width;
  let height;
  const pngSignature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (bytes.length >= 33 && pngSignature.every((value, index) => bytes[index] === value)) {
    if (view.getUint32(8) !== 13 || String.fromCharCode(...bytes.slice(12, 16)) !== "IHDR") {
      throw new Error("This file does not contain a valid PNG image header.");
    }
    width = view.getUint32(16);
    height = view.getUint32(20);
  } else if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    let offset = 2;
    const frameMarkers = [0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf];
    while (offset < bytes.length) {
      if (bytes[offset++] !== 0xff) break;
      while (offset < bytes.length && bytes[offset] === 0xff) offset += 1;
      if (offset >= bytes.length) break;
      const marker = bytes[offset++];
      if (marker === 0xda || marker === 0xd9 || marker === 0x00) break;
      if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
      if (offset + 2 > bytes.length) break;
      const length = view.getUint16(offset);
      if (length < 2 || offset + length > bytes.length) break;
      if (frameMarkers.includes(marker)) {
        if (length < 8) break;
        height = view.getUint16(offset + 3);
        width = view.getUint16(offset + 5);
        break;
      }
      offset += length;
    }
  }
  if (!width || !height) {
    throw new Error("This file does not contain a readable PNG or JPEG image header.");
  }
  // A marketed 24 MP phone photo can contain about 24.5 million pixels.
  if (width * height > 25_000_000) {
    throw new Error("This image is too large to decode safely. Choose an image up to 25 megapixels.");
  }
  return {width, height};
}

async function normalizedPNG(file) {
  await imageDimensions(file);
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, {imageOrientation: "from-image"});
  } catch {
    throw new Error("This file could not be decoded. Choose a valid PNG or JPEG image.");
  }
  const canvas = document.createElement("canvas");
  try {
    if (!bitmap.width || !bitmap.height) throw new Error("The image has no readable pixels.");
    const scale = Math.min(1, 1024 / Math.max(bitmap.width, bitmap.height));
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    const context = canvas.getContext("2d");
    if (!context) throw new Error("This browser could not prepare the image.");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    return await new Promise((resolve, reject) => {
      canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("This browser could not prepare the image.")), "image/png");
    });
  } finally {
    bitmap.close();
    canvas.width = 0;
    canvas.height = 0;
  }
}

$("upload-image").addEventListener("click", () => {
  if (!busy && exampleHasImage()) $("image-file").click();
});
$("clear-image").addEventListener("click", () => {
  if (busy || !exampleHasImage()) return;
  clearImage();
  clearResult();
  syncControls();
  setStatus("Choose a sample or upload an image.");
});
$("image-file").addEventListener("change", async () => {
  const file = $("image-file").files[0];
  $("image-file").value = "";
  if (busy || !exampleHasImage() || !file) return;
  if (file.size === 0 || file.size > 20 * 1024 * 1024) {
    setStatus("Choose a PNG or JPEG image up to 20 MiB.", "error");
    return;
  }
  clearResult();
  const revision = clearImage("Preparing image…");
  setBusy(true);
  setStatus("Preparing image…", "running");
  try {
    const png = await normalizedPNG(file);
    if (revision !== imageRevision) return;
    setStatus("Uploading image…", "running");
    const selection = await readResponse(await fetch("/api/image", {
      method: "POST", headers: {"Content-Type": "image/png"}, body: png,
    }));
    if (revision !== imageRevision) return;
    await displayImage(selection, revision);
    if (revision === imageRevision) setStatus("Ready to run.");
  } catch (error) {
    if (revision === imageRevision) {
      $("image-placeholder").textContent = error.message;
      setStatus(error.message, "error");
    }
  } finally {
    if (revision === imageRevision) setBusy(false);
  }
});

$("model-select").addEventListener("change", async () => {
  if (busy) return;
  const model = $("model-select").value;
  clearResult();
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
    imageSamples = config.image_samples || [];
    $("model-select").value = currentModel;
    for (const example of examples) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example-button";
      button.dataset.example = example.id;
      button.dataset.modality = exampleHasImage(example) ? "image" : "text";
      button.title = exampleHasImage(example) ? "Image example" : "Text-only example";
      button.setAttribute("aria-label", `${example.title} (${exampleHasImage(example) ? "image" : "text only"})`);
      const title = document.createElement("span");
      title.className = "example-title";
      title.textContent = example.title;
      const icon = document.createElement("span");
      icon.className = "example-kind-icon";
      icon.setAttribute("aria-hidden", "true");
      button.append(title, icon);
      button.addEventListener("click", () => { if (!busy) void selectExample(example); });
      $("example-buttons").append(button);
    }
    for (const sample of imageSamples) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "image-sample";
      button.dataset.image = sample.id;
      button.textContent = sample.title;
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => { if (!busy) void chooseImage(sample); });
      $("image-samples").append(button);
    }
    await selectExample(examples[0]);
    setBusy(false);
  } catch (error) {
    setStatus(`Could not connect: ${error.message} Refresh to retry.`, "error");
  }
}

initialize();
