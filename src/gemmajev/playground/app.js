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
let audioSamples = [];
let currentAttachment = null;
let attachmentRevision = 0;

function exampleModality(example = selectedExample) {
  return example?.modality || (example?.audio_id ? "audio" : example?.image_id ? "image" : "text");
}

function attachmentPayload() {
  return currentAttachment ? {[`${currentAttachment.kind}_id`]: currentAttachment.id} : {};
}

function sameAttachment(first, second) {
  return (first.image_id || null) === (second.image_id || null) && (first.audio_id || null) === (second.audio_id || null);
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
  const modality = exampleModality();
  $("attachment-panel").querySelectorAll("button, input, select").forEach((control) => {
    control.disabled = busy || modality === "text";
  });
  $("clear-attachment").disabled = busy || modality === "text" || currentAttachment === null;
  $("attachment-panel").setAttribute("aria-disabled", String(modality === "text"));
  $("image-samples").hidden = modality !== "image";
  $("audio-sample").hidden = modality !== "audio";
  $("attachment-file").accept = modality === "audio" ? ".wav,audio/wav,audio/x-wav" : "image/png,image/jpeg";
  // Playback does not change model input; users can always pause a loaded clip.
  $("audio-play").disabled = currentAttachment?.kind !== "audio";
  $("audio-seek").disabled = currentAttachment?.kind !== "audio";
  $("input-panel").setAttribute("aria-busy", String(busy));
  $("attachment-panel").setAttribute("aria-busy", String(busy));
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
    matches = lastResult.model === currentModel && sameAttachment(lastResult, attachmentPayload()) && JSON.stringify(getRequest()) === JSON.stringify(lastResult.request);
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

function clearAttachment(message) {
  attachmentRevision += 1;
  currentAttachment = null;
  const player = $("audio-source");
  player.pause();
  player.removeAttribute("src");
  player.load();
  $("audio-player").hidden = true;
  $("audio-sample").value = "";
  $("audio-play").textContent = "Play";
  $("audio-seek").value = "0";
  $("audio-time").textContent = "0:00 / 0:00";
  $("image-preview").removeAttribute("src");
  $("image-preview").hidden = true;
  const modality = exampleModality();
  $("attachment-placeholder").textContent = message || (modality === "text"
    ? "This example uses text only. Select an image or audio example to add an attachment."
    : modality === "audio" ? "Choose an audio sample or upload a WAV clip." : "No image attached");
  $("attachment-placeholder").hidden = false;
  document.querySelectorAll(".image-sample").forEach((button) => button.setAttribute("aria-pressed", "false"));
  return attachmentRevision;
}

function localAttachmentURL(value, kind) {
  const url = new URL(value, window.location.origin);
  const prefix = kind === "audio" ? "/api/audio/" : "/api/images/";
  if (url.origin !== window.location.origin || !url.pathname.startsWith(prefix)) {
    throw new Error("The attachment must come from this Playground.");
  }
  return url.href;
}

function playbackTime(value) {
  const seconds = Math.max(0, Math.floor(Number.isFinite(value) ? value : 0));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function updatePlayback() {
  const player = $("audio-source");
  const duration = Number.isFinite(player.duration) ? player.duration : 0;
  $("audio-play").textContent = player.paused || player.ended ? "Play" : "Pause";
  $("audio-seek").max = String(duration);
  $("audio-seek").value = String(player.currentTime || 0);
  $("audio-seek").setAttribute("aria-valuetext", `${playbackTime(player.currentTime)} of ${playbackTime(duration)}`);
  $("audio-time").textContent = `${playbackTime(player.currentTime)} / ${playbackTime(duration)}`;
}

function disposeAudio(player) {
  player.pause();
  player.removeAttribute("src");
  player.load();
}

async function displayAttachment(kind, selection, revision) {
  const url = localAttachmentURL(selection.url, kind);
  const id = selection.id || selection[`${kind}_id`];
  if (typeof id !== "string" || !id) throw new Error("The attachment has no valid ID.");
  if (kind === "image") {
    const preview = new Image();
    preview.id = "image-preview";
    preview.alt = selection.title || "Uploaded image";
    preview.src = url;
    try {
      await preview.decode();
    } catch {
      throw new Error("Could not load the image. Check that the Playground server is running, then try again.");
    }
    if (revision !== attachmentRevision) return;
    $("image-preview").replaceWith(preview);
    document.querySelectorAll(".image-sample").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.image === id));
    });
  } else {
    const player = new Audio();
    player.id = "audio-source";
    player.hidden = true;
    player.preload = "metadata";
    try {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => finish(new Error("Audio loading timed out. Try selecting it again.")), 15000);
        const loaded = () => finish();
        const failed = () => finish(new Error("Could not load the audio. Check that the Playground server is running, then try again."));
        function finish(error) {
          clearTimeout(timer);
          player.removeEventListener("loadedmetadata", loaded);
          player.removeEventListener("error", failed);
          error ? reject(error) : resolve();
        }
        player.addEventListener("loadedmetadata", loaded);
        player.addEventListener("error", failed);
        player.src = url;
        player.load();
      });
      if (!Number.isFinite(player.duration) || player.duration < 0.04 || player.duration > 30) {
        throw new Error("Choose an audio clip between 40 milliseconds and 30 seconds.");
      }
      if (revision !== attachmentRevision) {
        disposeAudio(player);
        return;
      }
      $("audio-source").replaceWith(player);
      for (const event of ["timeupdate", "play", "pause", "ended", "durationchange"]) {
        player.addEventListener(event, () => { if ($("audio-source") === player) updatePlayback(); });
      }
      player.addEventListener("error", () => {
        if ($("audio-source") === player && currentAttachment?.id === id) {
          setStatus("Audio playback failed. Select the clip again.", "error");
        }
      });
      $("audio-name").textContent = selection.title || "Uploaded audio";
      $("audio-player").hidden = false;
      $("audio-sample").value = audioSamples.some((sample) => sample.id === id) ? id : "";
      updatePlayback();
    } catch (error) {
      disposeAudio(player);
      throw error;
    }
  }
  $("attachment-placeholder").hidden = true;
  currentAttachment = {kind, id, url: selection.url};
}

async function chooseAttachment(kind, selection) {
  if (exampleModality() !== kind) return;
  clearResult();
  const revision = clearAttachment(`Loading ${kind}…`);
  setBusy(true);
  setStatus(`Loading ${kind}…`, "running");
  try {
    await displayAttachment(kind, selection, revision);
    if (revision === attachmentRevision) setStatus("Ready to run.");
  } catch (error) {
    if (revision === attachmentRevision) {
      $("attachment-placeholder").textContent = error.message;
      setStatus(error.message, "error");
    }
  } finally {
    if (revision === attachmentRevision) setBusy(false);
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
  clearAttachment();
  document.querySelectorAll(".example-button").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.example === example.id));
  });
  const kind = exampleModality();
  if (kind !== "text") {
    const sample = (kind === "audio" ? audioSamples : imageSamples).find((item) => item.id === example[`${kind}_id`]);
    if (sample) {
      await chooseAttachment(kind, sample);
      return;
    }
    setStatus(`Choose a sample or upload ${kind === "audio" ? "a WAV clip" : "an image"}.`);
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
  const comparable = previous && sameAttachment(result, previous) && sameOptions(result.request.options, previous.request.options);
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
    if (exampleModality() !== "text" && !currentAttachment) {
      throw new Error("Choose a sample or upload an attachment before running this example.");
    }
    request = getRequest();
  } catch (error) {
    setStatus(error.message + (lastResult ? " Previous result shown." : ""), "error");
    return;
  }
  setBusy(true);
  setStatus("Running…", "running");
  try {
    const result = await api("/api/decide", {model: currentModel, request, ...attachmentPayload()});
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

function wavInfo(buffer) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const tag = (offset) => String.fromCharCode(...bytes.subarray(offset, offset + 4));
  const invalid = () => new Error("Choose an uncompressed PCM or float WAV file with one or two channels.");
  if (bytes.length < 44 || bytes.length > 20 * 1024 * 1024 || tag(0) !== "RIFF" || tag(8) !== "WAVE" || view.getUint32(4, true) + 8 !== bytes.length) {
    throw invalid();
  }
  let format = null;
  let dataBytes = null;
  let offset = 12;
  while (offset < bytes.length) {
    if (offset + 8 > bytes.length) throw invalid();
    const kind = tag(offset);
    const length = view.getUint32(offset + 4, true);
    const start = offset + 8;
    const end = start + length;
    if (end > bytes.length) throw invalid();
    if (kind === "fmt ") {
      if (format || length < 16) throw invalid();
      format = {
        codec: view.getUint16(start, true), channels: view.getUint16(start + 2, true),
        rate: view.getUint32(start + 4, true), byteRate: view.getUint32(start + 8, true),
        align: view.getUint16(start + 12, true), bits: view.getUint16(start + 14, true),
      };
    } else if (kind === "data") {
      if (dataBytes !== null) throw invalid();
      dataBytes = length;
    }
    offset = end + (length % 2);
    if (offset > bytes.length) throw invalid();
  }
  if (!format || dataBytes === null || ![1, 2].includes(format.channels)
      || format.rate < 8000 || format.rate > 192000
      || !((format.codec === 1 && [8, 16, 24, 32].includes(format.bits)) || (format.codec === 3 && format.bits === 32))
      || format.align !== format.channels * format.bits / 8 || format.byteRate !== format.rate * format.align
      || dataBytes % format.align !== 0) throw invalid();
  const duration = dataBytes / format.align / format.rate;
  if (duration < 0.04 || duration > 30) {
    throw new Error("Choose an audio clip between 40 milliseconds and 30 seconds.");
  }
  return {duration, channels: format.channels, sampleRate: format.rate};
}

function pcm16WAV(audio) {
  if (audio.sampleRate !== 16000 || ![1, 2].includes(audio.numberOfChannels) || audio.length < 640 || audio.length > 480000) {
    throw new Error("Audio conversion must produce 40 milliseconds to 30 seconds at 16 kHz.");
  }
  const output = new ArrayBuffer(44 + audio.length * 2);
  const view = new DataView(output);
  const text = (offset, value) => { for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i)); };
  text(0, "RIFF"); view.setUint32(4, output.byteLength - 8, true); text(8, "WAVE");
  text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, 16000, true); view.setUint32(28, 32000, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, audio.length * 2, true);
  const channels = Array.from({length: audio.numberOfChannels}, (_, index) => audio.getChannelData(index));
  for (let frame = 0; frame < audio.length; frame += 1) {
    let sample = 0;
    for (const channel of channels) {
      if (!Number.isFinite(channel[frame])) throw new Error("The audio contains invalid sample values.");
      sample += channel[frame] / channels.length;
    }
    sample = Math.max(-1, Math.min(1, sample));
    view.setInt16(44 + frame * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
  }
  return new Blob([output], {type: "audio/wav"});
}

async function normalizedWAV(file) {
  const buffer = await file.arrayBuffer();
  wavInfo(buffer); // Check decoded duration and allocation bounds before invoking Web Audio.
  const OfflineContext = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!OfflineContext) throw new Error("This browser does not support audio conversion.");
  const context = new OfflineContext(1, 1, 16000);
  let audio;
  try {
    audio = await context.decodeAudioData(buffer);
  } catch {
    throw new Error("This WAV file could not be decoded. Choose another PCM or float WAV file.");
  }
  return pcm16WAV(audio);
}

$("upload-attachment").addEventListener("click", () => {
  if (!busy && exampleModality() !== "text") $("attachment-file").click();
});
$("clear-attachment").addEventListener("click", () => {
  if (busy || exampleModality() === "text") return;
  clearAttachment();
  clearResult();
  syncControls();
  setStatus("Choose a sample or upload an attachment.");
});
$("attachment-file").addEventListener("change", async () => {
  const file = $("attachment-file").files[0];
  $("attachment-file").value = "";
  const kind = exampleModality();
  if (busy || kind === "text" || !file) return;
  if (file.size === 0 || file.size > 20 * 1024 * 1024) {
    setStatus(`Choose ${kind === "audio" ? "a WAV file" : "a PNG or JPEG image"} up to 20 MiB.`, "error");
    return;
  }
  clearResult();
  const revision = clearAttachment(`Preparing ${kind}…`);
  setBusy(true);
  setStatus(`Preparing ${kind}…`, "running");
  try {
    const data = kind === "audio" ? await normalizedWAV(file) : await normalizedPNG(file);
    if (revision !== attachmentRevision) return;
    setStatus(`Uploading ${kind}…`, "running");
    const selection = await readResponse(await fetch(kind === "audio" ? "/api/audio" : "/api/image", {
      method: "POST", headers: {"Content-Type": kind === "audio" ? "audio/wav" : "image/png"}, body: data,
    }));
    if (revision !== attachmentRevision) return;
    await displayAttachment(kind, selection, revision);
    if (revision === attachmentRevision) setStatus("Ready to run.");
  } catch (error) {
    if (revision === attachmentRevision) {
      $("attachment-placeholder").textContent = error.message;
      setStatus(error.message, "error");
    }
  } finally {
    if (revision === attachmentRevision) setBusy(false);
  }
});
$("audio-sample").addEventListener("change", () => {
  if (busy || exampleModality() !== "audio") return;
  const sample = audioSamples.find((item) => item.id === $("audio-sample").value);
  if (sample) void chooseAttachment("audio", sample);
});
$("audio-play").addEventListener("click", async () => {
  if (currentAttachment?.kind !== "audio") return;
  const player = $("audio-source");
  if (!player.paused) { player.pause(); return; }
  try {
    if (player.ended) player.currentTime = 0;
    await player.play();
  } catch {
    if (player === $("audio-source") && currentAttachment?.kind === "audio") {
      setStatus("Could not play the audio. Press Play to try again.", "error");
    }
  }
});
$("audio-seek").addEventListener("input", () => {
  if (currentAttachment?.kind !== "audio") return;
  const player = $("audio-source");
  player.currentTime = Number($("audio-seek").value);
  updatePlayback();
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
    audioSamples = config.audio_samples || [];
    $("model-select").value = currentModel;
    for (const example of examples) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example-button";
      button.dataset.example = example.id;
      const modality = exampleModality(example);
      button.dataset.modality = modality;
      button.title = `${modality === "text" ? "Text-only" : modality === "audio" ? "Audio" : "Image"} example`;
      button.setAttribute("aria-label", `${example.title} (${modality === "text" ? "text only" : modality})`);
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
      button.addEventListener("click", () => { if (!busy) void chooseAttachment("image", sample); });
      $("image-samples").append(button);
    }
    for (const sample of audioSamples) {
      const option = document.createElement("option");
      option.value = sample.id;
      option.textContent = sample.title;
      $("audio-sample").append(option);
    }
    await selectExample(examples[0]);
    setBusy(false);
  } catch (error) {
    setStatus(`Could not connect: ${error.message} Refresh to retry.`, "error");
  }
}

initialize();
