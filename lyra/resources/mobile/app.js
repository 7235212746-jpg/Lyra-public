"use strict";

const storageKey = "lyra.mobile.credentials.v1";
let credentials = loadCredentials();
let tasks = [];
let capabilities = null;
let pollTimer = null;
let zoom = 1;

const $ = (id) => document.getElementById(id);

function loadCredentials() {
  try { return JSON.parse(sessionStorage.getItem(storageKey) || "null"); }
  catch (_) { return null; }
}

function setCredentials(value) {
  credentials = value;
  if (value) sessionStorage.setItem(storageKey, JSON.stringify(value));
  else sessionStorage.removeItem(storageKey);
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (credentials) headers.set("Authorization", `Bearer ${credentials.token}`);
  if (options.body !== undefined) {
    headers.set("Content-Type", "application/json");
    headers.set("X-Lyra-CSRF", credentials ? credentials.csrf_token : "");
  }
  const response = await fetch(path, {...options, headers, body: options.body === undefined ? undefined : JSON.stringify(options.body), cache: "no-store"});
  const payload = await response.json().catch(() => ({}));
  if (response.status === 401 && path !== "/api/pair") {
    setCredentials(null);
    showPairing();
  }
  if (!response.ok) throw new Error(payload.error?.message || `HTTP ${response.status}`);
  return payload;
}

async function refreshHealth() {
  try {
    await api("/api/health");
    $("status-dot").className = "dot online";
    $("connection-text").textContent = "在线";
  } catch (_) {
    $("status-dot").className = "dot offline";
    $("connection-text").textContent = "离线";
  }
}

function showPairing() {
  $("pair-view").hidden = false;
  $("tasks-view").hidden = true;
  $("new-view").hidden = true;
  $("device-view").hidden = true;
  $("bottom-nav").hidden = true;
}

function showApplication() {
  $("pair-view").hidden = true;
  $("bottom-nav").hidden = false;
  switchView("tasks-view");
  refreshCapabilities();
  refreshTasks();
  clearInterval(pollTimer);
  pollTimer = setInterval(refreshTasks, 2500);
}

async function refreshCapabilities() {
  try {
    const payload = await api("/api/capabilities");
    capabilities = payload.capabilities || {};
    const directories = capabilities.directories || [];
    [$("directory-id"), $("band-directory-id"), $("dos-directory-id")].forEach((select) => {
      select.replaceChildren();
      directories.forEach((directory) => {
        const option = document.createElement("option");
        option.value = directory.id;
        option.textContent = directory.label;
        select.append(option);
      });
      select.disabled = directories.length === 0;
    });
    const reason = capabilities.projection?.reason || "";
    $("capability-note").textContent = directories.length ? reason : "请先在电脑端授权计算目录。";
    $("task-form").querySelector("button[type=submit]").disabled = directories.length === 0;
  } catch (_) { $("capability-note").textContent = "暂时无法读取桌面端能力。"; }
}

function switchView(viewId) {
  ["tasks-view", "new-view", "device-view"].forEach((id) => { $(id).hidden = id !== viewId; });
  document.querySelectorAll(".bottom-nav button").forEach((button) => button.classList.toggle("active", button.dataset.view === viewId));
}

function statusLabel(state) {
  return ({queued: "等待中", running: "运行中", completed: "已完成", failed: "失败", cancelling: "取消中", cancelled: "已取消"})[state] || state;
}

function kindLabel(kind) {
  return ({band: "能带", dos: "态密度", combined: "能带 + 态密度"})[kind] || kind;
}

function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderTasks() {
  const list = $("task-list");
  list.replaceChildren();
  $("empty-state").hidden = tasks.length !== 0;
  $("running-count").textContent = tasks.filter((task) => ["queued", "running", "cancelling"].includes(task.state)).length;
  $("completed-count").textContent = tasks.filter((task) => task.state === "completed").length;
  $("failed-count").textContent = tasks.filter((task) => ["failed", "cancelled"].includes(task.state)).length;

  tasks.forEach((task) => {
    const card = make("article", "task-card");
    const head = make("div", "task-head");
    const title = make("h2", "", kindLabel(task.kind));
    const badge = make("span", `state ${task.state}`, statusLabel(task.state));
    head.append(title, badge);
    const time = make("time", "", new Date(task.updated_at * 1000).toLocaleString());
    const stage = make("p", "stage", task.stage || "等待更新");
    const progress = make("div", "progress");
    const fill = make("span");
    fill.style.width = `${Math.round(Math.max(0, Math.min(1, task.progress)) * 100)}%`;
    progress.append(fill);
    const actions = make("div", "task-actions");
    (task.resources || []).forEach((resource) => {
      if (resource.previewable && resource.preview_url) {
        const button = make("button", "", "预览");
        button.type = "button";
        button.addEventListener("click", () => openPreview(resource));
        actions.append(button);
      }
      const link = make("a", "", "下载");
      link.href = resource.download_url;
      link.addEventListener("click", (event) => downloadResource(event, resource));
      actions.append(link);
    });
    if (["queued", "running"].includes(task.state)) {
      const cancel = make("button", "cancel", "取消");
      cancel.type = "button";
      cancel.addEventListener("click", () => cancelTask(task.id));
      actions.append(cancel);
    }
    card.append(head, time, stage, progress, actions);
    list.append(card);
  });
}

async function refreshTasks() {
  if (!credentials) return;
  try {
    const payload = await api("/api/tasks");
    tasks = payload.tasks || [];
    renderTasks();
    $("status-dot").className = "dot online";
    $("connection-text").textContent = "在线";
  } catch (_) {
    $("status-dot").className = "dot offline";
    $("connection-text").textContent = "离线";
  }
}

async function cancelTask(id) {
  try { await api(`/api/tasks/${id}/cancel`, {method: "POST", body: {}}); await refreshTasks(); }
  catch (error) { window.alert(error.message); }
}

async function authenticatedBlob(url) {
  const response = await fetch(url, {headers: {Authorization: `Bearer ${credentials.token}`}, cache: "no-store"});
  if (response.status === 401) { setCredentials(null); showPairing(); }
  if (!response.ok) throw new Error("无法读取文件");
  return response.blob();
}

async function openPreview(resource) {
  try {
    const blob = await authenticatedBlob(resource.preview_url);
    const image = $("preview-image");
    if (image.dataset.objectUrl) URL.revokeObjectURL(image.dataset.objectUrl);
    const objectUrl = URL.createObjectURL(blob);
    image.dataset.objectUrl = objectUrl;
    image.src = objectUrl;
    $("preview-title").textContent = resource.label || "预览";
    zoom = 1;
    image.style.transform = "scale(1)";
    $("preview-dialog").showModal();
  } catch (error) { window.alert(error.message); }
}

async function downloadResource(event, resource) {
  event.preventDefault();
  try {
    const blob = await authenticatedBlob(resource.download_url);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = resource.label || "lyra-result";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { window.alert(error.message); }
}

function applyZoom(delta) {
  zoom = Math.max(.5, Math.min(5, zoom + delta));
  $("preview-image").style.transform = `scale(${zoom})`;
}

$("pair-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("pair-error").textContent = "";
  try {
    const result = await api("/api/pair", {method: "POST", body: {code: $("pair-code").value, device_name: $("device-name").value}});
    setCredentials(result);
    showApplication();
  } catch (error) { $("pair-error").textContent = error.message; }
});

$("task-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("task-error").textContent = "";
  const minimum = Number($("energy-min").value);
  const maximum = Number($("energy-max").value);
  if (!(minimum < maximum)) { $("task-error").textContent = "能量下限必须小于上限"; return; }
  try {
    const kind = $("task-kind").value;
    const parameters = {energy_range: [minimum, maximum], read_mode: $("read-mode").value};
    if (kind === "combined") {
      parameters.band_directory_id = $("band-directory-id").value;
      parameters.dos_directory_id = $("dos-directory-id").value;
    } else {
      parameters.directory_id = $("directory-id").value;
    }
    await api("/api/tasks", {method: "POST", body: {kind, parameters}});
    switchView("tasks-view");
    await refreshTasks();
  } catch (error) { $("task-error").textContent = error.message; }
});

$("task-kind").addEventListener("change", () => {
  const combined = $("task-kind").value === "combined";
  $("single-directory-row").hidden = combined;
  $("combined-directory-row").hidden = !combined;
});

document.querySelectorAll(".bottom-nav button").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
$("refresh-button").addEventListener("click", refreshTasks);
$("preview-close").addEventListener("click", () => $("preview-dialog").close());
$("zoom-in").addEventListener("click", () => applyZoom(.25));
$("zoom-out").addEventListener("click", () => applyZoom(-.25));
$("revoke-button").addEventListener("click", async () => {
  if (!credentials || !window.confirm("撤销后需要重新配对，确定继续吗？")) return;
  try { await api(`/api/devices/${credentials.device_id}/revoke`, {method: "POST", body: {}}); }
  finally { setCredentials(null); showPairing(); }
});

window.addEventListener("online", refreshHealth);
window.addEventListener("offline", refreshHealth);
refreshHealth();
if (credentials) showApplication(); else showPairing();
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js");
