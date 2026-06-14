const boardEl = document.querySelector("#board");
const colCoords = document.querySelector("#colCoords");
const rowCoords = document.querySelector("#rowCoords");
const statusPill = document.querySelector("#statusPill");
const simSlider = document.querySelector("#simSlider");
const simValue = document.querySelector("#simValue");
const sideLabel = document.querySelector("#sideLabel");
const movesEl = document.querySelector("#moves");
const turnEl = document.querySelector("#turn");
const historyList = document.querySelector("#historyList");
const historyCount = document.querySelector("#historyCount");
const candBody = document.querySelector("#candBody");
const policyCount = document.querySelector("#policyCount");
const toast = document.querySelector("#toast");
const newGameBtn = document.querySelector("#newGame");
const undoBtn = document.querySelector("#undo");
const analyzeBtn = document.querySelector("#analyze");
const pvToggle = document.querySelector("#pvToggle");
const overlayNote = document.querySelector("#overlayNote");
const evalBlack = document.querySelector("#evalBlack");
const evalLabel = document.querySelector("#evalLabel");
const winMeterBlack = document.querySelector("#winMeterBlack");
const winProbEl = document.querySelector("#winProb");
const evalSource = document.querySelector("#evalSource");
const statSims = document.querySelector("#statSims");
const statTime = document.querySelector("#statTime");
const chartCanvas = document.querySelector("#evalChart");
const loadLine = document.querySelector("#loadLine");
const loadBar = document.querySelector("#loadBar");
const loadText = document.querySelector("#loadText");
const sideButtons = [...document.querySelectorAll(".segment[data-side]")];
const overlayButtons = [...document.querySelectorAll(".segment[data-overlay]")];

let state = null;
let selectedSide = "black";
let overlayMode = "none";
let busy = true;
let cells = [];
let requestId = 0;
const pending = new Map();

const worker = new Worker("./engine.worker.js");

const playerName = (v) => (v === 1 ? "黑" : v === -1 ? "白" : "-");
const pct = (v, digits = 0) => `${(v * 100).toFixed(digits)}%`;

function showToast(message) {
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toast.hidden = true;
  }, 3200);
}

function setBusy(nextBusy, label = "") {
  busy = nextBusy;
  [newGameBtn, undoBtn, analyzeBtn, simSlider, ...sideButtons].forEach((el) => {
    el.disabled = busy;
  });
  boardEl.classList.toggle("busy", busy);
  if (busy) {
    statusPill.textContent = label || "AI 思考中";
    statusPill.className = "status-pill thinking";
  }
}

function callWorker(type, payload = {}) {
  const id = ++requestId;
  worker.postMessage({ id, type, payload });
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
  });
}

worker.addEventListener("message", (event) => {
  const msg = event.data;
  if (msg.type === "progress") {
    const pctValue = msg.payload.total ? msg.payload.loaded / msg.payload.total : 0;
    loadBar.style.width = `${Math.max(2, Math.min(100, pctValue * 100)).toFixed(1)}%`;
    loadText.textContent = msg.payload.label;
    return;
  }
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    if (msg.error) reject(new Error(msg.error));
    else resolve(msg.payload);
  }
});

worker.addEventListener("error", (event) => {
  showToast(event.message || "Worker 运行失败");
  setBusy(false);
});

function buildBoard(size) {
  boardEl.innerHTML = "";
  colCoords.innerHTML = "";
  rowCoords.innerHTML = "";
  cells = [];
  for (let i = 0; i < size; i += 1) {
    const col = document.createElement("div");
    col.textContent = i + 1;
    colCoords.appendChild(col);
    const row = document.createElement("div");
    row.textContent = i + 1;
    rowCoords.appendChild(row);
  }
  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const cell = document.createElement("button");
      cell.className = "cell";
      cell.type = "button";
      cell.setAttribute("role", "gridcell");
      cell.setAttribute("aria-label", `第 ${row + 1} 行第 ${col + 1} 列`);
      cell.addEventListener("click", () => makeMove(row, col));
      boardEl.appendChild(cell);
      cells.push(cell);
    }
  }
}

function renderBoard() {
  const size = state.size;
  const last = state.lastMove === null ? null : {
    row: Math.floor(state.lastMove / size),
    col: state.lastMove % size,
  };
  const isHumanTurn = !state.winner && state.currentPlayer === state.humanPlayer;

  cells.forEach((cell, idx) => {
    const row = Math.floor(idx / size);
    const col = idx % size;
    const value = state.board[row][col];
    cell.innerHTML = "";
    cell.className = "cell";
    if (value !== 0) {
      const stone = document.createElement("span");
      stone.className = `stone ${value === 1 ? "black" : "white"}`;
      cell.appendChild(stone);
      cell.classList.add("disabled");
    }
    if (!isHumanTurn || value !== 0 || busy) cell.classList.add("disabled");
    if (last && row === last.row && col === last.col) cell.classList.add("last");
  });
  paintOverlays();
}

function paintOverlays() {
  const a = state.analysis;
  if (!a || !a.visitMap) return;
  const map = overlayMode === "search" ? a.visitMap
    : overlayMode === "prior" ? a.priorMap : null;
  if (map) {
    const max = Math.max(...map, 1e-9);
    const rgb = overlayMode === "search"
      ? getComputedStyle(document.documentElement).getPropertyValue("--search-heat")
      : getComputedStyle(document.documentElement).getPropertyValue("--prior-heat");
    map.forEach((v, idx) => {
      if (v < 5e-4) return;
      const t = Math.sqrt(v / max);
      const heat = document.createElement("span");
      heat.className = "heat";
      heat.style.background = `rgba(${rgb.trim().replaceAll(" ", ",")},${(0.1 + 0.55 * t).toFixed(3)})`;
      cells[idx].appendChild(heat);
      if (v >= 0.02) {
        const label = document.createElement("span");
        label.className = "heatlabel";
        label.textContent = v >= 0.095 ? pct(v) : pct(v, 1);
        cells[idx].appendChild(label);
      }
    });
  }

  if (pvToggle.checked && a.pv && a.pv.length) {
    a.pv.forEach((m, i) => {
      if (state.board[m.row][m.col] !== 0) return;
      const mover = i % 2 === 0 ? a.player : -a.player;
      const badge = document.createElement("span");
      badge.className = `pvbadge ${mover === 1 ? "black" : "white"}`;
      badge.textContent = i + 1;
      cells[m.row * state.size + m.col].appendChild(badge);
    });
  }
}

function blackWinProb(a) {
  if (!a || a.winProb === undefined) return null;
  return a.player === 1 ? a.winProb : 1 - a.winProb;
}

function renderEval() {
  const a = state.analysis;
  const bw = blackWinProb(a);
  if (bw === null) {
    winProbEl.textContent = "-";
    evalLabel.textContent = "-";
    evalBlack.style.height = "50%";
    winMeterBlack.style.width = "50%";
    evalSource.textContent = "-";
    statSims.textContent = "-";
    statTime.textContent = "-";
  } else {
    winProbEl.textContent = pct(bw);
    evalLabel.textContent = pct(bw);
    evalBlack.style.height = `${(bw * 100).toFixed(1)}%`;
    winMeterBlack.style.width = `${(bw * 100).toFixed(1)}%`;
    evalSource.textContent = state.policySource === "analysis"
      ? `当前局面 · ${playerName(a.player)}方行棋`
      : `AI 第 ${a.moveNumber + 1} 手搜索`;
    statSims.textContent = a.simulations;
    statTime.textContent = a.elapsedMs >= 1000
      ? `${(a.elapsedMs / 1000).toFixed(1)}s`
      : `${Math.round(a.elapsedMs)}ms`;
  }
  movesEl.textContent = state.movesPlayed;
  turnEl.textContent = state.winner !== null ? "-" : playerName(state.currentPlayer);

  if (!a || !a.visitMap) {
    overlayNote.textContent = "搜索=实际模拟访问占比；先验=神经网络原始偏好";
  } else {
    const src = state.policySource === "analysis" ? "当前局面分析" : "AI 上一手搜索";
    overlayNote.textContent = `${src} · ${a.simulations} sims`;
  }
}

function renderCandidates() {
  const a = state.analysis;
  candBody.innerHTML = "";
  const cands = (a && a.candidates) || [];
  policyCount.textContent = cands.length;
  cands.forEach((c) => {
    const tr = document.createElement("tr");
    if (c.selected) tr.classList.add("best");
    const win = c.q === null ? null : (c.q + 1) / 2;
    const shareWidth = Math.max(2, Math.min(100, c.share * 100)).toFixed(1);
    tr.innerHTML = `
      <td>${c.selected ? "✓ " : ""}${c.row + 1},${c.col + 1}</td>
      <td>${c.visits}</td>
      <td><span class="share-cell" style="--share:${shareWidth}%"><span>${pct(c.share)}</span></span></td>
      <td>${pct(c.prior, 1)}</td>
      <td class="${win === null ? "" : win >= 0.5 ? "q-good" : "q-bad"}">
        ${win === null ? "-" : pct(win)}</td>`;
    const idx = c.row * state.size + c.col;
    tr.addEventListener("mouseenter", () => cells[idx].classList.add("hl"));
    tr.addEventListener("mouseleave", () => cells[idx].classList.remove("hl"));
    candBody.appendChild(tr);
  });
}

function renderHistory() {
  historyCount.textContent = state.movesPlayed;
  historyList.innerHTML = "";
  [...state.history].reverse().forEach((move, index) => {
    const item = document.createElement("li");
    const n = state.movesPlayed - index;
    item.innerHTML = `<span>${n}. ${move.player === "black" ? "黑" : "白"} ${move.row + 1},${move.col + 1}</span>
      <span class="move-source">${move.source === "ai" ? "AI" : "人类"}</span>`;
    historyList.appendChild(item);
  });
}

function renderChart() {
  const ctx = chartCanvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssW = chartCanvas.clientWidth || 280;
  const cssH = chartCanvas.clientHeight || 110;
  const style = getComputedStyle(document.documentElement);
  chartCanvas.width = cssW * dpr;
  chartCanvas.height = cssH * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = { l: 8, r: 8, t: 8, b: 16 };
  const w = cssW - pad.l - pad.r;
  const h = cssH - pad.t - pad.b;
  const data = state.evalHistory || [];

  ctx.strokeStyle = "rgba(111,120,109,0.22)";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.l, pad.t, w, h);
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t + h / 2);
  ctx.lineTo(pad.l + w, pad.t + h / 2);
  ctx.stroke();
  ctx.setLineDash([]);

  if (!data.length) return;

  const maxMove = Math.max(...data.map((d) => d.move), 1);
  const x = (m) => pad.l + (m / maxMove) * w;
  const y = (p) => pad.t + (1 - p) * h;

  ctx.beginPath();
  data.forEach((d, i) => {
    if (i === 0) ctx.moveTo(x(d.move), y(d.blackWinProb));
    else ctx.lineTo(x(d.move), y(d.blackWinProb));
  });
  ctx.lineTo(x(data[data.length - 1].move), pad.t + h);
  ctx.lineTo(x(data[0].move), pad.t + h);
  ctx.closePath();
  const fill = ctx.createLinearGradient(0, pad.t, 0, pad.t + h);
  fill.addColorStop(0, "rgba(27,122,115,0.18)");
  fill.addColorStop(1, "rgba(194,75,51,0.06)");
  ctx.fillStyle = fill;
  ctx.fill();

  ctx.beginPath();
  data.forEach((d, i) => {
    if (i === 0) ctx.moveTo(x(d.move), y(d.blackWinProb));
    else ctx.lineTo(x(d.move), y(d.blackWinProb));
  });
  ctx.strokeStyle = style.getPropertyValue("--accent-2").trim() || "rgba(27,122,115,0.95)";
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke();
}

function render(nextState) {
  state = nextState;
  if (!cells.length) buildBoard(state.size);

  renderBoard();
  renderEval();
  renderCandidates();
  renderHistory();
  renderChart();

  simValue.textContent = state.simulations;
  simSlider.value = Math.min(Math.max(state.simulations, simSlider.min), simSlider.max);
  sideLabel.textContent = state.humanPlayer === 1 ? "执黑" : "执白";
  statusPill.textContent = state.status;
  statusPill.className = `status-pill ${state.winner !== null ? "done" : ""}`;
  undoBtn.disabled = !state.canUndo || busy;
}

async function newGame() {
  try {
    setBusy(true, "新对局");
    render(await callWorker("newGame", {
      human: selectedSide,
      simulations: Number(simSlider.value),
    }));
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
    if (state) render(state);
  }
}

async function makeMove(row, col) {
  if (busy || !state || state.winner !== null || state.currentPlayer !== state.humanPlayer) return;
  if (state.board[row][col] !== 0) return;
  try {
    setBusy(true, "AI 思考中");
    render(await callWorker("move", {
      row,
      col,
      simulations: Number(simSlider.value),
    }));
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
    if (state) render(state);
  }
}

async function undo() {
  if (busy) return;
  try {
    setBusy(true, "悔棋");
    render(await callWorker("undo"));
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
    if (state) render(state);
  }
}

async function analyze() {
  if (busy || !state || state.winner !== null) return;
  try {
    setBusy(true, "分析中");
    render(await callWorker("analyze", { simulations: Number(simSlider.value) }));
    if (overlayMode === "none") setOverlay("search");
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
    if (state) render(state);
  }
}

function setOverlay(mode) {
  overlayMode = mode;
  overlayButtons.forEach((b) => b.classList.toggle("active", b.dataset.overlay === mode));
  if (state) renderBoard();
}

sideButtons.forEach((button) => {
  button.addEventListener("click", () => {
    selectedSide = button.dataset.side;
    sideButtons.forEach((item) => item.classList.toggle("active", item === button));
    sideLabel.textContent = selectedSide === "black" ? "执黑" : "执白";
  });
});

overlayButtons.forEach((button) => {
  button.addEventListener("click", () => setOverlay(button.dataset.overlay));
});

pvToggle.addEventListener("change", () => {
  if (state) renderBoard();
});
simSlider.addEventListener("input", () => {
  simValue.textContent = simSlider.value;
});
newGameBtn.addEventListener("click", newGame);
undoBtn.addEventListener("click", undo);
analyzeBtn.addEventListener("click", analyze);
window.addEventListener("resize", () => {
  if (state) renderChart();
});

async function boot() {
  try {
    setBusy(true, "加载模型");
    await callWorker("init");
    loadLine.classList.add("ready");
    setBusy(false);
    await newGame();
  } catch (error) {
    statusPill.textContent = "加载失败";
    statusPill.className = "status-pill done";
    showToast(error.message);
  }
}

boot();
