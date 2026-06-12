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
const winProbEl = document.querySelector("#winProb");
const evalSource = document.querySelector("#evalSource");
const statSims = document.querySelector("#statSims");
const statTime = document.querySelector("#statTime");
const chartCanvas = document.querySelector("#evalChart");
const sideButtons = [...document.querySelectorAll(".segment[data-side]")];
const overlayButtons = [...document.querySelectorAll(".segment[data-overlay]")];

let state = null;
let selectedSide = "black";
let overlayMode = "none";
let busy = false;
let cells = [];

const playerName = (v) => (v === 1 ? "黑" : v === -1 ? "白" : "—");
const pct = (v, digits = 0) => `${(v * 100).toFixed(digits)}%`;
const coordText = (m) => `${m.row + 1},${m.col + 1}`;

function showToast(message) {
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { toast.hidden = true; }, 2800);
}

function setBusy(nextBusy) {
  busy = nextBusy;
  [newGameBtn, undoBtn, analyzeBtn].forEach((b) => { b.disabled = busy; });
  boardEl.classList.toggle("busy", busy);
  if (busy) {
    statusPill.textContent = "AI 思考中…";
    statusPill.className = "status-pill thinking";
  }
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "请求失败");
  return payload;
}

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
      cell.setAttribute("aria-label", `第 ${row + 1} 行 第 ${col + 1} 列`);
      cell.addEventListener("click", () => makeMove(row, col));
      boardEl.appendChild(cell);
      cells.push(cell);
    }
  }
}

/* ---------- board rendering ---------- */

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
    if (last && row === last.row && col === last.col) {
      cell.classList.add("last");
      if (state.lastMove !== renderBoard.prevLast) {
        const stone = cell.querySelector(".stone");
        if (stone) stone.classList.add("new");
      }
    }
  });
  renderBoard.prevLast = state.lastMove;
  paintOverlays();
}

function paintOverlays() {
  const a = state.analysis;
  if (!a || !a.visitMap) return;
  const size = state.size;

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
      if (state.board[m.row][m.col] !== 0) return; // move already played
      const mover = i % 2 === 0 ? a.player : -a.player;
      const badge = document.createElement("span");
      badge.className = `pvbadge ${mover === 1 ? "black" : "white"}`;
      badge.textContent = i + 1;
      cells[m.row * state.size + m.col].appendChild(badge);
    });
  }
}

/* ---------- panels ---------- */

function blackWinProb(a) {
  if (!a || a.winProb === undefined) return null;
  return a.player === 1 ? a.winProb : 1 - a.winProb;
}

function renderEval() {
  const a = state.analysis;
  const bw = blackWinProb(a);
  if (bw === null) {
    winProbEl.textContent = "—";
    evalLabel.textContent = "—";
    evalBlack.style.height = "50%";
    evalSource.textContent = "—";
    statSims.textContent = "—";
    statTime.textContent = "—";
  } else {
    winProbEl.textContent = pct(bw);
    evalLabel.textContent = pct(bw);
    evalBlack.style.height = `${(bw * 100).toFixed(1)}%`;
    evalSource.textContent = state.policySource === "analysis"
      ? `当前局面 · ${playerName(a.player)}方行棋`
      : `AI 第 ${a.moveNumber + 1} 手搜索`;
    statSims.textContent = a.simulations;
    statTime.textContent = a.elapsedMs >= 1000
      ? `${(a.elapsedMs / 1000).toFixed(1)}s`
      : `${Math.round(a.elapsedMs)}ms`;
  }
  movesEl.textContent = state.movesPlayed;
  turnEl.textContent = state.winner !== null ? "—" : playerName(state.currentPlayer);

  if (!a || !a.visitMap) {
    overlayNote.textContent = "尚无分析数据（落一子或点「分析局面」）";
  } else {
    const src = state.policySource === "analysis" ? "当前局面分析" : "AI 上一手的搜索";
    overlayNote.textContent = `数据来源：${src} · ${a.simulations} sims`;
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
    tr.innerHTML = `
      <td>${c.selected ? "★ " : ""}${c.row + 1},${c.col + 1}</td>
      <td>${c.visits}</td>
      <td>${pct(c.share)}</td>
      <td>${pct(c.prior, 1)}</td>
      <td class="${win === null ? "" : win >= 0.5 ? "q-good" : "q-bad"}">
        ${win === null ? "—" : pct(win)}</td>`;
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
  chartCanvas.width = cssW * dpr;
  chartCanvas.height = cssH * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = { l: 8, r: 8, t: 8, b: 16 };
  const w = cssW - pad.l - pad.r;
  const h = cssH - pad.t - pad.b;
  const data = state.evalHistory || [];

  // frame + 50% line
  ctx.strokeStyle = "rgba(120,110,90,0.25)";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.l, pad.t, w, h);
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t + h / 2);
  ctx.lineTo(pad.l + w, pad.t + h / 2);
  ctx.stroke();
  ctx.setLineDash([]);

  if (!data.length) {
    ctx.fillStyle = "rgba(120,110,90,0.7)";
    ctx.font = "11px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("每次搜索后记录黑方胜率", cssW / 2, cssH / 2 + 4);
    return;
  }

  const maxMove = Math.max(...data.map((d) => d.move), 1);
  const x = (m) => pad.l + (m / maxMove) * w;
  const y = (p) => pad.t + (1 - p) * h;

  // area under curve (black advantage shading)
  ctx.beginPath();
  ctx.moveTo(x(data[0].move), y(data[0].blackWinProb));
  data.forEach((d) => ctx.lineTo(x(d.move), y(d.blackWinProb)));
  ctx.lineTo(x(data[data.length - 1].move), y(0.5));
  ctx.lineTo(x(data[0].move), y(0.5));
  ctx.closePath();
  ctx.fillStyle = "rgba(60,60,55,0.12)";
  ctx.fill();

  ctx.beginPath();
  data.forEach((d, i) => {
    if (i === 0) ctx.moveTo(x(d.move), y(d.blackWinProb));
    else ctx.lineTo(x(d.move), y(d.blackWinProb));
  });
  ctx.strokeStyle = "rgba(40,40,36,0.9)";
  ctx.lineWidth = 1.8;
  ctx.lineJoin = "round";
  ctx.stroke();

  data.forEach((d) => {
    ctx.beginPath();
    ctx.arc(x(d.move), y(d.blackWinProb), 2.4, 0, Math.PI * 2);
    ctx.fillStyle = d.blackWinProb >= 0.5 ? "#2b2b27" : "#b5482e";
    ctx.fill();
  });

  ctx.fillStyle = "rgba(120,110,90,0.8)";
  ctx.font = "10px system-ui";
  ctx.textAlign = "left";
  ctx.fillText("黑优", pad.l + 3, pad.t + 10);
  ctx.fillText("白优", pad.l + 3, pad.t + h - 3);
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
  statusPill.textContent = state.status
    .replace("Your turn", "轮到你")
    .replace("AI turn", "AI 行棋")
    .replace("You win", "你赢了！")
    .replace("AI wins", "AI 获胜")
    .replace("Draw", "平局");
  statusPill.className = `status-pill ${state.winner !== null ? "done" : ""}`;
  undoBtn.disabled = !state.canUndo || busy;
}

/* ---------- actions ---------- */

async function refresh() {
  try { render(await request("/api/state")); }
  catch (error) { showToast(error.message); }
}

async function newGame() {
  try {
    setBusy(true);
    render(await request("/api/new", {
      method: "POST",
      body: JSON.stringify({ human: selectedSide, simulations: Number(simSlider.value) }),
    }));
  } catch (error) { showToast(error.message); }
  finally { setBusy(false); render(state); }
}

async function makeMove(row, col) {
  if (busy || !state || state.winner !== null || state.currentPlayer !== state.humanPlayer) return;
  if (state.board[row][col] !== 0) return;
  try {
    setBusy(true);
    render(await request("/api/move", {
      method: "POST",
      body: JSON.stringify({ row, col, simulations: Number(simSlider.value) }),
    }));
  } catch (error) { showToast(error.message); }
  finally { setBusy(false); render(state); }
}

async function undo() {
  if (busy) return;
  try {
    setBusy(true);
    render(await request("/api/undo", { method: "POST", body: "{}" }));
  } catch (error) { showToast(error.message); }
  finally { setBusy(false); render(state); }
}

async function analyze() {
  if (busy || !state || state.winner !== null) return;
  try {
    setBusy(true);
    render(await request("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ simulations: Number(simSlider.value) }),
    }));
    if (overlayMode === "none") setOverlay("search");
  } catch (error) { showToast(error.message); }
  finally { setBusy(false); render(state); }
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

pvToggle.addEventListener("change", () => { if (state) renderBoard(); });
simSlider.addEventListener("input", () => { simValue.textContent = simSlider.value; });
newGameBtn.addEventListener("click", newGame);
undoBtn.addEventListener("click", undo);
analyzeBtn.addEventListener("click", analyze);
window.addEventListener("resize", () => { if (state) renderChart(); });

refresh().then(() => {
  if (!state || state.movesPlayed === 0) newGame();
});
