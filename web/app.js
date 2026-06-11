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
const policyList = document.querySelector("#policyList");
const policyCount = document.querySelector("#policyCount");
const toast = document.querySelector("#toast");
const newGameBtn = document.querySelector("#newGame");
const undoBtn = document.querySelector("#undo");
const analyzeBtn = document.querySelector("#analyze");
const refreshBtn = document.querySelector("#refresh");
const sideButtons = [...document.querySelectorAll(".segment")];

let state = null;
let selectedSide = "black";
let busy = false;

function playerName(value) {
  if (value === 1) return "Black";
  if (value === -1) return "White";
  return "None";
}

function stoneClass(value) {
  if (value === 1) return "black";
  if (value === -1) return "white";
  return "";
}

function showToast(message) {
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { toast.hidden = true; }, 2800);
}

function setBusy(nextBusy) {
  busy = nextBusy;
  newGameBtn.disabled = busy;
  undoBtn.disabled = busy;
  analyzeBtn.disabled = busy;
  refreshBtn.disabled = busy;
  boardEl.classList.toggle("busy", busy);
  if (busy) {
    statusPill.textContent = "AI thinking";
    statusPill.className = "status-pill thinking";
  }
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

function buildBoard(size) {
  boardEl.innerHTML = "";
  colCoords.innerHTML = "";
  rowCoords.innerHTML = "";
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
      cell.setAttribute("aria-label", `Row ${row + 1}, column ${col + 1}`);
      cell.dataset.row = row;
      cell.dataset.col = col;
      cell.addEventListener("click", () => makeMove(row, col));
      boardEl.appendChild(cell);
    }
  }
}

function render(nextState) {
  state = nextState;
  if (!boardEl.children.length) buildBoard(state.size);

  const last = state.lastMove === null ? null : {
    row: Math.floor(state.lastMove / state.size),
    col: state.lastMove % state.size,
  };
  const isHumanTurn = !state.winner && state.currentPlayer === state.humanPlayer;

  [...boardEl.children].forEach((cell) => {
    const row = Number(cell.dataset.row);
    const col = Number(cell.dataset.col);
    const value = state.board[row][col];
    cell.innerHTML = "";
    cell.className = "cell";
    if (value !== 0) {
      const stone = document.createElement("span");
      stone.className = `stone ${stoneClass(value)}`;
      cell.appendChild(stone);
      cell.classList.add("disabled");
    }
    if (!isHumanTurn || value !== 0 || busy) cell.classList.add("disabled");
    if (last && row === last.row && col === last.col) cell.classList.add("last");
  });

  movesEl.textContent = state.movesPlayed;
  turnEl.textContent = playerName(state.currentPlayer);
  simValue.textContent = state.simulations;
  simSlider.value = state.simulations;
  sideLabel.textContent = playerName(state.humanPlayer);
  statusPill.textContent = state.status;
  statusPill.className = `status-pill ${state.winner !== null ? "done" : ""}`;

  // undo only meaningful when moves exist and game not over
  undoBtn.disabled = !state.canUndo || !!state.winner;

  historyCount.textContent = state.history.length;
  historyList.innerHTML = "";
  [...state.history].reverse().forEach((move, index) => {
    const item = document.createElement("li");
    item.innerHTML = `<span>${state.history.length - index}. ${move.player} ${move.row + 1},${move.col + 1}</span><span class="move-source">${move.source}</span>`;
    historyList.appendChild(item);
  });

  policyCount.textContent = state.aiPolicy.length;
  policyList.innerHTML = "";
  state.aiPolicy.forEach((entry, index) => {
    const item = document.createElement("li");
    const percent = Math.round(entry.share * 100);
    item.innerHTML = `<span>${index + 1}. ${entry.row + 1},${entry.col + 1}</span><span>${percent}%${entry.selected ? " ✓" : ""}</span>`;
    policyList.appendChild(item);
  });
}

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
  finally { setBusy(false); }
}

async function makeMove(row, col) {
  if (busy || !state || state.winner !== null || state.currentPlayer !== state.humanPlayer) return;
  if (state.board[row][col] !== 0) return;
  try {
    setBusy(true);
    render(await request("/api/move", {
      method: "POST",
      body: JSON.stringify({ row, col }),
    }));
  } catch (error) { showToast(error.message); }
  finally { setBusy(false); }
}

async function undo() {
  if (busy) return;
  try {
    setBusy(true);
    render(await request("/api/undo", { method: "POST", body: "{}" }));
  } catch (error) { showToast(error.message); }
  finally { setBusy(false); }
}

async function analyze() {
  if (busy || !state || state.winner !== null) return;
  try {
    setBusy(true);
    render(await request("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ simulations: Number(simSlider.value) }),
    }));
  } catch (error) { showToast(error.message); }
  finally { setBusy(false); }
}

sideButtons.forEach((button) => {
  button.addEventListener("click", () => {
    selectedSide = button.dataset.side;
    sideButtons.forEach((item) => item.classList.toggle("active", item === button));
    sideLabel.textContent = selectedSide === "black" ? "Black" : "White";
  });
});

simSlider.addEventListener("input", () => { simValue.textContent = simSlider.value; });

newGameBtn.addEventListener("click", newGame);
undoBtn.addEventListener("click", undo);
analyzeBtn.addEventListener("click", analyze);
refreshBtn.addEventListener("click", refresh);

refresh().then(() => {
  if (!state || state.movesPlayed === 0) newGame();
});
