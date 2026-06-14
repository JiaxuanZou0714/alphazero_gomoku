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
const candidateTitle = document.querySelector("#candidateTitle");
const candBody = document.querySelector("#candBody");
const policyCount = document.querySelector("#policyCount");
const treeSvg = document.querySelector("#treeSvg");
const treeTitle = document.querySelector("#treeTitle");
const treeCount = document.querySelector("#treeCount");
const treeDepth = document.querySelector("#treeDepth");
const treeCopy = document.querySelector("#treeCopy");
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
const loadPercent = document.querySelector("#loadPercent");
const computeOverlay = document.querySelector("#computeOverlay");
const computeTitle = document.querySelector("#computeTitle");
const computeDetail = document.querySelector("#computeDetail");
const recommendationTitle = document.querySelector("#recommendationTitle");
const recommendationMove = document.querySelector("#recommendationMove");
const recommendationMain = document.querySelector("#recommendationMain");
const recSearch = document.querySelector("#recSearch");
const recPolicy = document.querySelector("#recPolicy");
const recWin = document.querySelector("#recWin");
const recommendationWhy = document.querySelector("#recommendationWhy");
const sideButtons = [...document.querySelectorAll(".segment[data-side]")];
const overlayButtons = [...document.querySelectorAll(".segment[data-overlay]")];

let state = null;
let selectedSide = "white";
let overlayMode = "none";
let busy = true;
let cells = [];
let requestId = 0;
const pending = new Map();

const worker = new Worker("./engine.worker.js");

const SIMULATION_OPTIONS = [16, 32, 64, 128, 256, 512];
const playerName = (v) => (v === 1 ? "黑" : v === -1 ? "白" : "-");
const pct = (v, digits = 0) => `${(v * 100).toFixed(digits)}%`;
const moveText = (move) => (move ? `${move.row + 1},${move.col + 1}` : "-");

function simulationIndexFor(value) {
  let bestIndex = 0;
  let bestDistance = Infinity;
  SIMULATION_OPTIONS.forEach((option, index) => {
    const distance = Math.abs(option - value);
    if (distance < bestDistance) {
      bestIndex = index;
      bestDistance = distance;
    }
  });
  return bestIndex;
}

function selectedSimulations() {
  return SIMULATION_OPTIONS[Number(simSlider.value)] || 256;
}

function syncSimulationDisplay(value = selectedSimulations()) {
  simValue.textContent = value;
  simSlider.setAttribute("aria-valuetext", `${value} 次`);
}

function syncOverlayButtons() {
  overlayButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.overlay === overlayMode);
  });
}

function resetAnalysisView() {
  overlayMode = "none";
  syncOverlayButtons();
}

function activeAnalysisContext() {
  if (!state) return null;
  if (state.policySource === "analysis"
    && state.analysis
    && state.analysis.candidates
    && state.analysis.player === state.currentPlayer) {
    return {
      type: "current",
      analysis: state.analysis,
      label: "当前局面建议",
    };
  }
  return null;
}

function showToast(message) {
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toast.hidden = true;
  }, 3200);
}

function busyCopy(label) {
  if (label.includes("加载")) {
    return {
      title: "加载模型",
      detail: "正在初始化浏览器推理后端",
      tone: "loading",
    };
  }
  if (label.includes("提示")) {
    return {
      title: "生成提示",
      detail: "MCTS 正在重新评估当前局面",
      tone: "thinking",
    };
  }
  if (label.includes("思考")) {
    return {
      title: "AI 思考中",
      detail: "搜索候选点并回传胜率评估",
      tone: "thinking",
    };
  }
  if (label.includes("新对局") || label.includes("开局")) {
    return {
      title: "准备新对局",
      detail: "重置棋盘、胜率和搜索树",
      tone: "loading",
    };
  }
  if (label.includes("悔棋")) {
    return {
      title: "回退局面",
      detail: "恢复上一手棋和分析状态",
      tone: "loading",
    };
  }
  return {
    title: label || "处理中",
    detail: "正在更新局面",
    tone: "thinking",
  };
}

function setBusy(nextBusy, label = "") {
  busy = nextBusy;
  [newGameBtn, undoBtn, analyzeBtn, simSlider, ...sideButtons].forEach((el) => {
    el.disabled = busy;
  });
  boardEl.classList.toggle("busy", busy);
  boardEl.setAttribute("aria-busy", String(busy));
  computeOverlay.hidden = !busy;
  computeOverlay.setAttribute("aria-hidden", String(!busy));
  if (busy) {
    const copy = busyCopy(label);
    computeTitle.textContent = copy.title;
    computeDetail.textContent = copy.detail;
    computeOverlay.dataset.tone = copy.tone;
    statusPill.textContent = label || copy.title;
    statusPill.className = `status-pill ${copy.tone}`;
  } else {
    delete computeOverlay.dataset.tone;
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
    const percent = Math.max(0, Math.min(100, pctValue * 100));
    loadBar.style.transform = `scaleX(${Math.max(0.02, percent / 100).toFixed(3)})`;
    loadPercent.textContent = `${Math.round(percent)}%`;
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
  const context = activeAnalysisContext();
  const a = context && context.analysis;
  if (!a || !a.visitMap) return;
  const map = overlayMode === "search" ? a.visitMap
    : overlayMode === "policy" ? a.priorMap : null;
  if (map) {
    const max = Math.max(...map, 1e-9);
    const rgb = overlayMode === "search"
      ? getComputedStyle(document.documentElement).getPropertyValue("--search-heat")
      : getComputedStyle(document.documentElement).getPropertyValue("--policy-heat");
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

  if (overlayMode === "none") {
    const selected = a.candidates && (a.candidates.find((c) => c.selected) || a.candidates[0]);
    if (selected && state.board[selected.row][selected.col] === 0) {
      const marker = document.createElement("span");
      marker.className = "suggestion-marker";
      cells[selected.row * state.size + selected.col].appendChild(marker);
    }
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
  const context = activeAnalysisContext();
  const a = context && context.analysis;
  const bw = blackWinProb(a);
  if (bw === null) {
    winProbEl.textContent = "-";
    evalLabel.textContent = "-";
    evalBlack.style.transform = "scaleY(0.5)";
    winMeterBlack.style.transform = "scaleX(0.5)";
    if (state && state.winner !== null) {
      evalSource.textContent = "对局已结束。";
    } else {
      evalSource.textContent = "点“提示”后更新。";
    }
    statSims.textContent = "-";
    statTime.textContent = "-";
  } else {
    winProbEl.textContent = pct(bw);
    evalLabel.textContent = pct(bw);
    evalBlack.style.transform = `scaleY(${bw.toFixed(3)})`;
    winMeterBlack.style.transform = `scaleX(${bw.toFixed(3)})`;
    evalSource.textContent = `当前局面 · ${playerName(a.player)}方行棋`;
    statSims.textContent = a.simulations;
    statTime.textContent = a.elapsedMs >= 1000
      ? `${(a.elapsedMs / 1000).toFixed(1)}s`
      : `${Math.round(a.elapsedMs)}ms`;
  }
  movesEl.textContent = state.movesPlayed;
  turnEl.textContent = state.winner !== null ? "-" : playerName(state.currentPlayer);

  if (!a || !a.visitMap) {
    overlayNote.textContent = "点“提示”后显示候选。";
  } else {
    overlayNote.textContent = `当前局面 · ${a.simulations} sims`;
  }
}

function renderRecommendation() {
  const context = activeAnalysisContext();
  const a = context && context.analysis;
  const cands = (a && a.candidates) || [];
  const isHumanTurn = state && state.winner === null && state.currentPlayer === state.humanPlayer;

  if (!a || !cands.length) {
    recommendationTitle.textContent = "提示";
    recommendationMove.textContent = "-";
    recSearch.textContent = "-";
    recPolicy.textContent = "-";
    recWin.textContent = "-";

    if (state && state.winner !== null) {
      recommendationMain.textContent = `${playerName(state.winner)}方获胜`;
    } else if (state && state.movesPlayed === 0 && isHumanTurn) {
      recommendationMain.textContent = "你先手。直接下，或点“提示”。";
    } else if (isHumanTurn) {
      recommendationMain.textContent = "轮到你。点“提示”获取建议。";
    } else {
      recommendationMain.textContent = "等待 AI 落子。";
    }
    recommendationWhy.textContent = "";
    return;
  }

  const selected = cands.find((c) => c.selected) || cands[0];
  const selectedWin = selected.q === null ? null : (selected.q + 1) / 2;
  const selectedMove = moveText(selected);

  recommendationTitle.textContent = "建议";
  recommendationMove.textContent = selectedMove;
  recSearch.textContent = pct(selected.share);
  recPolicy.textContent = pct(selected.prior, 1);
  recWin.textContent = selectedWin === null ? "-" : pct(selectedWin);
  recommendationMain.textContent = `下在 ${selectedMove}`;
  recommendationWhy.textContent = `${a.simulations} 次深算`;
}

function renderCandidates() {
  const context = activeAnalysisContext();
  const a = context && context.analysis;
  candBody.innerHTML = "";
  const cands = (a && a.candidates) || [];
  candidateTitle.textContent = "候选点";
  policyCount.textContent = cands.length;
  if (!cands.length) {
    const tr = document.createElement("tr");
    tr.className = "empty-row";
    tr.innerHTML = `<td colspan="5">点“提示”后显示候选。</td>`;
    candBody.appendChild(tr);
    return;
  }
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

function svgEl(name, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
  return el;
}

function renderTree() {
  treeSvg.innerHTML = "";
  const context = activeAnalysisContext();
  const a = context && context.analysis;
  const tree = a && a.tree;
  const nodes = tree && tree.nodes ? tree.nodes : [];
  const edges = tree && tree.edges ? tree.edges : [];
  treeTitle.textContent = "MCTS 树";
  treeCopy.textContent = context
    ? "当前局面的搜索摘要。"
    : "点“提示”后显示。";
  treeCount.textContent = Math.max(0, nodes.length - 1);
  treeDepth.textContent = nodes.length ? "深度 -" : "等待建议";
  treeSvg.setAttribute("aria-label", nodes.length
    ? `${treeTitle.textContent}，显示 ${Math.max(0, nodes.length - 1)} 个已展开节点`
    : "MCTS 搜索树，尚无搜索数据");

  if (!nodes.length) {
    const text = svgEl("text", {
      x: 420,
      y: 160,
      class: "tree-empty",
      "text-anchor": "middle",
    });
    const lines = ["还没有搜索树", "点“提示”后显示"];
    lines.forEach((line, index) => {
      const tspan = svgEl("tspan", {
        x: 420,
        dy: index === 0 ? 0 : 20,
      });
      tspan.textContent = line;
      text.appendChild(tspan);
    });
    treeSvg.appendChild(text);
    return;
  }

  const width = 840;
  const height = 380;
  const maxDepth = Math.max(...nodes.map((node) => node.depth));
  const principalDepth = nodes.filter((node) => node.principal).length - 1;
  treeDepth.textContent = `主线 ${Math.max(0, principalDepth)} 层 / 摘要 ${maxDepth} 层`;
  const depthY = [34, 92, 150, 208, 266, 322, 356];
  const depthLabels = ["当前", "候选落点", "回应", "再展开", "深层", "尾部", "末端"];
  const byDepth = new Map();
  nodes.forEach((node) => {
    if (!byDepth.has(node.depth)) byDepth.set(node.depth, []);
    byDepth.get(node.depth).push(node);
  });

  const positions = new Map();
  const root = nodes.find((node) => node.depth === 0);
  if (root) positions.set(root.id, { x: width / 2, y: depthY[0] });
  const first = byDepth.get(1) || [];
  first.forEach((node, index) => {
    const x = 74 + ((index + 1) / (first.length + 1)) * (width - 148);
    positions.set(node.id, { x, y: depthY[1] });
  });

  for (let depth = 2; depth <= maxDepth; depth += 1) {
    const byParent = new Map();
    (byDepth.get(depth) || []).forEach((node) => {
      if (!byParent.has(node.parentId)) byParent.set(node.parentId, []);
      byParent.get(node.parentId).push(node);
    });
    for (const [parentId, group] of byParent.entries()) {
      const parent = positions.get(parentId);
      if (!parent) continue;
      const step = depth === 2 ? 38 : depth === 3 ? 26 : depth === 4 ? 18 : 13;
      group.forEach((node, index) => {
        const spread = group.length === 1 ? 0 : (index - (group.length - 1) / 2) * step;
        positions.set(node.id, {
          x: Math.max(20, Math.min(width - 20, parent.x + spread)),
          y: depthY[depth] || (height - 24),
        });
      });
    }
  }

  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const backdrop = svgEl("g", { class: "tree-backdrop" });
  depthY.slice(1, Math.min(maxDepth + 1, depthY.length)).forEach((y) => {
    backdrop.appendChild(svgEl("line", {
      x1: 26,
      y1: y.toFixed(1),
      x2: (width - 26).toFixed(1),
      y2: y.toFixed(1),
      class: "tree-depth-line",
    }));
    const label = svgEl("text", {
      x: 34,
      y: (y - 8).toFixed(1),
      class: "tree-depth-label",
    });
    label.textContent = depthLabels[depthY.indexOf(y)] || `第 ${depthY.indexOf(y)} 层`;
    backdrop.appendChild(label);
  });
  treeSvg.appendChild(backdrop);

  const curvePoint = (a, b, t) => {
    const midY = a.y + (b.y - a.y) * 0.48;
    const c1 = { x: a.x, y: midY };
    const c2 = { x: b.x, y: midY };
    const mt = 1 - t;
    return {
      x: mt ** 3 * a.x + 3 * mt ** 2 * t * c1.x + 3 * mt * t ** 2 * c2.x + t ** 3 * b.x,
      y: mt ** 3 * a.y + 3 * mt ** 2 * t * c1.y + 3 * mt * t ** 2 * c2.y + t ** 3 * b.y,
    };
  };

  const edgeLayer = svgEl("g", { class: "tree-edges" });
  const pulseLayer = svgEl("g", { class: "tree-pulses" });
  edges.forEach((edge) => {
    const a = positions.get(edge.from);
    const b = positions.get(edge.to);
    if (!a || !b) return;
    const target = nodeById.get(edge.to);
    const midY = a.y + (b.y - a.y) * 0.48;
    edgeLayer.appendChild(svgEl("path", {
      d: `M ${a.x.toFixed(1)} ${a.y.toFixed(1)} C ${a.x.toFixed(1)} ${midY.toFixed(1)}, ${b.x.toFixed(1)} ${midY.toFixed(1)}, ${b.x.toFixed(1)} ${b.y.toFixed(1)}`,
      class: `tree-edge depth-${target ? target.depth : 1} ${edge.principal ? "best" : edge.share > 0.45 ? "strong" : ""}`,
      opacity: (0.24 + Math.min(0.55, edge.share * 0.7)).toFixed(2),
      "stroke-width": (0.65 + Math.sqrt(Math.max(0.02, edge.share)) * (edge.principal ? 5.4 : 3.6)).toFixed(2),
    }));

    const pulseCount = edge.principal ? 5 : edge.share > 0.25 ? 3 : 2;
    for (let i = 0; i < pulseCount; i += 1) {
      const p = curvePoint(a, b, (i + 1) / (pulseCount + 1));
      const pulse = svgEl("circle", {
        cx: p.x.toFixed(1),
        cy: p.y.toFixed(1),
        r: (edge.principal ? 1.9 : 1.15).toFixed(1),
        class: `tree-pulse ${edge.principal ? "best" : ""}`,
      });
      pulse.setAttribute("style", `animation-delay:${(-(i * 0.22 + (target ? target.depth : 1) * 0.12)).toFixed(2)}s`);
      pulseLayer.appendChild(pulse);
    }
  });
  treeSvg.appendChild(edgeLayer);
  treeSvg.appendChild(pulseLayer);

  const nodeLayer = svgEl("g", { class: "tree-nodes" });
  nodes.forEach((node) => {
    const p = positions.get(node.id);
    if (!p) return;
    const isRoot = node.depth === 0;
    const r = isRoot ? 10 : node.depth === 1
      ? 6 + Math.sqrt(Math.max(0.01, node.share)) * 18
      : node.depth === 2
        ? 4.5 + Math.sqrt(Math.max(0.01, node.branchShare || node.share)) * 7
        : node.depth === 3
          ? 2.9 + Math.sqrt(Math.max(0.01, node.branchShare || node.share)) * 4
          : 2.4 + Math.sqrt(Math.max(0.01, node.branchShare || node.share)) * 3;
    const group = svgEl("g", {
      class: `tree-node depth-${node.depth} ${isRoot ? "root" : node.mover === 1 ? "black" : "white"} ${node.principal && !isRoot ? "best" : ""}`,
    });
    const circle = svgEl("circle", {
      cx: p.x.toFixed(1),
      cy: p.y.toFixed(1),
      r: Math.min(node.depth >= 4 ? 4.2 : node.depth === 3 ? 5.6 : node.depth === 2 ? 8.5 : 18, r).toFixed(1),
    });
    const title = svgEl("title");
    const move = isRoot ? "当前局面" : `${node.mover === 1 ? "黑" : "白"} ${node.row + 1},${node.col + 1}`;
    const win = node.winProb === null ? "-" : pct(node.winProb);
    title.textContent = `${move} · 模拟 ${node.visits} · 分支 ${pct(node.branchShare || 0)} · 当前方胜率 ${win}`;
    circle.appendChild(title);
    group.appendChild(circle);

    const label = svgEl("text", {
      x: p.x.toFixed(1),
      y: (p.y + Math.min(node.depth >= 4 ? 4.2 : node.depth === 3 ? 5.6 : node.depth === 2 ? 8.5 : 18, r) + 13).toFixed(1),
      class: `tree-label depth-${node.depth}`,
      "text-anchor": "middle",
    });
    label.textContent = isRoot ? "当前" : `${node.row + 1},${node.col + 1}`;
    group.appendChild(label);
    nodeLayer.appendChild(group);
  });
  treeSvg.appendChild(nodeLayer);
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
  renderRecommendation();
  renderCandidates();
  renderTree();
  renderHistory();
  renderChart();

  const simIndex = simulationIndexFor(state.simulations);
  simSlider.value = simIndex;
  syncSimulationDisplay(SIMULATION_OPTIONS[simIndex]);
  sideLabel.textContent = state.humanPlayer === 1 ? "执黑" : "执白";
  statusPill.textContent = state.status;
  statusPill.className = `status-pill ${state.winner !== null ? "done" : ""}`;
  undoBtn.disabled = !state.canUndo || busy;
}

async function newGame() {
  try {
    setBusy(true, "新对局");
    const nextState = await callWorker("newGame", {
      human: selectedSide,
      simulations: selectedSimulations(),
    });
    resetAnalysisView();
    render(nextState);
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
    const nextState = await callWorker("move", {
      row,
      col,
      simulations: selectedSimulations(),
    });
    resetAnalysisView();
    render(nextState);
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
    const nextState = await callWorker("undo");
    resetAnalysisView();
    render(nextState);
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
    setBusy(true, "生成提示");
    render(await callWorker("analyze", { simulations: selectedSimulations() }));
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
    if (state) render(state);
  }
}

function setOverlay(mode) {
  overlayMode = mode;
  syncOverlayButtons();
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
  syncSimulationDisplay();
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
    loadBar.style.transform = "scaleX(1)";
    loadPercent.textContent = "100%";
    loadText.textContent = "模型就绪，正在自动开局";
    setBusy(false);
    await newGame();
    loadLine.classList.add("ready");
  } catch (error) {
    statusPill.textContent = "加载失败";
    statusPill.className = "status-pill done";
    showToast(error.message);
  }
}

boot();
