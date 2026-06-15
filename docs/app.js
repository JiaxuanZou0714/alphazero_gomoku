const boardEl = document.querySelector("#board");
const colCoords = document.querySelector("#colCoords");
const rowCoords = document.querySelector("#rowCoords");
const statusPill = document.querySelector("#statusPill");
const simSlider = document.querySelector("#simSlider");
const simValue = document.querySelector("#simValue");
const sideLabel = document.querySelector("#sideLabel");
const modelLabel = document.querySelector("#modelLabel");
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
const networkDiagram = document.querySelector("#networkDiagram");
const pvToggle = document.querySelector("#pvToggle");
const overlayNote = document.querySelector("#overlayNote");
const evalBlack = document.querySelector("#evalBlack");
const evalLabel = document.querySelector("#evalLabel");
const winMeterBlack = document.querySelector("#winMeterBlack");
const winProbEl = document.querySelector("#winProb");
const evalSource = document.querySelector("#evalSource");
const statSims = document.querySelector("#statSims");
const statTime = document.querySelector("#statTime");
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
const recommendationPanel = document.querySelector(".recommendation");
const recSearch = document.querySelector("#recSearch");
const recPolicy = document.querySelector("#recPolicy");
const recWin = document.querySelector("#recWin");
const recommendationWhy = document.querySelector("#recommendationWhy");
const analysisDetails = document.querySelector(".analysis-details");
const networkDetails = document.querySelector(".network-details");
const sideInputs = [...document.querySelectorAll("input[name='side']")];
const modelInputs = [...document.querySelectorAll("input[name='model']")];
const overlayInputs = [...document.querySelectorAll("input[name='overlay']")];

const APP_VERSION = "2026-06-15-multi-model";
let state = null;
let selectedSide = "white";
let selectedModelId = "v3";
let overlayMode = "none";
let busy = true;
let cells = [];
let requestId = 0;
let evalRequestId = 0;
let pendingEvalKey = null;
let d3Tree = null;
let d3TreeError = false;
let d3TreePromise = null;
let renderedNetworkKey = "";
let modelCatalog = null;
const pending = new Map();

const SEARCH_HEAT = getComputedStyle(document.documentElement)
  .getPropertyValue("--search-heat").trim().replaceAll(" ", ",");
const POLICY_HEAT = getComputedStyle(document.documentElement)
  .getPropertyValue("--policy-heat").trim().replaceAll(" ", ",");

const worker = new Worker(`./engine.worker.js?v=${APP_VERSION}`);

const SIMULATION_OPTIONS = [16, 32, 64, 128, 256, 512, 1024, 2048];
function networkDiagramConfig(model) {
  const cfg = model && model.config ? model.config : {};
  return {
    channels: cfg.channels || 128,
    blocks: cfg.residual_blocks || 8,
    policyChannels: cfg.policy_channels || 12,
    valueHidden: cfg.value_hidden || 384,
  };
}
const playerName = (v) => (v === 1 ? "黑" : v === -1 ? "白" : "-");
const pct = (v, digits = 0) => `${(v * 100).toFixed(digits)}%`;
const moveText = (move) => (move ? `${move.row + 1},${move.col + 1}` : "-");

function stateKey(s) {
  if (!s) return "";
  const boardKey = Array.isArray(s.board)
    ? s.board.map((row) => row.join(",")).join(",")
    : "";
  return [
    s.movesPlayed,
    s.currentPlayer,
    s.lastMove === null ? "none" : s.lastMove,
    s.winner === null ? "none" : s.winner,
    boardKey,
  ].join(":");
}

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
  return SIMULATION_OPTIONS[Number(simSlider.value)] || 512;
}

function syncSimulationDisplay(value = selectedSimulations()) {
  simValue.textContent = value;
  simSlider.setAttribute("aria-valuetext", `${value} 次`);
}

function syncOverlayButtons() {
  overlayInputs.forEach((input) => {
    input.checked = input.value === overlayMode;
  });
}

function modelDisplayName(modelId = selectedModelId) {
  if (state && state.model && state.model.id === modelId) return state.model.label || modelId;
  const input = modelInputs.find((item) => item.value === modelId);
  return input ? input.closest(".segment").querySelector("span").textContent : modelId;
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
      detail: "MCTS 正在比较候选落点",
      tone: "thinking",
    };
  }
  if (label.includes("思考")) {
    return {
      title: "AI 思考中",
      detail: "搜索树正在扩展",
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
  [newGameBtn, undoBtn, analyzeBtn, simSlider, ...sideInputs, ...modelInputs].forEach((el) => {
    el.disabled = busy;
  });
  newGameBtn.textContent = busy && label.includes("新对局") ? "开局中" : "新对局";
  undoBtn.textContent = busy && label.includes("悔棋") ? "回退中" : "悔棋";
  analyzeBtn.textContent = busy && label.includes("提示") ? "深算中" : "提示";
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
  if (msg.type === "info") {
    showToast(msg.payload.message);
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
  const message = event.message || "Worker 运行失败";
  showToast(message);
  // Reject every in-flight request so callers' awaits don't hang forever.
  pending.forEach(({ reject }) => reject(new Error(message)));
  pending.clear();
  setBusy(false);
});

function buildBoard(size) {
  boardEl.replaceChildren();
  colCoords.replaceChildren();
  rowCoords.replaceChildren();
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
    cell.replaceChildren();
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
    const rgb = overlayMode === "search" ? SEARCH_HEAT : POLICY_HEAT;
    map.forEach((v, idx) => {
      if (v < 5e-4) return;
      const t = Math.sqrt(v / max);
      const heat = document.createElement("span");
      heat.className = "heat";
      heat.style.background = `rgba(${rgb},${(0.1 + 0.55 * t).toFixed(3)})`;
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

function renderEval() {
  const evaluation = state && state.evaluation;
  const bw = evaluation ? evaluation.blackWinProb : null;
  if (bw === null) {
    const evalPending = state && state.evaluationPending;
    winProbEl.textContent = evalPending ? "..." : "-";
    evalLabel.textContent = evalPending ? "..." : "-";
    evalBlack.style.transform = "scaleY(0.5)";
    winMeterBlack.style.transform = "scaleX(0.5)";
    if (state && state.winner !== null) {
      evalSource.textContent = "对局已结束。";
    } else if (evalPending) {
      evalSource.textContent = "模型实时评估中。";
    } else {
      evalSource.textContent = "模型就绪后自动更新。";
    }
    statSims.textContent = evalPending ? "模型" : "-";
    statTime.textContent = "-";
  } else {
    winProbEl.textContent = pct(bw);
    evalLabel.textContent = pct(bw);
    evalBlack.style.transform = `scaleY(${bw.toFixed(3)})`;
    winMeterBlack.style.transform = `scaleX(${bw.toFixed(3)})`;
    if (evaluation.source === "terminal") {
      evalSource.textContent = "终局结果。";
      statSims.textContent = "终局";
    } else if (evaluation.source === "mcts") {
      evalSource.textContent = "黑方胜率来自本次 MCTS 搜索。";
      statSims.textContent = "MCTS";
    } else {
      evalSource.textContent = "黑方胜率来自神经网络单次评估。";
      statSims.textContent = "模型";
    }
    statTime.textContent = evaluation.elapsedMs >= 1000
      ? `${(evaluation.elapsedMs / 1000).toFixed(1)}s`
      : `${Math.round(evaluation.elapsedMs)}ms`;
  }
  movesEl.textContent = state.movesPlayed;
  turnEl.textContent = state.winner !== null ? "-" : playerName(state.currentPlayer);

  const context = activeAnalysisContext();
  const a = context && context.analysis;
  if (!a || !a.visitMap) {
    overlayNote.textContent = "点“提示”后显示候选。";
  } else {
    overlayNote.textContent = `当前局面 · 本次 ${a.requestedSimulations || a.simulations} / 累计访问 ${a.simulations}`;
  }
}

function renderRecommendation() {
  const context = activeAnalysisContext();
  const a = context && context.analysis;
  const cands = (a && a.candidates) || [];
  const isHumanTurn = state && state.winner === null && state.currentPlayer === state.humanPlayer;

  if (!a || !cands.length) {
    recommendationPanel.classList.remove("has-analysis");
    recommendationTitle.textContent = "建议";
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

  recommendationPanel.classList.add("has-analysis");
  const selected = cands.find((c) => c.selected) || cands[0];
  const selectedWin = selected.q === null ? null : (selected.q + 1) / 2;
  const selectedMove = moveText(selected);

  recommendationTitle.textContent = "建议";
  recommendationMove.textContent = selectedMove;
  recSearch.textContent = pct(selected.share);
  recPolicy.textContent = pct(selected.prior, 1);
  recWin.textContent = selectedWin === null ? "-" : pct(selectedWin);
  recommendationMain.textContent = `下在 ${selectedMove}`;
  recommendationWhy.textContent = `本次 ${a.requestedSimulations || a.simulations} 次模拟。访问最多的分支就是 MCTS 最后最认可的选择。`;
}

function renderCandidates() {
  const context = activeAnalysisContext();
  const a = context && context.analysis;
  candBody.replaceChildren();
  const cands = (a && a.candidates) || [];
  candidateTitle.textContent = "候选点";
  policyCount.textContent = cands.length;
  if (!cands.length) {
    const tr = document.createElement("tr");
    tr.className = "empty-row";
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = "点“提示”后显示候选。";
    tr.appendChild(td);
    candBody.appendChild(tr);
    return;
  }
  cands.forEach((c) => {
    const tr = document.createElement("tr");
    if (c.selected) tr.classList.add("best");
    const win = c.q === null ? null : (c.q + 1) / 2;
    const shareWidth = Math.max(2, Math.min(100, c.share * 100)).toFixed(1);
    const idx = c.row * state.size + c.col;
    const labelParts = [
      c.selected ? "建议点" : "候选点",
      `${c.row + 1},${c.col + 1}`,
      `访问 ${c.visits}`,
      `占比 ${pct(c.share)}`,
      `模型倾向 ${pct(c.prior, 1)}`,
    ];
    if (win !== null) labelParts.push(`胜率 ${pct(win)}`);
    tr.tabIndex = 0;
    tr.setAttribute("aria-label", labelParts.join("，"));
    const moveCell = document.createElement("td");
    moveCell.textContent = `${c.selected ? "✓ " : ""}${c.row + 1},${c.col + 1}`;
    tr.appendChild(moveCell);

    const visitsCell = document.createElement("td");
    visitsCell.textContent = c.visits;
    tr.appendChild(visitsCell);

    const shareCell = document.createElement("td");
    const share = document.createElement("span");
    const shareText = document.createElement("span");
    share.className = "share-cell";
    share.style.setProperty("--share", `${shareWidth}%`);
    shareText.textContent = pct(c.share);
    share.appendChild(shareText);
    shareCell.appendChild(share);
    tr.appendChild(shareCell);

    const priorCell = document.createElement("td");
    priorCell.textContent = pct(c.prior, 1);
    tr.appendChild(priorCell);

    const winCell = document.createElement("td");
    if (win !== null) winCell.className = win >= 0.5 ? "q-good" : "q-bad";
    winCell.textContent = win === null ? "-" : pct(win);
    tr.appendChild(winCell);

    const mark = () => cells[idx].classList.add("hl");
    const unmark = () => cells[idx].classList.remove("hl");
    tr.addEventListener("mouseenter", mark);
    tr.addEventListener("mouseleave", unmark);
    tr.addEventListener("focus", mark);
    tr.addEventListener("blur", unmark);
    candBody.appendChild(tr);
  });
}

function svgEl(name, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
  return el;
}

function loadTreeLibrary() {
  if (!d3TreePromise) {
    d3TreePromise = new Promise((resolve, reject) => {
      if (window.d3) {
        resolve(window.d3);
        return;
      }
      const script = document.createElement("script");
      script.src = "./assets/vendor/d3/d3.min.js";
      script.async = true;
      script.onload = () => (window.d3 ? resolve(window.d3) : reject(new Error("D3 未暴露全局对象")));
      script.onerror = () => reject(new Error("D3 加载失败"));
      document.head.appendChild(script);
    });
  }
  return d3TreePromise;
}

function renderTreeMessage(title, detail) {
  treeSvg.replaceChildren();
  treeSvg.setAttribute("viewBox", "0 0 920 460");
  const group = svgEl("g", { class: "tree-message", transform: "translate(460 220)" });
  const titleText = svgEl("text", {
    class: "tree-empty tree-empty-title",
    "text-anchor": "middle",
  });
  titleText.textContent = title;
  const detailText = svgEl("text", {
    class: "tree-empty",
    y: 24,
    "text-anchor": "middle",
  });
  detailText.textContent = detail;
  group.append(titleText, detailText);
  treeSvg.appendChild(group);
}

function buildTreeData(nodes) {
  const byId = new Map(nodes.map((node) => [node.id, { ...node, children: [] }]));
  let root = null;
  nodes.forEach((node) => {
    const next = byId.get(node.id);
    if (node.parentId === null) {
      root = next;
      return;
    }
    const parent = byId.get(node.parentId);
    if (parent) parent.children.push(next);
  });
  return root;
}

function treeNodeRadius(node) {
  if (node.depth === 0) return 10;
  const share = Math.max(0.01, node.branchShare || node.share || 0.01);
  if (node.depth === 1) return 7 + Math.sqrt(share) * 13;
  if (node.depth === 2) return 5 + Math.sqrt(share) * 7;
  if (node.depth === 3) return 4 + Math.sqrt(share) * 4;
  return 3.2 + Math.sqrt(share) * 2.8;
}

function treeNodeLabel(node) {
  if (node.depth === 0) return "当前";
  if (node.depth <= 1 || node.principal) return `${node.row + 1},${node.col + 1}`;
  return "";
}

function treeNodeTitle(node) {
  if (node.depth === 0) return "当前局面";
  const move = `${node.mover === 1 ? "黑" : "白"} ${node.row + 1},${node.col + 1}`;
  const win = node.winProb === null ? "-" : pct(node.winProb);
  return `${move} · 访问 ${node.visits} · 分支 ${pct(node.branchShare || 0)} · 当前方胜率 ${win}`;
}

function renderTreeWithD3(nodes, edges) {
  const rootData = buildTreeData(nodes);
  if (!rootData) {
    renderTreeMessage("搜索树数据不完整", "候选点仍然可以在下方表格查看");
    return;
  }

  const width = 920;
  const leafIds = new Set(nodes.map((node) => node.parentId).filter((id) => id !== null));
  const leafCount = Math.max(1, nodes.filter((node) => !leafIds.has(node.id)).length);
  const height = Math.max(420, Math.min(720, 150 + leafCount * 42));
  const margin = { top: 42, right: 46, bottom: 42, left: 58 };
  const edgeByTarget = new Map(edges.map((edge) => [edge.to, edge]));
  const root = d3Tree.hierarchy(rootData);
  root.sort((a, b) => Number(b.data.principal) - Number(a.data.principal)
    || (b.data.visits || 0) - (a.data.visits || 0)
    || (a.data.rank || 0) - (b.data.rank || 0));

  d3Tree.tree()
    .size([height - margin.top - margin.bottom, width - margin.left - margin.right])
    .separation((a, b) => (a.parent === b.parent ? 1 : 1.35) + Math.abs(a.depth - b.depth) * 0.08)(root);

  root.each((node) => {
    node.x += margin.top;
    node.y += margin.left;
  });

  const svg = d3Tree.select(treeSvg);
  svg.selectAll("*").remove();
  svg.attr("viewBox", `0 0 ${width} ${height}`);

  const depthLabels = ["当前", "候选", "回应", "再展开", "深层", "尾部", "末端"];
  const rows = Array.from(d3Tree.group(root.descendants(), (node) => node.depth), ([depth, group]) => ({
    depth,
    y: group[0].y,
  })).sort((a, b) => a.depth - b.depth);

  const backdrop = svg.append("g").attr("class", "tree-backdrop");
  backdrop.selectAll("line")
    .data(rows)
    .join("line")
    .attr("class", "tree-depth-line")
    .attr("x1", (row) => row.y)
    .attr("x2", (row) => row.y)
    .attr("y1", 26)
    .attr("y2", height - 26);

  backdrop.selectAll("text")
    .data(rows)
    .join("text")
    .attr("class", "tree-depth-label")
    .attr("x", (row) => row.y)
    .attr("y", 24)
    .attr("text-anchor", "middle")
    .text((row) => depthLabels[row.depth] || `第 ${row.depth} 层`);

  const link = d3Tree.linkHorizontal()
    .x((node) => node.y)
    .y((node) => node.x);

  svg.append("g")
    .attr("class", "tree-edges")
    .selectAll("path")
    .data(root.links())
    .join("path")
    .attr("class", (linkData) => {
      const edge = edgeByTarget.get(linkData.target.data.id);
      const strong = edge && edge.share > 0.45;
      return `tree-edge depth-${linkData.target.depth} ${edge && edge.principal ? "best" : strong ? "strong" : ""}`;
    })
    .attr("d", link)
    .attr("opacity", (linkData) => {
      const edge = edgeByTarget.get(linkData.target.data.id);
      return edge ? (0.25 + Math.min(0.55, edge.share * 0.72)).toFixed(2) : 0.28;
    })
    .attr("stroke-width", (linkData) => {
      const edge = edgeByTarget.get(linkData.target.data.id);
      const share = edge ? edge.share : 0.06;
      return (0.85 + Math.sqrt(Math.max(0.02, share)) * (edge && edge.principal ? 5.8 : 3.4)).toFixed(2);
    });

  const node = svg.append("g")
    .attr("class", "tree-nodes")
    .selectAll("g")
    .data(root.descendants())
    .join("g")
    .attr("class", (item) => {
      const data = item.data;
      const color = item.depth === 0 ? "root" : data.mover === 1 ? "black" : "white";
      return `tree-node depth-${item.depth} ${color} ${data.principal && item.depth > 0 ? "best" : ""}`;
    })
    .attr("transform", (item) => `translate(${item.y.toFixed(1)},${item.x.toFixed(1)})`);

  node.append("circle")
    .attr("r", (item) => Math.min(item.depth >= 4 ? 5 : item.depth === 3 ? 6.2 : item.depth === 2 ? 9.5 : 22, treeNodeRadius(item.data)).toFixed(1));

  node.append("title")
    .text((item) => treeNodeTitle(item.data));

  node.append("text")
    .attr("text-anchor", "middle")
    .attr("y", (item) => Math.min(item.depth >= 4 ? 5 : item.depth === 3 ? 6.2 : item.depth === 2 ? 9.5 : 22, treeNodeRadius(item.data)) + 15)
    .text((item) => treeNodeLabel(item.data));
}

function renderTree() {
  treeSvg.replaceChildren();
  const context = activeAnalysisContext();
  const a = context && context.analysis;
  const tree = a && a.tree;
  const nodes = tree && tree.nodes ? tree.nodes : [];
  const edges = tree && tree.edges ? tree.edges : [];
  treeTitle.textContent = "MCTS 搜索树";
  treeCopy.textContent = context
    ? "这里只显示访问最多的一小部分搜索树。越往右代表后续层数越深；线越粗，访问越多。"
    : "点“提示”后显示。";
  treeCount.textContent = Math.max(0, nodes.length - 1);
  treeDepth.textContent = nodes.length ? "深度 -" : "等待建议";
  treeSvg.setAttribute("aria-label", nodes.length
    ? `${treeTitle.textContent}，显示 ${Math.max(0, nodes.length - 1)} 个已展开节点`
    : "MCTS 搜索树，尚无搜索数据");

  if (!nodes.length) {
    renderTreeMessage("还没有搜索树", "点“提示”后显示");
    return;
  }

  const maxDepth = Math.max(...nodes.map((node) => node.depth));
  const principalDepth = nodes.filter((node) => node.principal).length - 1;
  treeDepth.textContent = `主线 ${Math.max(0, principalDepth)} 层 / 摘要 ${maxDepth} 层`;

  if (!analysisDetails.open && !d3Tree) {
    renderTreeMessage("搜索树已就绪", "展开“分析详情”后加载树布局");
    return;
  }

  if (d3TreeError) {
    renderTreeMessage("树图加载失败", "D3 暂时不可用，候选表仍可用");
    return;
  }

  if (!d3Tree) {
    renderTreeMessage("加载树布局", "首次显示会载入 D3");
    loadTreeLibrary()
      .then((module) => {
        d3Tree = module;
        if (state) renderTree();
      })
      .catch(() => {
        d3TreeError = true;
        renderTreeMessage("树图加载失败", "D3 暂时不可用，候选表仍可用");
      });
    return;
  }

  renderTreeWithD3(nodes, edges);
}

function renderHistory() {
  historyCount.textContent = state.movesPlayed;
  historyList.replaceChildren();
  [...state.history].reverse().forEach((move, index) => {
    const item = document.createElement("li");
    const n = state.movesPlayed - index;
    const moveLabel = document.createElement("span");
    moveLabel.textContent = `${n}. ${move.player === "black" ? "黑" : "白"} ${move.row + 1},${move.col + 1}`;
    const source = document.createElement("span");
    source.className = "move-source";
    source.textContent = move.source === "ai" ? "AI" : "人类";
    item.append(moveLabel, source);
    historyList.appendChild(item);
  });
}

function applyEvaluation(evaluation) {
  if (!state || evaluation.key !== stateKey(state)) return;
  if (state.policySource === "analysis" && evaluation.source === "model") return;
  state.evaluation = evaluation;
  state.evaluationPending = false;
  pendingEvalKey = null;
  renderEval();
}

function requestLiveEvaluation() {
  if (!state) return;
  const key = stateKey(state);
  if (state.evaluation && state.evaluation.key === key) return;
  if (state.policySource === "analysis") return;
  if (pendingEvalKey === key) return;

  pendingEvalKey = key;
  state.evaluationPending = true;
  renderEval();
  const id = ++evalRequestId;
  callWorker("evaluate")
    .then((evaluation) => {
      if (id !== evalRequestId && evaluation.key !== stateKey(state)) return;
      applyEvaluation(evaluation);
    })
    .catch((error) => {
      if (pendingEvalKey === key && state && stateKey(state) === key) {
        pendingEvalKey = null;
        state.evaluationPending = false;
        renderEval();
      }
      showToast(error.message || "局面评估失败");
    });
}

function render(nextState) {
  state = nextState;
  if (state.evaluation) {
    state.evaluationPending = false;
    pendingEvalKey = null;
  }
  if (!cells.length) buildBoard(state.size);

  renderBoard();
  renderEval();
  renderRecommendation();
  renderCandidates();
  renderTree();
  renderHistory();

  const simIndex = simulationIndexFor(state.simulations);
  simSlider.value = simIndex;
  syncSimulationDisplay(SIMULATION_OPTIONS[simIndex]);
  sideLabel.textContent = state.humanPlayer === 1 ? "执黑" : "执白";
  selectedModelId = state.model && state.model.id ? state.model.id : selectedModelId;
  modelInputs.forEach((input) => {
    input.checked = input.value === selectedModelId;
  });
  modelLabel.textContent = state.model && state.model.label ? state.model.label : modelDisplayName();
  statusPill.textContent = state.status;
  statusPill.className = `status-pill ${state.winner !== null ? "done" : ""}`;
  undoBtn.disabled = !state.canUndo || busy;
  if (networkDetails.open) initNetworkDiagram();
  requestLiveEvaluation();
}

// Shared busy/try/catch envelope for a worker action that returns a new state.
// On success `render(next)` already painted, so we only re-render in the error
// path to avoid a redundant d3 tree relayout.
async function runAction(label, type, payload = {}, { after } = {}) {
  let ok = false;
  try {
    setBusy(true, label);
    const nextState = await callWorker(type, payload);
    resetAnalysisView();
    if (after) after();
    render(nextState);
    ok = true;
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
    if (!ok && state) render(state);
  }
}

async function switchModel(modelId) {
  if (busy || modelId === selectedModelId) return;
  selectedModelId = modelId;
  resetAnalysisView();
  renderedNetworkKey = "";
  if (networkDiagram) {
    networkDiagram.textContent = "网络结构图加载中";
    networkDiagram.classList.remove("is-rendered", "diagram-error");
  }
  let ok = false;
  try {
    setBusy(true, `加载 ${modelDisplayName(modelId)}`);
    loadLine.classList.remove("ready");
    loadBar.style.transform = "scaleX(0.02)";
    loadPercent.textContent = "0%";
    loadText.textContent = "准备加载";
    await callWorker("init", { modelId });
    loadBar.style.transform = "scaleX(1)";
    loadPercent.textContent = "100%";
    loadText.textContent = `${modelDisplayName(modelId)} 就绪，正在开局`;
    setBusy(false);
    await newGame();
    loadLine.classList.add("ready");
    ok = true;
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
    if (!ok && state) render(state);
  }
}

async function newGame() {
  await runAction("新对局", "newGame", {
    human: selectedSide,
    simulations: selectedSimulations(),
  });
}

async function makeMove(row, col) {
  if (busy || !state || state.winner !== null || state.currentPlayer !== state.humanPlayer) return;
  if (state.board[row][col] !== 0) return;
  await runAction("AI 思考中", "move", {
    row,
    col,
    simulations: selectedSimulations(),
  });
}

async function undo() {
  if (busy) return;
  await runAction("悔棋", "undo");
}

async function analyze() {
  if (busy || !state || state.winner !== null) return;
  await runAction("生成提示", "analyze", { simulations: selectedSimulations() }, {
    after: () => { analysisDetails.open = true; },
  });
}

function networkNode(x, y, w, h, title, subtitle) {
  const g = svgEl("g", { class: "netnode", transform: `translate(${x} ${y})` });
  g.appendChild(svgEl("rect", { x: 0, y: 0, width: w, height: h, rx: 8, ry: 8 }));
  const titleText = svgEl("text", {
    class: "netnode-title",
    x: w / 2,
    y: h / 2 - 6,
    "text-anchor": "middle",
  });
  titleText.textContent = title;
  const subText = svgEl("text", {
    class: "netnode-sub",
    x: w / 2,
    y: h / 2 + 12,
    "text-anchor": "middle",
  });
  subText.textContent = subtitle;
  g.append(titleText, subText);
  return g;
}

function networkEdge(x1, y1, x2, y2) {
  return svgEl("line", {
    class: "netedge",
    x1,
    y1,
    x2,
    y2,
    "marker-end": "url(#netArrow)",
  });
}

function initNetworkDiagram() {
  if (!networkDiagram) return;
  const model = state && state.model ? state.model : null;
  const key = model ? `${model.id}:${model.iteration || ""}` : "unknown";
  if (renderedNetworkKey === key && networkDiagram.classList.contains("is-rendered")) return;
  const { channels, blocks, policyChannels, valueHidden } = networkDiagramConfig(model);

  const W = 760;
  const H = 240;
  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`,
    class: "network-svg",
    role: "img",
    "aria-label": `网络结构：棋盘输入 → 卷积特征 ${channels} 通道 → 残差塔 ${blocks} 块 → 落点倾向 ${policyChannels} 通道与局面评估 hidden ${valueHidden} → MCTS`,
  });

  const defs = svgEl("defs");
  const marker = svgEl("marker", {
    id: "netArrow",
    viewBox: "0 0 10 10",
    refX: 9,
    refY: 5,
    markerWidth: 7,
    markerHeight: 7,
    orient: "auto-start-reverse",
  });
  marker.appendChild(svgEl("path", { d: "M0,0 L10,5 L0,10 z", class: "netarrow" }));
  defs.appendChild(marker);
  svg.appendChild(defs);

  const boxW = 116;
  const boxH = 52;
  const midY = (H - boxH) / 2;
  const topY = 36;
  const botY = H - boxH - 36;
  const colX = [16, 172, 328, 500, 644];

  const inputNode = networkNode(colX[0], midY, boxW, boxH, "棋盘输入", "黑 / 白两层");
  const stemNode = networkNode(colX[1], midY, boxW, boxH, "卷积特征", `${channels} 通道`);
  const towerNode = networkNode(colX[2], midY, boxW, boxH, "残差塔", `${blocks} 块`);
  const policyNode = networkNode(colX[3], topY, boxW, boxH, "落点倾向", `${policyChannels} 通道头`);
  const valueNode = networkNode(colX[3], botY, boxW, boxH, "局面评估", `hidden ${valueHidden}`);
  const mctsNode = networkNode(colX[4], midY, boxW, boxH, "MCTS", "反复搜索");

  svg.append(
    networkEdge(colX[0] + boxW, midY + boxH / 2, colX[1], midY + boxH / 2),
    networkEdge(colX[1] + boxW, midY + boxH / 2, colX[2], midY + boxH / 2),
    networkEdge(colX[2] + boxW, midY + boxH / 2 - 8, colX[3], topY + boxH / 2),
    networkEdge(colX[2] + boxW, midY + boxH / 2 + 8, colX[3], botY + boxH / 2),
    networkEdge(colX[3] + boxW, topY + boxH / 2, colX[4], midY + boxH / 2 - 8),
    networkEdge(colX[3] + boxW, botY + boxH / 2, colX[4], midY + boxH / 2 + 8),
  );
  svg.append(inputNode, stemNode, towerNode, policyNode, valueNode, mctsNode);

  networkDiagram.replaceChildren(svg);
  networkDiagram.classList.add("is-rendered");
  networkDiagram.classList.remove("diagram-error");
  renderedNetworkKey = key;
}

function setOverlay(mode) {
  overlayMode = mode;
  syncOverlayButtons();
  if (state) renderBoard();
}

sideInputs.forEach((input) => {
  input.addEventListener("change", () => {
    if (!input.checked) return;
    selectedSide = input.value;
    sideLabel.textContent = selectedSide === "black" ? "执黑" : "执白";
  });
});

modelInputs.forEach((input) => {
  input.addEventListener("change", () => {
    if (!input.checked) return;
    switchModel(input.value);
  });
});

overlayInputs.forEach((input) => {
  input.addEventListener("change", () => {
    if (input.checked) setOverlay(input.value);
  });
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
analysisDetails.addEventListener("toggle", () => {
  if (!analysisDetails.open) return;
  renderTree();
});
networkDetails.addEventListener("toggle", () => {
  if (networkDetails.open) initNetworkDiagram();
});

function registerServiceWorker() {
  // Fail-safe: no-op if unsupported or registration rejects. Relative scope so
  // it works under the GitHub Pages subpath.
  try {
    if (!("serviceWorker" in navigator)) return;
    window.addEventListener("load", () => {
      navigator.serviceWorker
        .register("./sw.js", { scope: "./" })
        .catch(() => undefined);
    });
  } catch (error) {
    /* ignore */
  }
}

// Catalog is the source of truth for the model-selector copy. Manifest config
// drives the per-model numbers; the catalog's `arch`/`iteration` are used only
// as a fallback so labels still render before a manifest is fetched.
async function loadCatalogForSelector() {
  if (modelCatalog) return modelCatalog;
  try {
    const response = await fetch(`./assets/models/catalog.json?v=${APP_VERSION}`);
    if (response.ok) modelCatalog = await response.json();
  } catch (error) {
    modelCatalog = null;
  }
  return modelCatalog;
}

function archLabel(arch, iteration) {
  if (!arch) return "";
  const { channels, residual_blocks: blocks } = arch;
  const size = channels && blocks ? `${channels}×${blocks}` : "";
  const iter = iteration === null || iteration === undefined ? "" : `iter ${iteration}`;
  return [size, iter].filter(Boolean).join(" · ");
}

function populateModelLabels(catalog) {
  if (!catalog || !catalog.models) return;
  const byId = new Map(catalog.models.map((model) => [model.id, model]));
  modelInputs.forEach((input) => {
    const entry = byId.get(input.value);
    if (!entry) return;
    const small = input.closest(".model-option")?.querySelector("small");
    const text = archLabel(entry.arch, entry.iteration);
    if (small && text) small.textContent = text;
  });
}

async function boot() {
  registerServiceWorker();
  const catalog = await loadCatalogForSelector();
  if (catalog) {
    populateModelLabels(catalog);
    if (catalog.defaultModel) {
      selectedModelId = catalog.defaultModel;
      modelInputs.forEach((input) => { input.checked = input.value === selectedModelId; });
    }
  }
  try {
    setBusy(true, "加载模型");
    await callWorker("init", { modelId: selectedModelId });
    loadBar.style.transform = "scaleX(1)";
    loadPercent.textContent = "100%";
    loadText.textContent = `${modelDisplayName(selectedModelId)} 就绪，正在自动开局`;
    setBusy(false);
    await newGame();
    loadLine.classList.add("ready");
  } catch (error) {
    statusPill.textContent = "加载失败";
    statusPill.className = "status-pill done";
    showToast(error.message);
    // Re-enable controls so the user is not stuck, and offer a one-click retry.
    setBusy(false);
    loadLine.classList.remove("ready");
    loadText.textContent = "加载失败，点这里重试";
    loadLine.classList.add("retry");
    loadLine.style.cursor = "pointer";
    const retry = () => {
      loadLine.removeEventListener("click", retry);
      loadLine.classList.remove("retry");
      loadLine.style.cursor = "";
      boot();
    };
    loadLine.addEventListener("click", retry, { once: true });
  }
}

boot();
