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
const recSearch = document.querySelector("#recSearch");
const recPolicy = document.querySelector("#recPolicy");
const recWin = document.querySelector("#recWin");
const recommendationWhy = document.querySelector("#recommendationWhy");
const analysisDetails = document.querySelector(".analysis-details");
const sideInputs = [...document.querySelectorAll("input[name='side']")];
const overlayInputs = [...document.querySelectorAll("input[name='overlay']")];

let state = null;
let selectedSide = "white";
let overlayMode = "none";
let busy = true;
let cells = [];
let requestId = 0;
let evalRequestId = 0;
let pendingEvalKey = null;
let d3Tree = null;
let d3TreeError = false;
let d3TreePromise = null;
let mermaidReady = false;
let mermaidError = false;
let mermaidPromise = null;
const pending = new Map();

const worker = new Worker("./engine.worker.js");

const SIMULATION_OPTIONS = [16, 32, 64, 128, 256, 512];
const NETWORK_DIAGRAM = String.raw`
flowchart LR
  input["棋盘输入<br/>2×10×10"]
  stem["3×3 卷积<br/>192 通道"]
  tower["12× 残差块<br/>Conv + skip"]
  policy["策略头<br/>100 个落点"]
  value["价值头<br/>1 个胜率"]
  mcts["MCTS<br/>反复调用"]
  input --> stem --> tower
  tower --> policy --> mcts
  tower --> value --> mcts
`;
const playerName = (v) => (v === 1 ? "黑" : v === -1 ? "白" : "-");
const pct = (v, digits = 0) => `${(v * 100).toFixed(digits)}%`;
const moveText = (move) => (move ? `${move.row + 1},${move.col + 1}` : "-");

function stateKey(s) {
  if (!s) return "";
  return [
    s.movesPlayed,
    s.currentPlayer,
    s.lastMove === null ? "none" : s.lastMove,
    s.winner === null ? "none" : s.winner,
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
  return SIMULATION_OPTIONS[Number(simSlider.value)] || 256;
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
  [newGameBtn, undoBtn, analyzeBtn, simSlider, ...sideInputs].forEach((el) => {
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

function renderEval() {
  const evaluation = state && state.evaluation;
  const bw = evaluation ? evaluation.blackWinProb : null;
  if (bw === null) {
    const pending = state && state.evaluationPending;
    winProbEl.textContent = pending ? "..." : "-";
    evalLabel.textContent = pending ? "..." : "-";
    evalBlack.style.transform = "scaleY(0.5)";
    winMeterBlack.style.transform = "scaleX(0.5)";
    if (state && state.winner !== null) {
      evalSource.textContent = "对局已结束。";
    } else if (pending) {
      evalSource.textContent = "模型实时评估中。";
    } else {
      evalSource.textContent = "模型就绪后自动更新。";
    }
    statSims.textContent = pending ? "模型" : "-";
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
      evalSource.textContent = `复用 MCTS · ${playerName(evaluation.player)}方行棋`;
      statSims.textContent = "MCTS";
    } else {
      evalSource.textContent = `模型实时 · ${playerName(evaluation.player)}方行棋`;
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
  if (node.depth === 0) return 11;
  const share = Math.max(0.01, node.branchShare || node.share || 0.01);
  if (node.depth === 1) return 7 + Math.sqrt(share) * 15;
  if (node.depth === 2) return 5 + Math.sqrt(share) * 8;
  if (node.depth === 3) return 4 + Math.sqrt(share) * 5;
  return 3.3 + Math.sqrt(share) * 3.2;
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
  return `${move} · 模拟 ${node.visits} · 分支 ${pct(node.branchShare || 0)} · 当前方胜率 ${win}`;
}

function renderTreeWithD3(nodes, edges) {
  const rootData = buildTreeData(nodes);
  if (!rootData) {
    renderTreeMessage("搜索树数据不完整", "候选点仍然可以在下方表格查看");
    return;
  }

  const maxDepth = Math.max(...nodes.map((node) => node.depth));
  const width = 920;
  const height = Math.max(460, 148 + Math.max(1, maxDepth) * 72);
  const margin = { top: 46, right: 52, bottom: 42, left: 52 };
  const edgeByTarget = new Map(edges.map((edge) => [edge.to, edge]));
  const root = d3Tree.hierarchy(rootData);
  root.sort((a, b) => Number(b.data.principal) - Number(a.data.principal)
    || (b.data.visits || 0) - (a.data.visits || 0)
    || (a.data.rank || 0) - (b.data.rank || 0));

  d3Tree.tree()
    .size([width - margin.left - margin.right, height - margin.top - margin.bottom])
    .separation((a, b) => (a.parent === b.parent ? 1 : 1.35) + Math.abs(a.depth - b.depth) * 0.08)(root);

  root.each((node) => {
    node.x += margin.left;
    node.y += margin.top;
  });

  const svg = d3Tree.select(treeSvg);
  svg.selectAll("*").remove();
  svg.attr("viewBox", `0 0 ${width} ${height}`);

  const depthLabels = ["当前", "候选落点", "回应", "再展开", "深层", "尾部", "末端"];
  const rows = Array.from(d3Tree.group(root.descendants(), (node) => node.depth), ([depth, group]) => ({
    depth,
    y: group[0].y,
  })).sort((a, b) => a.depth - b.depth);

  const backdrop = svg.append("g").attr("class", "tree-backdrop");
  backdrop.selectAll("line")
    .data(rows)
    .join("line")
    .attr("class", "tree-depth-line")
    .attr("x1", 28)
    .attr("x2", width - 28)
    .attr("y1", (row) => row.y)
    .attr("y2", (row) => row.y);

  backdrop.selectAll("text")
    .data(rows)
    .join("text")
    .attr("class", "tree-depth-label")
    .attr("x", 34)
    .attr("y", (row) => row.y - 10)
    .text((row) => depthLabels[row.depth] || `第 ${row.depth} 层`);

  const link = d3Tree.linkVertical()
    .x((node) => node.x)
    .y((node) => node.y);

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
    .attr("transform", (item) => `translate(${item.x.toFixed(1)},${item.y.toFixed(1)})`);

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
  treeTitle.textContent = "MCTS 树";
  treeCopy.textContent = context
    ? "这里只显示访问最多的摘要树。线越粗，代表这条分支被模拟得越多；红线是当前主线。"
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
    renderTreeMessage("树图加载失败", "D3 CDN 暂时不可用，候选表仍可用");
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
        renderTreeMessage("树图加载失败", "D3 CDN 暂时不可用，候选表仍可用");
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
  statusPill.textContent = state.status;
  statusPill.className = `status-pill ${state.winner !== null ? "done" : ""}`;
  undoBtn.disabled = !state.canUndo || busy;
  requestLiveEvaluation();
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

async function initNetworkDiagram() {
  if (!networkDiagram || mermaidReady || mermaidError) return;
  networkDiagram.textContent = "网络结构图加载中";
  try {
    if (!mermaidPromise) {
      mermaidPromise = import("./assets/vendor/mermaid/mermaid.esm.min.mjs");
    }
    const { default: mermaid } = await mermaidPromise;
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      flowchart: {
        curve: "basis",
        htmlLabels: true,
      },
      themeVariables: {
        background: "transparent",
        primaryColor: "#ffffff",
        primaryTextColor: "#20231f",
        primaryBorderColor: "#aab4a6",
        lineColor: "#596255",
        secondaryColor: "#f1f3ef",
        tertiaryColor: "#fbfcfa",
        fontFamily: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
      },
    });
    const { svg } = await mermaid.render("networkDiagramSvg", NETWORK_DIAGRAM);
    networkDiagram.innerHTML = svg;
    networkDiagram.classList.add("is-rendered");
    mermaidReady = true;
  } catch (error) {
    networkDiagram.textContent = "网络结构图加载失败。请检查 Mermaid CDN 连接。";
    networkDiagram.classList.add("diagram-error");
    mermaidError = true;
  }
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
  initNetworkDiagram();
  renderTree();
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
