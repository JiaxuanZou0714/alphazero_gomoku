/* global ort */

const ORT_VERSION = "1.20.1";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

let session = null;
let modelConfig = null;
let game = null;

function sendProgress(label, loaded, total) {
  self.postMessage({ type: "progress", payload: { label, loaded, total } });
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function statusText(state, humanPlayer) {
  if (state.winner !== null) {
    if (state.winner === 0) return "平局";
    return state.winner === humanPlayer ? "你赢了" : "AI 获胜";
  }
  return state.currentPlayer === humanPlayer ? "轮到你" : "AI 行棋";
}

class GomokuState {
  constructor({
    board,
    currentPlayer = 1,
    lastMove = null,
    winner = null,
    movesPlayed = 0,
    winLength = 5,
  }) {
    this.board = board || new Int8Array(100);
    this.currentPlayer = currentPlayer;
    this.lastMove = lastMove;
    this.winner = winner;
    this.movesPlayed = movesPlayed;
    this.winLength = winLength;
    this.size = 10;
    this.actionSize = 100;
  }

  static new(size = 10, winLength = 5) {
    if (size !== 10) throw new Error("Pages 版当前只支持 10x10 模型");
    return new GomokuState({ board: new Int8Array(size * size), winLength });
  }

  clone() {
    return new GomokuState({
      board: new Int8Array(this.board),
      currentPlayer: this.currentPlayer,
      lastMove: this.lastMove,
      winner: this.winner,
      movesPlayed: this.movesPlayed,
      winLength: this.winLength,
    });
  }

  actionToCoord(action) {
    return [Math.floor(action / this.size), action % this.size];
  }

  coordToAction(row, col) {
    if (row < 0 || row >= this.size || col < 0 || col >= this.size) {
      throw new Error(`坐标越界: ${row + 1},${col + 1}`);
    }
    return row * this.size + col;
  }

  legalMask() {
    const mask = new Array(this.actionSize).fill(false);
    if (this.winner !== null) return mask;
    for (let i = 0; i < this.actionSize; i += 1) mask[i] = this.board[i] === 0;
    return mask;
  }

  legalActions() {
    const actions = [];
    if (this.winner !== null) return actions;
    for (let i = 0; i < this.actionSize; i += 1) {
      if (this.board[i] === 0) actions.push(i);
    }
    return actions;
  }

  apply(action) {
    if (this.winner !== null) throw new Error("终局后不能落子");
    const [row, col] = this.actionToCoord(action);
    if (this.board[action] !== 0) throw new Error(`该点已有棋子: ${row + 1},${col + 1}`);
    const board = new Int8Array(this.board);
    board[action] = this.currentPlayer;
    const movesPlayed = this.movesPlayed + 1;
    let winner = null;
    if (this.isWinningMove(board, row, col, this.currentPlayer)) winner = this.currentPlayer;
    else if (movesPlayed === this.actionSize) winner = 0;
    return new GomokuState({
      board,
      currentPlayer: -this.currentPlayer,
      lastMove: action,
      winner,
      movesPlayed,
      winLength: this.winLength,
    });
  }

  terminalValueForCurrentPlayer() {
    if (this.winner === null) throw new Error("局面不是终局");
    if (this.winner === 0) return 0;
    return this.winner === this.currentPlayer ? 1 : -1;
  }

  encodeInto(buffer, offset) {
    const planeSize = this.actionSize;
    for (let i = 0; i < planeSize; i += 1) {
      const v = this.board[i];
      buffer[offset + i] = v === this.currentPlayer ? 1 : 0;
      buffer[offset + planeSize + i] = v === -this.currentPlayer ? 1 : 0;
    }
  }

  isWinningMove(board, row, col, player) {
    const dirs = [[1, 0], [0, 1], [1, 1], [1, -1]];
    for (const [dr, dc] of dirs) {
      const count = 1
        + this.countDirection(board, row, col, player, dr, dc)
        + this.countDirection(board, row, col, player, -dr, -dc);
      if (count >= this.winLength) return true;
    }
    return false;
  }

  countDirection(board, row, col, player, dr, dc) {
    let count = 0;
    let r = row + dr;
    let c = col + dc;
    while (r >= 0 && r < this.size && c >= 0 && c < this.size && board[r * this.size + c] === player) {
      count += 1;
      r += dr;
      c += dc;
    }
    return count;
  }
}

class Node {
  constructor(prior, rawPrior = 0) {
    this.prior = prior;
    this.rawPrior = rawPrior;
    this.visitCount = 0;
    this.valueSum = 0;
    this.valueSqSum = 0;
    this.children = new Map();
  }

  get expanded() {
    return this.children.size > 0;
  }

  get value() {
    return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
  }

  get valueVar() {
    if (this.visitCount < 2) return 1;
    const mean = this.valueSum / this.visitCount;
    return Math.max(0, this.valueSqSum / this.visitCount - mean * mean);
  }
}

class BrowserMCTS {
  constructor(config) {
    this.config = {
      simulations: 256,
      cPuct: Number(config.mcts_c_puct ?? 1.5),
      evalBatchSize: Math.max(1, Math.min(Number(config.mcts_batch_size ?? 16), 64)),
      rootPolicyTemp: Number(config.mcts_root_policy_temp ?? 1.0),
      dynamicCpuct: Boolean(config.mcts_dynamic_cpuct),
      fpuReduction: Number(config.mcts_fpu_reduction ?? 0),
    };
  }

  async search(state, simulations = null) {
    if (state.winner !== null) throw new Error("终局不能搜索");
    const sims = Math.max(1, simulations ?? this.config.simulations);
    const root = new Node(1, 1);
    await this.expand(root, state, true);

    let simulationsDone = 0;
    while (simulationsDone < sims) {
      const batchSize = Math.min(this.config.evalBatchSize, sims - simulationsDone);
      const leaves = [];
      const seen = new Set();
      let attempts = 0;
      let terminalBackups = 0;

      while (leaves.length < batchSize && attempts < batchSize * 4) {
        attempts += 1;
        let node = root;
        let scratch = state.clone();
        const searchPath = [node];

        while (node.expanded) {
          const [action, nextNode] = this.selectChild(node, node === root);
          node = nextNode;
          scratch = scratch.apply(action);
          searchPath.push(node);
        }

        if (scratch.winner !== null) {
          this.backpropagate(searchPath, scratch.terminalValueForCurrentPlayer());
          terminalBackups += 1;
          simulationsDone += 1;
          if (simulationsDone >= sims) break;
          continue;
        }
        if (seen.has(node)) continue;
        seen.add(node);
        this.addVirtualVisits(searchPath);
        leaves.push({ searchPath, node, scratch });
      }

      if (!leaves.length) {
        if (terminalBackups === 0) break;
        continue;
      }

      const evaluations = await evaluateBatch(leaves.map((leaf) => leaf.scratch));
      for (let i = 0; i < leaves.length; i += 1) {
        const leaf = leaves[i];
        this.removeVirtualVisits(leaf.searchPath);
        this.expandWithPolicy(leaf.node, leaf.scratch, evaluations[i].policy);
        this.backpropagate(leaf.searchPath, evaluations[i].value);
      }
      simulationsDone += leaves.length;
    }
    return root;
  }

  async expand(node, state, isRoot) {
    const [{ policy }] = await evaluateBatch([state], isRoot ? this.config.rootPolicyTemp : 1.0);
    this.expandWithPolicy(node, state, policy);
  }

  expandWithPolicy(node, state, policy) {
    for (const action of state.legalActions()) {
      const prior = Number(policy[action]);
      node.children.set(action, new Node(prior, prior));
    }
  }

  selectChild(node, isRoot) {
    const parentVisits = Math.max(1, node.visitCount);
    const sqrtParent = Math.sqrt(parentVisits);
    let c = this.config.cPuct;
    if (this.config.dynamicCpuct) c *= Math.sqrt(Math.max(0.25, node.valueVar));

    const fpu = isRoot ? 0 : this.config.fpuReduction;
    let fpuValue = 0;
    if (fpu > 0) {
      let visitedMass = 0;
      for (const child of node.children.values()) {
        if (child.visitCount > 0) visitedMass += child.prior;
      }
      fpuValue = node.value - fpu * Math.sqrt(visitedMass);
    }

    let bestScore = -Infinity;
    const best = [];
    for (const [action, child] of node.children.entries()) {
      const priorScore = c * child.prior * sqrtParent / (child.visitCount + 1);
      const q = child.visitCount > 0 ? -child.value : fpuValue;
      const score = q + priorScore;
      if (score > bestScore) {
        bestScore = score;
        best.length = 0;
        best.push([action, child]);
      } else if (score === bestScore) {
        best.push([action, child]);
      }
    }
    return best[Math.floor(Math.random() * best.length)];
  }

  addVirtualVisits(searchPath) {
    for (const node of searchPath) node.visitCount += 1;
  }

  removeVirtualVisits(searchPath) {
    for (const node of searchPath) node.visitCount -= 1;
  }

  backpropagate(searchPath, value) {
    let v = value;
    for (let i = searchPath.length - 1; i >= 0; i -= 1) {
      const node = searchPath[i];
      node.valueSum += v;
      node.valueSqSum += v * v;
      node.visitCount += 1;
      v = -v;
    }
  }
}

async function evaluateBatch(states, rootTemp = 1.0) {
  const n = states.length;
  const actionSize = 100;
  const encoded = new Float32Array(n * 2 * actionSize);
  for (let i = 0; i < n; i += 1) states[i].encodeInto(encoded, i * 2 * actionSize);

  const input = new ort.Tensor("float32", encoded, [n, 2, 10, 10]);
  const outputs = await session.run({ board: input });
  const logits = outputs.policy_logits.data;
  const values = outputs.value.data;
  const result = [];

  for (let i = 0; i < n; i += 1) {
    const state = states[i];
    const mask = state.legalMask();
    const offset = i * actionSize;
    let maxLogit = -1e9;
    for (let a = 0; a < actionSize; a += 1) {
      if (!mask[a]) continue;
      let logit = logits[offset + a];
      if (!Number.isFinite(logit)) logit = 0;
      if (rootTemp !== 1.0 && rootTemp > 0) logit /= rootTemp;
      if (logit > maxLogit) maxLogit = logit;
    }

    const policy = new Float32Array(actionSize);
    let total = 0;
    let legalCount = 0;
    for (let a = 0; a < actionSize; a += 1) {
      if (!mask[a]) continue;
      legalCount += 1;
      let logit = logits[offset + a];
      if (!Number.isFinite(logit)) logit = 0;
      if (rootTemp !== 1.0 && rootTemp > 0) logit /= rootTemp;
      const e = Math.exp(logit - maxLogit);
      policy[a] = e;
      total += e;
    }
    if (total > 0) {
      for (let a = 0; a < actionSize; a += 1) policy[a] /= total;
    } else {
      const fallback = legalCount > 0 ? 1 / legalCount : 0;
      for (let a = 0; a < actionSize; a += 1) policy[a] = mask[a] ? fallback : 0;
    }

    result.push({ policy, value: clamp(Number(values[i] ?? 0), -1, 1) });
  }
  return result;
}

function rootPolicy(root) {
  const visits = new Float32Array(100);
  for (const [action, child] of root.children.entries()) visits[action] = child.visitCount;
  let total = 0;
  for (const v of visits) total += v;
  if (total > 0) {
    for (let i = 0; i < visits.length; i += 1) visits[i] /= total;
  }
  return visits;
}

function searchAnalysis(root, state, elapsedMs) {
  const actionSize = state.actionSize;
  const visitMap = Array.from(rootPolicy(root));
  const priorMap = new Array(actionSize).fill(0);
  const qMap = new Array(actionSize).fill(null);
  let totalVisits = 0;
  const items = [...root.children.entries()];
  for (const [, child] of items) totalVisits += child.visitCount;
  totalVisits = totalVisits || 1;

  for (const [action, child] of items) {
    priorMap[action] = child.rawPrior;
    if (child.visitCount > 0) qMap[action] = -child.value;
  }

  const bestAction = items.reduce((best, item) => (
    item[1].visitCount > best[1].visitCount ? item : best
  ), items[0])[0];
  const ranked = [...items].sort((a, b) => b[1].visitCount - a[1].visitCount).slice(0, 8);
  const candidates = ranked.map(([action, child]) => ({
    row: Math.floor(action / state.size),
    col: action % state.size,
    visits: child.visitCount,
    share: child.visitCount / totalVisits,
    prior: child.rawPrior,
    q: child.visitCount > 0 ? -child.value : null,
    selected: action === bestAction,
  }));

  const pv = [];
  let node = root;
  while (node.expanded && pv.length < 8) {
    const children = [...node.children.entries()];
    const [action, child] = children.reduce((best, item) => (
      item[1].visitCount > best[1].visitCount ? item : best
    ), children[0]);
    if (child.visitCount === 0) break;
    pv.push({ row: Math.floor(action / state.size), col: action % state.size });
    node = child;
  }

  const tree = buildSearchTree(root, state);

  return {
    action: bestAction,
    analysis: {
      player: state.currentPlayer,
      moveNumber: state.movesPlayed,
      rootValue: root.value,
      winProb: (root.value + 1) / 2,
      simulations: root.visitCount,
      elapsedMs,
      visitMap,
      priorMap,
      qMap,
      candidates,
      pv,
      tree,
    },
  };
}

function buildSearchTree(root, state) {
  const nodes = [];
  const edges = [];
  const maxDepth = 5;
  const principalChildLimit = [6, 3, 2, 2, 2];
  const sideChildLimit = [6, 2, 1, 0, 0];
  const rootVisits = Math.max(1, root.visitCount);

  function addNode(
    node,
    nodeState,
    depth,
    parentId = null,
    action = null,
    mover = null,
    branchShare = 1,
    rank = 0,
    principal = true,
  ) {
    const id = nodes.length;
    const isRoot = parentId === null;
    const q = isRoot ? node.value : (node.visitCount > 0 ? -node.value : null);
    const row = action === null ? null : Math.floor(action / nodeState.size);
    const col = action === null ? null : action % nodeState.size;
    nodes.push({
      id,
      parentId,
      depth,
      row,
      col,
      mover,
      visits: node.visitCount,
      share: node.visitCount / rootVisits,
      branchShare,
      rank,
      principal,
      prior: node.rawPrior,
      winProb: q === null ? null : (q + 1) / 2,
    });
    if (parentId !== null) edges.push({ from: parentId, to: id, share: branchShare, rank, principal });

    if (depth >= maxDepth || !node.expanded) return id;
    const limit = principal ? principalChildLimit[depth] : sideChildLimit[depth];
    if (!limit) return id;
    const children = [...node.children.entries()]
      .filter(([, child]) => child.visitCount > 0)
      .sort((a, b) => b[1].visitCount - a[1].visitCount)
      .slice(0, limit);
    const total = children.reduce((sum, [, child]) => sum + child.visitCount, 0) || 1;
    children.forEach(([childAction, child], index) => {
      addNode(
        child,
        nodeState.apply(childAction),
        depth + 1,
        id,
        childAction,
        nodeState.currentPlayer,
        child.visitCount / total,
        index,
        principal && index === 0,
      );
    });
    return id;
  }

  addNode(root, state, 0);
  return { nodes, edges };
}

class GameSession {
  constructor(config) {
    this.config = config;
    this.defaultSimulations = 256;
    this.mcts = new BrowserMCTS(config);
    this.newGame("white", this.defaultSimulations);
  }

  newGame(human = "white", simulations = this.defaultSimulations) {
    this.state = GomokuState.new(
      Number(this.config.board_size ?? 10),
      Number(this.config.win_length ?? 5),
    );
    this.humanPlayer = -1;
    this.simulations = Math.max(1, Number(simulations));
    this.history = [];
    this.lastAnalysis = {};
    this.policySource = "none";
    this.policyPlayer = null;
    this.evalHistory = [];
    this.undoStack = [];
    return this.playOpeningIfNeeded();
  }

  async playOpeningIfNeeded() {
    if (this.state.currentPlayer !== this.humanPlayer) await this.aiMove();
    return this.snapshot();
  }

  async humanMove(row, col, simulations = null) {
    if (simulations !== null) this.simulations = Math.max(1, Number(simulations));
    if (this.state.winner !== null) throw new Error("对局已经结束");
    if (this.state.currentPlayer !== this.humanPlayer) throw new Error("还没轮到你");
    const action = this.state.coordToAction(row, col);
    if (this.state.board[action] !== 0) throw new Error("这个点已经有棋子");
    this.undoStack.push({
      state: this.state.clone(),
      history: this.history.map((m) => ({ ...m })),
      lastAnalysis: structuredClone(this.lastAnalysis),
      policySource: this.policySource,
      policyPlayer: this.policyPlayer,
      evalHistory: this.evalHistory.map((e) => ({ ...e })),
    });
    this.state = this.state.apply(action);
    this.lastAnalysis = {};
    this.policySource = "none";
    this.policyPlayer = null;
    this.history.push({ player: this.state.currentPlayer === -1 ? "black" : "white", source: "human", row, col });
    if (this.state.winner === null) await this.aiMove();
    return this.snapshot();
  }

  undo() {
    if (!this.undoStack.length) throw new Error("没有可悔的棋");
    const previous = this.undoStack.pop();
    this.state = previous.state;
    this.history = previous.history;
    this.lastAnalysis = previous.lastAnalysis;
    this.policySource = previous.policySource;
    this.policyPlayer = previous.policyPlayer;
    this.evalHistory = previous.evalHistory;
    return this.snapshot();
  }

  async analyze(simulations = null) {
    if (simulations !== null) this.simulations = Math.max(1, Number(simulations));
    if (this.state.winner !== null) throw new Error("终局后不能生成提示");
    const { analysis } = await this.searchPolicy();
    this.lastAnalysis = analysis;
    this.policySource = "analysis";
    this.policyPlayer = this.state.currentPlayer;
    this.recordEval(analysis);
    return this.snapshot();
  }

  async aiMove() {
    const player = this.state.currentPlayer;
    const { action, analysis } = await this.searchPolicy();
    if (action < 0 || action === undefined) throw new Error("AI 没找到合法落点");
    this.lastAnalysis = analysis;
    this.policySource = "ai_move";
    this.policyPlayer = player;
    this.recordEval(analysis);
    const row = Math.floor(action / this.state.size);
    const col = action % this.state.size;
    this.state = this.state.apply(action);
    this.history.push({ player: player === 1 ? "black" : "white", source: "ai", row, col });
  }

  async searchPolicy() {
    const start = performance.now();
    const root = await this.mcts.search(this.state, this.simulations);
    return searchAnalysis(root, this.state, performance.now() - start);
  }

  recordEval(analysis) {
    if (!analysis) return;
    const blackWinProb = analysis.player === 1 ? analysis.winProb : 1 - analysis.winProb;
    const move = this.state.movesPlayed;
    this.evalHistory = this.evalHistory.filter((e) => e.move !== move);
    this.evalHistory.push({ move, blackWinProb });
  }

  snapshot() {
    const rows = [];
    for (let r = 0; r < this.state.size; r += 1) {
      rows.push(Array.from(this.state.board.slice(r * this.state.size, (r + 1) * this.state.size)));
    }
    return {
      board: rows,
      size: this.state.size,
      currentPlayer: this.state.currentPlayer,
      humanPlayer: this.humanPlayer,
      winner: this.state.winner,
      movesPlayed: this.state.movesPlayed,
      lastMove: this.state.lastMove,
      history: this.history,
      analysis: this.lastAnalysis,
      policySource: this.policySource,
      policyPlayer: this.policyPlayer,
      evalHistory: this.evalHistory,
      simulations: this.simulations,
      canUndo: this.undoStack.length > 0,
      status: statusText(this.state, this.humanPlayer),
    };
  }
}

async function fetchBytes(url, onProgress = null) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`下载失败: ${url}`);
  if (!response.body || !onProgress) return new Uint8Array(await response.arrayBuffer());

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function loadModel() {
  sendProgress("加载运行时", 0, 1);
  importScripts(`${ORT_BASE}ort.webgpu.min.js`);
  ort.env.wasm.wasmPaths = ORT_BASE;

  const manifestUrl = new URL("assets/model/manifest.json", self.location.href);
  const manifest = await fetch(manifestUrl).then((r) => {
    if (!r.ok) throw new Error("没有找到模型 manifest，请先运行导出脚本");
    return r.json();
  });

  const total = manifest.bytes || manifest.chunks.reduce((sum, chunk) => sum + chunk.bytes, 0);
  const parts = [];
  let loaded = 0;
  for (const [index, chunk] of manifest.chunks.entries()) {
    const url = new URL(chunk.file, manifestUrl);
    const bytes = await fetchBytes(url, (partLoaded) => {
      sendProgress(`加载模型 ${index + 1}/${manifest.chunks.length}`, loaded + partLoaded, total);
    });
    parts.push(bytes);
    loaded += bytes.byteLength;
    sendProgress(`加载模型 ${index + 1}/${manifest.chunks.length}`, loaded, total);
  }

  const modelBytes = new Uint8Array(loaded);
  let offset = 0;
  for (const part of parts) {
    modelBytes.set(part, offset);
    offset += part.byteLength;
  }

  sendProgress("初始化模型", total, total);
  modelConfig = manifest.config || {};
  try {
    session = await ort.InferenceSession.create(modelBytes.buffer, {
      executionProviders: ["webgpu", "wasm"],
      graphOptimizationLevel: "all",
    });
  } catch (error) {
    session = await ort.InferenceSession.create(modelBytes.buffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }
  game = new GameSession(modelConfig);
}

self.addEventListener("message", async (event) => {
  const { id, type, payload = {} } = event.data;
  try {
    let result;
    if (type === "init") {
      await loadModel();
      result = { ok: true };
    } else {
      if (!game) throw new Error("模型尚未加载");
      if (type === "newGame") result = await game.newGame(payload.human, payload.simulations);
      else if (type === "move") result = await game.humanMove(payload.row, payload.col, payload.simulations);
      else if (type === "undo") result = game.undo();
      else if (type === "analyze") result = await game.analyze(payload.simulations);
      else throw new Error(`未知命令: ${type}`);
    }
    self.postMessage({ id, payload: result });
  } catch (error) {
    self.postMessage({ id, error: error && error.message ? error.message : String(error) });
  }
});
