/* global ort */

const ORT_BASE = new URL("assets/vendor/onnxruntime-web/", self.location.href).href;

let session = null;
let modelConfig = null;
let activeModel = null;
let modelCatalog = null;
const sessionCache = new Map();
let game = null;

/* Fixed-shape batching for WebGPU: the EP JIT-compiles a fresh compute pipeline
 * for every distinct input batch dimension it sees, so a search that runs
 * batches of 32, 32, …, 8, plus singleton expand/evaluate calls, pays repeated
 * shader recompiles. We pad every multi-leaf batch up to `evalPad` so the GPU
 * only ever sees two shapes (1 and evalPad); padded rows are zero-filled empty
 * boards and discarded. The wasm backend recompiles nothing, so padding there
 * would only waste CPU — gate it on the backend. */
let evalPad = 1;
let padEnabled = false;

function configureEvalShape() {
  evalPad = Math.max(1, Math.min(Number((modelConfig && modelConfig.mcts_batch_size) ?? 16), 64));
  padEnabled = !!(activeModel && activeModel.backend === "webgpu");
}

function sendProgress(label, loaded, total) {
  self.postMessage({ type: "progress", payload: { label, loaded, total } });
}

function sendInfo(message) {
  self.postMessage({ type: "info", payload: { message } });
}

/* Persistent model-bytes cache in IndexedDB, keyed by the manifest sha256 so a
 * re-export with a new hash auto-invalidates the old entry. */
const IDB_NAME = "az-gomoku-models";
const IDB_STORE = "modelBytes";
let idbPromise = null;

function openModelDb() {
  if (idbPromise) return idbPromise;
  idbPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB 不可用"));
      return;
    }
    const request = indexedDB.open(IDB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) db.createObjectStore(IDB_STORE);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("无法打开模型缓存"));
  }).catch((error) => {
    idbPromise = null;
    throw error;
  });
  return idbPromise;
}

async function idbGetModel(key) {
  try {
    const db = await openModelDb();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, "readonly");
      const req = tx.objectStore(IDB_STORE).get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  } catch (error) {
    return null;
  }
}

async function idbPutModel(key, bytes) {
  try {
    const db = await openModelDb();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, "readwrite");
      tx.objectStore(IDB_STORE).put(bytes, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (error) {
    /* Caching is best-effort; ignore quota or unavailability errors. */
  }
}

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
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

function modelSummary(manifest, entry = null) {
  const cfg = manifest.config || {};
  return {
    id: manifest.id || (entry && entry.id) || "default",
    label: manifest.label || (entry && entry.label) || "model",
    iteration: manifest.checkpointIteration ?? (entry && entry.iteration) ?? null,
    bytes: manifest.bytes || (entry && entry.bytes) || 0,
    config: cfg,
  };
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
      simulations: 512,
      cPuct: Number(config.mcts_c_puct ?? 1.5),
      evalBatchSize: Math.max(1, Math.min(Number(config.mcts_batch_size ?? 16), 64)),
      rootPolicyTemp: Number(config.mcts_root_policy_temp ?? 1.0),
      dynamicCpuct: Boolean(config.mcts_dynamic_cpuct),
      fpuReduction: Number(config.mcts_fpu_reduction ?? 0),
    };
  }

  async search(state, simulations = null, reuseRoot = null) {
    if (state.winner !== null) throw new Error("终局不能搜索");
    const sims = Math.max(1, simulations ?? this.config.simulations);
    const root = reuseRoot instanceof Node ? reuseRoot : new Node(1, 1);
    if (!root.expanded) await this.expand(root, state, true);

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
        const searchPath = [node];

        // In-place descent: copy the board once and mutate it per ply instead of
        // allocating a fresh state on every apply(). Every node we pass THROUGH is
        // expanded ⇒ non-terminal (terminals are backed up, never expanded), so its
        // arriving move was neither a win nor a board-fill — we therefore run the
        // win check only on the final (leaf) move. Verified equivalent to the
        // chained immutable apply() over 20k random descents.
        const board = new Int8Array(state.board);
        let player = state.currentPlayer;
        let moves = state.movesPlayed;
        let lastAction = state.lastMove;
        let applied = false;
        while (node.expanded) {
          const [action, nextNode] = this.selectChild(node, node === root);
          node = nextNode;
          board[action] = player;
          lastAction = action;
          player = -player;
          moves += 1;
          applied = true;
          searchPath.push(node);
        }
        let winner = state.winner;
        if (applied) {
          const mover = -player; // the player who placed lastAction (player flipped after)
          const row = Math.floor(lastAction / 10);
          const col = lastAction % 10;
          if (state.isWinningMove(board, row, col, mover)) winner = mover;
          else winner = moves === 100 ? 0 : null;
        }
        const scratch = new GomokuState({
          board,
          currentPlayer: player,
          lastMove: lastAction,
          winner,
          movesPlayed: moves,
          winLength: state.winLength,
        });

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

/* Transposition / eval cache: the network output depends only on the board +
 * side-to-move (the encode is canonical), so positions reached by different move
 * orders share an evaluation. Gomoku produces many such transpositions within a
 * single search, so caching skips the NN (and a GPU round-trip) for repeats.
 * Results are READ-only in the MCTS (expand/backprop never mutate policy/value),
 * so a cached object can be safely shared across hits. Cleared on model switch
 * (loadModel) since a different net gives different outputs. Bounded FIFO. */
const evalCache = new Map();
const EVAL_CACHE_MAX = 20000;

function evalCacheKey(state, rootTemp) {
  return `${rootTemp}|${state.currentPlayer}|${Array.from(state.board).join(",")}`;
}

async function evaluateBatch(states, rootTemp = 1.0) {
  const n = states.length;
  const actionSize = 100;
  const results = new Array(n);
  const missIdx = [];
  const missStates = [];
  const missKeys = [];
  for (let i = 0; i < n; i += 1) {
    const key = evalCacheKey(states[i], rootTemp);
    const hit = evalCache.get(key);
    if (hit !== undefined) {
      results[i] = hit;
    } else {
      missIdx.push(i);
      missStates.push(states[i]);
      missKeys.push(key);
    }
  }
  if (missStates.length === 0) return results;

  const m = missStates.length;
  // Pad multi-leaf batches up to evalPad so WebGPU only compiles two shapes
  // (1 and evalPad). Rows [m, padN) stay zero-filled (empty board) and the
  // output loop below reads only the real rows, so per-sample results are
  // unaffected (BN runs on frozen running stats, so batch composition is inert).
  const padN = padEnabled && m > 1 && m < evalPad ? evalPad : m;
  const encoded = new Float32Array(padN * 2 * actionSize);
  for (let i = 0; i < m; i += 1) missStates[i].encodeInto(encoded, i * 2 * actionSize);

  const input = new ort.Tensor("float32", encoded, [padN, 2, 10, 10]);
  const outputs = await session.run({ board: input });
  const logits = outputs.policy_logits.data;
  const values = outputs.value.data;

  for (let i = 0; i < m; i += 1) {
    const state = missStates[i];
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

    const entry = { policy, value: clamp(Number(values[i] ?? 0), -1, 1) };
    results[missIdx[i]] = entry;
    evalCache.set(missKeys[i], entry);
    if (evalCache.size > EVAL_CACHE_MAX) {
      evalCache.delete(evalCache.keys().next().value);
    }
  }
  return results;
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

function searchAnalysis(root, state, elapsedMs, requestedSimulations) {
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
      requestedSimulations,
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

function stateKey(state) {
  return [
    state.movesPlayed,
    state.currentPlayer,
    state.lastMove === null ? "none" : state.lastMove,
    state.winner === null ? "none" : state.winner,
    Array.from(state.board).join(","),
  ].join(":");
}

function terminalEvaluation(state, elapsedMs = 0) {
  const blackWinProb = state.winner === 0 ? 0.5 : state.winner === 1 ? 1 : 0;
  const currentWinProb = state.winner === 0 ? 0.5 : state.winner === state.currentPlayer ? 1 : 0;
  return {
    key: stateKey(state),
    source: "terminal",
    player: state.currentPlayer,
    moveNumber: state.movesPlayed,
    lastMove: state.lastMove,
    winner: state.winner,
    value: null,
    winProb: currentWinProb,
    blackWinProb,
    elapsedMs,
  };
}

function rootEvaluation(analysis, state, source = "mcts") {
  const blackWinProb = analysis.player === 1 ? analysis.winProb : 1 - analysis.winProb;
  return {
    key: stateKey(state),
    source,
    player: state.currentPlayer,
    moveNumber: state.movesPlayed,
    lastMove: state.lastMove,
    winner: state.winner,
    value: analysis.rootValue,
    winProb: state.currentPlayer === 1 ? blackWinProb : 1 - blackWinProb,
    blackWinProb,
    simulations: analysis.simulations,
    elapsedMs: analysis.elapsedMs,
  };
}

function childEvaluation(analysis, action, stateAfterMove) {
  if (stateAfterMove.winner !== null) {
    return {
      ...terminalEvaluation(stateAfterMove, analysis.elapsedMs),
      source: "mcts",
      simulations: analysis.simulations,
    };
  }

  const childQ = analysis.qMap && analysis.qMap[action] !== undefined
    ? analysis.qMap[action]
    : null;
  if (childQ === null) return null;

  const rootWinProb = (childQ + 1) / 2;
  const blackWinProb = analysis.player === 1 ? rootWinProb : 1 - rootWinProb;
  return {
    key: stateKey(stateAfterMove),
    source: "mcts",
    player: stateAfterMove.currentPlayer,
    moveNumber: stateAfterMove.movesPlayed,
    lastMove: stateAfterMove.lastMove,
    winner: stateAfterMove.winner,
    value: stateAfterMove.currentPlayer === analysis.player ? childQ : -childQ,
    winProb: stateAfterMove.currentPlayer === 1 ? blackWinProb : 1 - blackWinProb,
    blackWinProb,
    simulations: analysis.simulations,
    elapsedMs: analysis.elapsedMs,
  };
}

function buildSearchTree(root, state) {
  const nodes = [];
  const edges = [];
  const maxDepth = 6;
  const principalChildLimit = [6, 3, 2, 2, 1, 1];
  const sideChildLimit = [6, 2, 1, 1, 0, 0];
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
  constructor(config, modelMeta) {
    this.config = config;
    this.modelMeta = modelMeta;
    this.defaultSimulations = 512;
    this.mcts = new BrowserMCTS(config);
    this.state = GomokuState.new(
      Number(this.config.board_size ?? 10),
      Number(this.config.win_length ?? 5),
    );
    this.humanPlayer = -1;
    this.simulations = this.defaultSimulations;
    this.history = [];
    this.lastAnalysis = {};
    this.policySource = "none";
    this.policyPlayer = null;
    this.evaluation = null;
    this.evalHistory = [];
    this.undoStack = [];
    this.reusableRoot = null;
    this.reusableKey = null;
  }

  currentReusableRoot() {
    return this.reusableRoot && this.reusableKey === stateKey(this.state)
      ? this.reusableRoot
      : null;
  }

  storeReusableRoot(root, state = this.state) {
    if (root && state.winner === null) {
      this.reusableRoot = root;
      this.reusableKey = stateKey(state);
    } else {
      this.reusableRoot = null;
      this.reusableKey = null;
    }
  }

  childReusableRoot(action) {
    const root = this.currentReusableRoot();
    return root ? root.children.get(action) || null : null;
  }

  newGame(human = "white", simulations = this.defaultSimulations) {
    this.state = GomokuState.new(
      Number(this.config.board_size ?? 10),
      Number(this.config.win_length ?? 5),
    );
    this.humanPlayer = human === "black" ? 1 : -1;
    this.simulations = Math.max(1, Number(simulations));
    this.history = [];
    this.lastAnalysis = {};
    this.policySource = "none";
    this.policyPlayer = null;
    this.evaluation = null;
    this.evalHistory = [];
    this.undoStack = [];
    this.storeReusableRoot(null);
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
    const nextRoot = this.childReusableRoot(action);
    const nextState = this.state.apply(action);
    this.undoStack.push({
      state: this.state.clone(),
      history: this.history.map((m) => ({ ...m })),
      lastAnalysis: structuredClone(this.lastAnalysis),
      policySource: this.policySource,
      policyPlayer: this.policyPlayer,
      evaluation: structuredClone(this.evaluation),
      evalHistory: this.evalHistory.map((e) => ({ ...e })),
      reusableRoot: this.reusableRoot,
      reusableKey: this.reusableKey,
    });
    this.state = nextState;
    this.lastAnalysis = {};
    this.policySource = "none";
    this.policyPlayer = null;
    this.evaluation = this.state.winner === null ? null : terminalEvaluation(this.state);
    this.storeReusableRoot(nextRoot);
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
    this.evaluation = previous.evaluation;
    this.evalHistory = previous.evalHistory;
    this.reusableRoot = previous.reusableRoot;
    this.reusableKey = previous.reusableKey;
    return this.snapshot();
  }

  async analyze(simulations = null) {
    if (simulations !== null) this.simulations = Math.max(1, Number(simulations));
    if (this.state.winner !== null) throw new Error("终局后不能生成提示");
    const { analysis } = await this.searchPolicy();
    this.lastAnalysis = analysis;
    this.policySource = "analysis";
    this.policyPlayer = this.state.currentPlayer;
    this.evaluation = rootEvaluation(analysis, this.state);
    this.recordEval(this.evaluation);
    return this.snapshot();
  }

  async evaluateCurrent() {
    const target = this.state.clone();
    const start = performance.now();

    if (target.winner !== null) {
      const evaluation = terminalEvaluation(target);
      this.evaluation = evaluation;
      return evaluation;
    } else {
      const [evaluation] = await evaluateBatch([target]);
      const currentWinProb = (evaluation.value + 1) / 2;
      const blackWinProb = target.currentPlayer === 1 ? currentWinProb : 1 - currentWinProb;
      const result = {
        key: stateKey(target),
        source: "model",
        player: target.currentPlayer,
        moveNumber: target.movesPlayed,
        lastMove: target.lastMove,
        winner: target.winner,
        value: evaluation.value,
        winProb: currentWinProb,
        blackWinProb,
        elapsedMs: performance.now() - start,
      };
      this.evaluation = result;
      return result;
    }
  }

  async aiMove() {
    const player = this.state.currentPlayer;
    const { action, analysis, root } = await this.searchPolicy();
    if (action < 0 || action === undefined) throw new Error("AI 没找到合法落点");
    this.lastAnalysis = analysis;
    this.policySource = "ai_move";
    this.policyPlayer = player;
    const row = Math.floor(action / this.state.size);
    const col = action % this.state.size;
    const nextState = this.state.apply(action);
    this.evaluation = childEvaluation(analysis, action, nextState);
    if (this.evaluation) this.recordEval(this.evaluation);
    this.state = nextState;
    this.storeReusableRoot(root.children.get(action) || null);
    this.history.push({ player: player === 1 ? "black" : "white", source: "ai", row, col });
  }

  async searchPolicy() {
    const start = performance.now();
    const root = await this.mcts.search(this.state, this.simulations, this.currentReusableRoot());
    this.storeReusableRoot(root);
    const result = searchAnalysis(root, this.state, performance.now() - start, this.simulations);
    result.root = root;
    return result;
  }

  recordEval(evaluation) {
    if (!evaluation) return;
    const move = evaluation.moveNumber;
    this.evalHistory = this.evalHistory.filter((e) => e.move !== move);
    this.evalHistory.push({ move, blackWinProb: evaluation.blackWinProb });
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
      evaluation: this.evaluation,
      evalHistory: this.evalHistory,
      simulations: this.simulations,
      canUndo: this.undoStack.length > 0,
      status: statusText(this.state, this.humanPlayer),
      model: this.modelMeta,
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

async function ensureRuntime() {
  sendProgress("加载运行时", 0, 1);
  if (!self.ort) {
    importScripts(`${ORT_BASE}ort.webgpu.min.js`);
    ort.env.wasm.wasmPaths = ORT_BASE;
  }
}

async function loadCatalog() {
  if (modelCatalog) return modelCatalog;
  const catalogUrl = new URL("assets/models/catalog.json", self.location.href);
  let response;
  try {
    response = await fetch(catalogUrl);
  } catch (error) {
    response = null;
  }
  if (response && response.ok) {
    const catalog = await response.json();
    modelCatalog = { catalog, baseUrl: catalogUrl };
    return modelCatalog;
  }
  // Fallback mirrors the shipped catalog so the UI selector never silently
  // serves a different model than the one the user picked. Keyed off the same
  // assets/models/ base as the real catalog.
  modelCatalog = {
    baseUrl: catalogUrl,
    catalog: {
      defaultModel: "v5",
      models: [
        { id: "v5", label: "v5", manifest: "v5/manifest.json" },
        { id: "v4", label: "v4", manifest: "v4/manifest.json" },
        { id: "v3", label: "v3", manifest: "v3/manifest.json" },
        { id: "v1", label: "v1", manifest: "v1/manifest.json" },
      ],
    },
  };
  return modelCatalog;
}

async function resolveModelManifest(modelId = null) {
  const { catalog, baseUrl } = await loadCatalog();
  const selectedId = modelId || catalog.defaultModel || (catalog.models[0] && catalog.models[0].id);
  const entry = catalog.models.find((item) => item.id === selectedId) || catalog.models[0];
  if (!entry) throw new Error("模型目录为空");
  const manifestUrl = new URL(entry.manifest, baseUrl);
  const manifest = await fetch(manifestUrl).then((r) => {
    if (!r.ok) throw new Error(`没有找到模型 manifest: ${entry.manifest}`);
    return r.json();
  });
  return { entry, manifest, manifestUrl };
}

async function loadModel(modelId = null) {
  await ensureRuntime();
  const { entry, manifest, manifestUrl } = await resolveModelManifest(modelId);
  const meta = modelSummary(manifest, entry);
  if (activeModel && activeModel.id === meta.id && session && modelConfig) {
    return;
  }
  if (sessionCache.has(meta.id)) {
    const cached = sessionCache.get(meta.id);
    evalCache.clear();
    session = cached.session;
    modelConfig = cached.modelConfig;
    activeModel = cached.activeModel;
    configureEvalShape();
    game = new GameSession(modelConfig, activeModel);
    sendProgress(`${activeModel.label} 已缓存`, 1, 1);
    return;
  }

  const modelBytes = await assembleModelBytes(manifest, manifestUrl, meta);

  sendProgress(`初始化 ${meta.label}`, modelBytes.byteLength, modelBytes.byteLength);
  evalCache.clear();
  modelConfig = manifest.config || {};
  let backend = "wasm";
  try {
    session = await ort.InferenceSession.create(modelBytes.buffer, {
      executionProviders: ["webgpu", "wasm"],
      graphOptimizationLevel: "all",
    });
    backend = "webgpu";
  } catch (error) {
    // Surface the fallback so the UI can warn the user that analysis is slower,
    // but keep the original error for the console log.
    console.warn("WebGPU 初始化失败，回退到 CPU (wasm):", error);
    sendInfo("WebGPU 不可用，已回退到 CPU，分析会更慢");
    session = await ort.InferenceSession.create(modelBytes.buffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    backend = "wasm";
  }
  activeModel = { ...meta, backend };
  configureEvalShape();
  await warmupSession();
  sessionCache.set(activeModel.id, { session, modelConfig, activeModel });
  game = new GameSession(modelConfig, activeModel);
}

/* Pre-compile the WebGPU pipelines for both batch shapes (1 and evalPad) on a
 * throwaway empty board so the user's first move doesn't pay shader-compile
 * latency. Best-effort: any failure here is non-fatal. */
async function warmupSession() {
  if (!padEnabled) return;
  try {
    const probe = GomokuState.new(
      Number(modelConfig.board_size ?? 10),
      Number(modelConfig.win_length ?? 5),
    );
    await evaluateBatch([probe]);
    if (evalPad > 1) {
      await evaluateBatch(new Array(evalPad).fill(probe));
    }
  } catch (error) {
    /* warmup is best-effort; real inference will compile on demand */
  }
}

/* Returns the assembled model bytes, preferring a verified IndexedDB hit keyed
 * on the manifest sha256, otherwise downloading the chunks concurrently,
 * verifying integrity, and persisting for next time. */
async function assembleModelBytes(manifest, manifestUrl, meta) {
  const expectedSha = manifest.sha256 || null;

  if (expectedSha) {
    const cached = await idbGetModel(expectedSha);
    if (cached) {
      const bytes = cached instanceof Uint8Array ? cached : new Uint8Array(cached);
      sendProgress(`${meta.label} 已缓存`, bytes.byteLength, bytes.byteLength);
      return bytes;
    }
  }

  const chunks = manifest.chunks || [];
  const total = manifest.bytes || chunks.reduce((sum, chunk) => sum + chunk.bytes, 0);
  const loadedPerChunk = new Array(chunks.length).fill(0);
  const reportProgress = () => {
    const loaded = loadedPerChunk.reduce((sum, value) => sum + value, 0);
    sendProgress(`${meta.label} ${chunks.length} 分片`, loaded, total);
  };

  // Fetch concurrently but keep results ordered for reassembly.
  const parts = await Promise.all(
    chunks.map(async (chunk, index) => {
      const url = new URL(chunk.file, manifestUrl);
      const bytes = await fetchBytes(url, (partLoaded) => {
        loadedPerChunk[index] = partLoaded;
        reportProgress();
      });
      loadedPerChunk[index] = bytes.byteLength;
      reportProgress();
      if (chunk.bytes && bytes.byteLength !== chunk.bytes) {
        throw new Error("下载损坏，请重试");
      }
      return bytes;
    }),
  );

  const totalBytes = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const modelBytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const part of parts) {
    modelBytes.set(part, offset);
    offset += part.byteLength;
  }

  if (expectedSha && crypto && crypto.subtle) {
    const actualSha = await sha256Hex(modelBytes);
    if (actualSha !== expectedSha) {
      throw new Error("下载损坏，请重试");
    }
  }

  if (expectedSha) await idbPutModel(expectedSha, modelBytes);
  return modelBytes;
}

self.addEventListener("message", async (event) => {
  const { id, type, payload = {} } = event.data;
  try {
    let result;
    if (type === "init") {
      await loadModel(payload.modelId);
      result = { ok: true, model: activeModel };
    } else {
      if (!game) throw new Error("模型尚未加载");
      if (type === "newGame") result = await game.newGame(payload.human, payload.simulations);
      else if (type === "move") result = await game.humanMove(payload.row, payload.col, payload.simulations);
      else if (type === "undo") result = game.undo();
      else if (type === "analyze") result = await game.analyze(payload.simulations);
      else if (type === "evaluate") result = await game.evaluateCurrent();
      else throw new Error(`未知命令: ${type}`);
    }
    self.postMessage({ id, payload: result });
  } catch (error) {
    self.postMessage({ id, error: error && error.message ? error.message : String(error) });
  }
});
