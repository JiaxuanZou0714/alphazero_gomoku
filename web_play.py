from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .game import GomokuState
from .mcts import MCTS, MCTSConfig
from .utils import load_model, resolve_device


class GameSession:
    def __init__(
        self,
        model,
        cfg: dict,
        device: str,
        simulations: int,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.default_simulations = simulations
        self.lock = threading.Lock()
        self.state = GomokuState.new(
            size=int(cfg.get("board_size", 10)),
            win_length=int(cfg.get("win_length", 5)),
        )
        self.human_player = 1
        self.simulations = simulations
        self.history: list[dict] = []
        self.last_analysis: dict = {}
        self.policy_source = "none"
        self.policy_player: int | None = None
        self.eval_history: list[dict] = []
        self.undo_stack: list[tuple] = []
        # reuse a single MCTS object; its model ref is stable
        self._mcts = self._make_mcts(simulations)

    def _make_mcts(self, simulations: int) -> MCTS:
        amp_dtype = str(
            self.cfg.get(
                "mcts_amp_dtype",
                str(self.cfg.get("amp_dtype", "bf16"))
                if bool(self.cfg.get("amp", True))
                else "none",
            )
        )
        return MCTS(
            self.model,
            MCTSConfig(
                simulations=simulations,
                c_puct=float(self.cfg.get("mcts_c_puct", 1.5)),
                dirichlet_alpha=float(self.cfg.get("mcts_dirichlet_alpha", 0.3)),
                dirichlet_fraction=float(self.cfg.get("mcts_dirichlet_fraction", 0.25)),
                eval_batch_size=min(
                    int(self.cfg.get("mcts_batch_size", 16)),
                    max(1, simulations),
                ),
                amp_dtype=amp_dtype,
                fpu_reduction=float(self.cfg.get("mcts_fpu_reduction", 0.0) or 0.0),
            ),
            device=self.device,
        )

    def set_simulations(self, simulations: int) -> None:
        simulations = max(1, int(simulations))
        if simulations != self.simulations:
            self.simulations = simulations
            self._mcts = self._make_mcts(simulations)

    def new_game(self, human: str = "black", simulations: int | None = None) -> dict:
        self.state = GomokuState.new(
            size=int(self.cfg.get("board_size", 10)),
            win_length=int(self.cfg.get("win_length", 5)),
        )
        self.human_player = 1 if human == "black" else -1
        self.set_simulations(int(simulations or self.default_simulations))
        self.history = []
        self.last_analysis = {}
        self.policy_source = "none"
        self.policy_player = None
        self.eval_history = []
        self.undo_stack = []
        if self.state.current_player != self.human_player:
            self._ai_move()
        return self.snapshot()

    def human_move(self, row: int, col: int, simulations: int | None = None) -> dict:
        if simulations is not None:
            self.set_simulations(simulations)
        if self.state.is_terminal:
            raise ValueError("game is already over")
        if self.state.current_player != self.human_player:
            raise ValueError("it is not the human player's turn")
        action = self.state.coord_to_action(row, col)
        if not self.state.legal_mask()[action]:
            raise ValueError("that point is already occupied")
        self.undo_stack.append(
            (
                self.state,
                list(self.history),
                dict(self.last_analysis),
                self.policy_source,
                self.policy_player,
                list(self.eval_history),
            )
        )
        self.state = self.state.apply(action)
        self.last_analysis = {}
        self.policy_source = "none"
        self.policy_player = None
        self.history.append(
            {
                "player": "black" if -self.state.current_player == 1 else "white",
                "source": "human",
                "row": row,
                "col": col,
            }
        )
        if not self.state.is_terminal:
            self._ai_move()
        return self.snapshot()

    def undo(self) -> dict:
        if not self.undo_stack:
            raise ValueError("nothing to undo")
        state, history, analysis, policy_source, policy_player, evals = self.undo_stack.pop()
        self.state = state
        self.history = history
        self.last_analysis = analysis
        self.policy_source = policy_source
        self.policy_player = policy_player
        self.eval_history = evals
        return self.snapshot()

    def analyze(self, simulations: int | None = None) -> dict:
        if simulations is not None:
            self.set_simulations(simulations)
        if self.state.is_terminal:
            raise ValueError("game is already over")
        _, analysis = self._search_policy()
        self.last_analysis = analysis
        self.policy_source = "analysis"
        self.policy_player = self.state.current_player
        self._record_eval(analysis)
        return self.snapshot()

    def _ai_move(self) -> None:
        player = self.state.current_player
        action, analysis = self._search_policy()
        if action < 0:
            raise ValueError("AI could not find a legal move")
        self.last_analysis = analysis
        self.policy_source = "ai_move"
        self.policy_player = player
        self._record_eval(analysis)
        row, col = self.state.action_to_coord(action)
        self.state = self.state.apply(action)
        self.history.append(
            {
                "player": "black" if player == 1 else "white",
                "source": "ai",
                "row": row,
                "col": col,
            }
        )

    def _record_eval(self, analysis: dict) -> None:
        """Track black's win probability over the game for the eval chart."""
        if not analysis:
            return
        black_win = (
            analysis["winProb"] if analysis["player"] == 1 else 1.0 - analysis["winProb"]
        )
        move = self.state.moves_played
        self.eval_history = [e for e in self.eval_history if e["move"] != move]
        self.eval_history.append({"move": move, "blackWinProb": black_win})

    def _search_policy(self) -> tuple[int, dict]:
        """Run a search and export its internals for visualisation.

        The analysis describes the position *before* the chosen move is played:
        network priors (intuition), MCTS visit distribution (search result),
        per-move Q values, root value and the principal variation.
        """
        start = time.monotonic()
        root = self._mcts.search(self.state, add_exploration_noise=False)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        if not root.children:
            return -1, {}
        size = self.state.size
        action_size = self.state.action_size
        items = list(root.children.items())
        total_visits = sum(child.visit_count for _, child in items) or 1

        visit_map = [0.0] * action_size
        prior_map = [0.0] * action_size
        q_map: list[float | None] = [None] * action_size
        for a, child in items:
            visit_map[a] = child.visit_count / total_visits
            prior_map[a] = float(child.raw_prior)
            if child.visit_count > 0:
                q_map[a] = float(-child.value)  # mover's perspective

        action = max(items, key=lambda kv: kv[1].visit_count)[0]
        ranked = sorted(items, key=lambda kv: kv[1].visit_count, reverse=True)[:8]
        candidates = [
            {
                "row": a // size,
                "col": a % size,
                "visits": int(child.visit_count),
                "share": child.visit_count / total_visits,
                "prior": float(child.raw_prior),
                "q": float(-child.value) if child.visit_count > 0 else None,
                "selected": a == action,
            }
            for a, child in ranked
        ]

        pv = []
        node = root
        while node.expanded and len(pv) < 8:
            pv_action, pv_child = max(
                node.children.items(), key=lambda kv: kv[1].visit_count
            )
            if pv_child.visit_count == 0:
                break
            pv.append({"row": pv_action // size, "col": pv_action % size})
            node = pv_child

        analysis = {
            "player": self.state.current_player,
            "moveNumber": self.state.moves_played,
            "rootValue": float(root.value),
            "winProb": (float(root.value) + 1.0) / 2.0,
            "simulations": int(root.visit_count),
            "elapsedMs": elapsed_ms,
            "visitMap": visit_map,
            "priorMap": prior_map,
            "qMap": q_map,
            "candidates": candidates,
            "pv": pv,
        }
        return action, analysis

    def snapshot(self) -> dict:
        return {
            "board": self.state.board.tolist(),
            "size": self.state.size,
            "currentPlayer": self.state.current_player,
            "humanPlayer": self.human_player,
            "winner": self.state.winner,
            "movesPlayed": self.state.moves_played,
            "lastMove": self.state.last_move,
            "history": self.history[-30:],
            "analysis": self.last_analysis,
            "evalHistory": sorted(self.eval_history, key=lambda e: e["move"]),
            "policySource": self.policy_source,
            "policyPlayer": self.policy_player,
            "canUndo": bool(self.undo_stack),
            "simulations": self.simulations,
            "device": self.device,
            "status": self._status_text(),
        }

    def _status_text(self) -> str:
        if self.state.winner == 0:
            return "Draw"
        if self.state.winner is not None:
            return "You win" if self.state.winner == self.human_player else "AI wins"
        return "Your turn" if self.state.current_player == self.human_player else "AI turn"


class WebPlayHandler(BaseHTTPRequestHandler):
    session: GameSession
    static_root: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            with self.session.lock:
                self._send_json(self.session.snapshot())
            return
        if parsed.path == "/api/health":
            self._send_json({"ok": True, "device": self.session.device})
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self._read_json()
            with self.session.lock:
                if parsed.path == "/api/new":
                    result = self.session.new_game(
                        human=str(body.get("human", "black")),
                        simulations=int(body.get("simulations", self.session.default_simulations)),
                    )
                elif parsed.path == "/api/move":
                    result = self.session.human_move(
                        int(body["row"]),
                        int(body["col"]),
                        int(body["simulations"]) if "simulations" in body else None,
                    )
                elif parsed.path == "/api/undo":
                    result = self.session.undo()
                elif parsed.path == "/api/analyze":
                    result = self.session.analyze(
                        int(body["simulations"]) if "simulations" in body else None
                    )
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "not found")
                    return
            self._send_json(result)
        except (KeyError, ValueError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _serve_static(self, path: str) -> None:
        if path in ("", "/"):
            path = "/index.html"
        requested = (self.static_root / path.lstrip("/")).resolve()
        if not requested.is_relative_to(self.static_root):
            self._send_error(HTTPStatus.FORBIDDEN, "forbidden")
            return
        if not requested.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
        data = requested.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, data: dict) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        payload = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        # only surface 4xx/5xx to avoid spamming normal requests
        if args and str(args[1]).startswith(("4", "5")):
            import sys
            print(fmt % args, file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Gomoku web UI.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    WebPlayHandler.session = GameSession(model, cfg, device, args.simulations)
    WebPlayHandler.static_root = (Path(__file__).parent / "web").resolve()
    server = ThreadingHTTPServer((args.host, args.port), WebPlayHandler)
    print(f"serving http://{args.host}:{args.port}")
    print(f"device={device} checkpoint={args.checkpoint}")
    server.serve_forever()


if __name__ == "__main__":
    main()
