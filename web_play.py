from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

from .game import GomokuState
from .mcts import MCTS, MCTSConfig, visit_count_policy
from .model import PolicyValueNet, build_model_from_config


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this Python environment cannot see it")
    return device


def load_model(checkpoint: Path, device: str) -> tuple[PolicyValueNet, dict]:
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    cfg = payload.get("config", {})
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg


class GameSession:
    def __init__(
        self,
        model: PolicyValueNet,
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
        self.last_ai_policy: list[dict] = []
        self.policy_source = "none"
        self.policy_player: int | None = None
        self.undo_stack: list[tuple[GomokuState, list[dict], list[dict], str, int | None]] = []

    def new_game(self, human: str = "black", simulations: int | None = None) -> dict:
        self.state = GomokuState.new(
            size=int(self.cfg.get("board_size", 10)),
            win_length=int(self.cfg.get("win_length", 5)),
        )
        self.human_player = 1 if human == "black" else -1
        self.set_simulations(int(simulations or self.default_simulations))
        self.history = []
        self.last_ai_policy = []
        self.policy_source = "none"
        self.policy_player = None
        self.undo_stack = []
        if self.state.current_player != self.human_player:
            self._ai_move()
        return self.snapshot()

    def set_simulations(self, simulations: int) -> None:
        self.simulations = max(1, int(simulations))

    def human_move(self, row: int, col: int, simulations: int | None = None) -> dict:
        if simulations is not None:
            self.set_simulations(simulations)
        if self.state.is_terminal:
            raise ValueError("game is already over")
        if self.state.current_player != self.human_player:
            raise ValueError("it is not the human player's turn")
        action = self.state.coord_to_action(row, col)
        if action not in set(map(int, self.state.legal_actions())):
            raise ValueError("that point is already occupied")
        self.undo_stack.append(
            (
                self.state,
                [dict(item) for item in self.history],
                [dict(item) for item in self.last_ai_policy],
                self.policy_source,
                self.policy_player,
            )
        )
        self.state = self.state.apply(action)
        self.last_ai_policy = []
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
        state, history, policy, policy_source, policy_player = self.undo_stack.pop()
        self.state = state
        self.history = history
        self.last_ai_policy = policy
        self.policy_source = policy_source
        self.policy_player = policy_player
        return self.snapshot()

    def analyze(self, simulations: int | None = None) -> dict:
        if simulations is not None:
            self.set_simulations(simulations)
        if self.state.is_terminal:
            raise ValueError("game is already over")
        _, policy = self._search_policy()
        self.last_ai_policy = policy
        self.policy_source = "analysis"
        self.policy_player = self.state.current_player
        return self.snapshot()

    def _ai_move(self) -> None:
        player = self.state.current_player
        action, policy = self._search_policy()
        if action < 0:
            raise ValueError("AI could not find a legal move")
        self.last_ai_policy = policy
        self.policy_source = "ai_move"
        self.policy_player = player
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

    def _search_policy(self) -> tuple[int, list[dict]]:
        mcts = MCTS(
            self.model,
            MCTSConfig(
                simulations=self.simulations,
                c_puct=float(self.cfg.get("mcts_c_puct", 1.5)),
                dirichlet_alpha=float(self.cfg.get("mcts_dirichlet_alpha", 0.3)),
                dirichlet_fraction=float(self.cfg.get("mcts_dirichlet_fraction", 0.25)),
                eval_batch_size=min(
                    int(self.cfg.get("mcts_batch_size", 16)),
                    max(1, self.simulations),
                ),
                amp_dtype=str(
                    self.cfg.get(
                        "mcts_amp_dtype",
                        (
                            str(self.cfg.get("amp_dtype", "bf16"))
                            if bool(self.cfg.get("amp", True))
                            else "none"
                        ),
                    )
                ),
            ),
            device=self.device,
        )
        root = mcts.search(self.state, add_exploration_noise=False)
        policy = visit_count_policy(root, self.state.action_size, temperature=0.0)
        action = int(policy.argmax())
        if not root.children:
            return -1, []
        ranked = sorted(
            ((int(action_id), float(child.visit_count)) for action_id, child in root.children.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:8]
        total_visits = sum(float(child.visit_count) for child in root.children.values()) or 1.0
        policy_rows = [
            {
                "row": self.state.action_to_coord(action_id)[0],
                "col": self.state.action_to_coord(action_id)[1],
                "visits": visits,
                "share": visits / total_visits,
                "selected": action_id == action,
            }
            for action_id, visits in ranked
        ]
        return action, policy_rows

    def snapshot(self) -> dict:
        return {
            "board": self.state.board.tolist(),
            "size": self.state.size,
            "currentPlayer": self.state.current_player,
            "humanPlayer": self.human_player,
            "winner": self.state.winner,
            "movesPlayed": self.state.moves_played,
            "lastMove": self.state.last_move,
            "history": self.history[-20:],
            "aiPolicy": self.last_ai_policy,
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
            if self.state.winner == self.human_player:
                return "You win"
            return "AI wins"
        if self.state.current_player == self.human_player:
            return "Your turn"
        return "AI turn"


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
        if self.static_root not in requested.parents and requested != self.static_root:
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

    def log_message(self, format: str, *args: object) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Gomoku web UI.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    WebPlayHandler.session = GameSession(model, cfg, device, args.simulations)
    WebPlayHandler.static_root = Path(__file__).parent / "web"
    server = ThreadingHTTPServer((args.host, args.port), WebPlayHandler)
    print(f"serving http://{args.host}:{args.port}")
    print(f"device={device} checkpoint={args.checkpoint}")
    server.serve_forever()


if __name__ == "__main__":
    main()
