from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import multiprocessing as mp
import queue
import random
import shutil
import tempfile
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .game import GomokuState
from .mcts import MCTS, MCTSConfig, visit_count_policy
from .model import PolicyValueNet, build_model_from_config, model_kwargs_from_config
from .torch_compat import tensor_from_array

Example = tuple[np.ndarray, np.ndarray, float]


@dataclass
class TrainConfig:
    preset: str = "local"
    board_size: int = 10
    win_length: int = 5
    iterations: int = 5
    games_per_iteration: int = 4
    simulations: int = 64
    mcts_batch_size: int = 1
    epochs: int = 2
    train_steps_per_iteration: int = 0
    batch_size: int = 64
    replay_size: int = 10_000
    learning_rate: float = 1.0e-3
    min_learning_rate: float = 0.0
    lr_schedule: str = "constant"
    warmup_iterations: int = 0
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 5.0
    max_loss: float = 1_000.0
    temperature_moves: int = 20
    mcts_c_puct: float = 1.5
    mcts_dirichlet_alpha: float = 0.3
    mcts_dirichlet_fraction: float = 0.25
    seed: int = 7
    channels: int = 64
    residual_blocks: int = 4
    policy_channels: int = 2
    value_channels: int = 1
    value_hidden: int = 128
    use_se: bool = False
    se_ratio: int = 16
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    augment_symmetries: bool = True
    replay_path: str = ""
    replay_save_interval: int = 5
    eval_interval: int = 0
    eval_games: int = 0
    eval_simulations: int = 128
    promotion_threshold: float = 0.55
    gate_evaluation: bool = False
    metrics_path: str = ""
    checkpoint_dir: str = "outputs/checkpoints"
    resume: str = ""
    device: str = "auto"
    self_play_workers: int = 1
    self_play_devices: str = "auto"
    data_parallel: bool = False
    compile_model: bool = False
    amp: bool = True
    amp_dtype: str = "none"
    mcts_amp_dtype: str = "bf16"


@dataclass
class TrainStats:
    loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    policy_kl: float = 0.0
    target_entropy: float = 0.0
    pred_entropy: float = 0.0
    policy_top1: float = 0.0
    value_mae: float = 0.0
    value_acc: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    skipped: int = 0
    value_clamps: int = 0
    batches: int = 0
    optimizer_steps: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def enable_fast_math() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass


def limit_worker_threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this Python environment cannot see it")
    return device


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def load_torch(path: Path | str, *, map_location: str | torch.device = "cpu") -> object:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def make_grad_scaler(enabled: bool, amp_dtype: str) -> object | None:
    if not enabled or amp_dtype != "fp16":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device: str, enabled: bool, amp_dtype: str) -> object:
    if not enabled or not device.startswith("cuda") or amp_dtype == "none":
        return nullcontext()
    if amp_dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp_dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"unknown amp_dtype: {amp_dtype}")


def scheduled_learning_rate(cfg: TrainConfig, iteration: int, end_iteration: int) -> float:
    if cfg.warmup_iterations > 0 and iteration <= cfg.warmup_iterations:
        return cfg.learning_rate * iteration / cfg.warmup_iterations
    if cfg.lr_schedule == "constant":
        return cfg.learning_rate
    if cfg.lr_schedule != "cosine":
        raise ValueError(f"unknown lr_schedule: {cfg.lr_schedule}")
    decay_start = max(1, cfg.warmup_iterations)
    decay_span = max(1, end_iteration - decay_start)
    progress = min(1.0, max(0.0, (iteration - decay_start) / decay_span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + (cfg.learning_rate - cfg.min_learning_rate) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def append_metrics(path: str, row: dict[str, object]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def preset_config(name: str) -> TrainConfig:
    cfg = TrainConfig(preset=name)
    if name == "local":
        cfg.augment_symmetries = True
        return cfg
    if name == "a100-4":
        cfg.iterations = 100
        cfg.games_per_iteration = 96
        cfg.simulations = 384
        cfg.mcts_batch_size = 48
        cfg.epochs = 3
        cfg.train_steps_per_iteration = 96
        cfg.batch_size = 4096
        cfg.replay_size = 500_000
        cfg.learning_rate = 2.0e-4
        cfg.min_learning_rate = 2.0e-5
        cfg.lr_schedule = "cosine"
        cfg.warmup_iterations = 5
        cfg.weight_decay = 1.0e-4
        cfg.temperature_moves = 12
        cfg.mcts_c_puct = 1.25
        cfg.mcts_dirichlet_alpha = 0.15
        cfg.channels = 192
        cfg.residual_blocks = 12
        cfg.policy_channels = 16
        cfg.value_channels = 8
        cfg.value_hidden = 512
        cfg.use_se = True
        cfg.checkpoint_dir = "alphazero_gomoku/outputs/checkpoints/a100-4"
        cfg.replay_path = "alphazero_gomoku/outputs/replay/a100-4_replay.pt"
        cfg.metrics_path = "alphazero_gomoku/outputs/metrics/a100-4.jsonl"
        cfg.self_play_workers = 16
        cfg.self_play_devices = "auto"
        cfg.data_parallel = False
        cfg.eval_interval = 5
        cfg.eval_games = 16
        cfg.eval_simulations = 128
        cfg.promotion_threshold = 0.55
        return cfg
    if name == "a100-prod":
        cfg.iterations = 300
        cfg.games_per_iteration = 256
        cfg.simulations = 384
        cfg.mcts_batch_size = 64
        cfg.epochs = 2
        cfg.train_steps_per_iteration = 192
        cfg.batch_size = 8192
        cfg.replay_size = 1_500_000
        cfg.learning_rate = 1.5e-4
        cfg.min_learning_rate = 1.5e-5
        cfg.lr_schedule = "cosine"
        cfg.warmup_iterations = 10
        cfg.weight_decay = 1.0e-4
        cfg.temperature_moves = 12
        cfg.mcts_c_puct = 1.25
        cfg.mcts_dirichlet_alpha = 0.15
        cfg.channels = 256
        cfg.residual_blocks = 16
        cfg.policy_channels = 32
        cfg.value_channels = 16
        cfg.value_hidden = 768
        cfg.use_se = True
        cfg.checkpoint_dir = "alphazero_gomoku/outputs/checkpoints/a100-prod"
        cfg.replay_path = "alphazero_gomoku/outputs/replay/a100-prod_replay.pt"
        cfg.metrics_path = "alphazero_gomoku/outputs/metrics/a100-prod.jsonl"
        cfg.self_play_workers = 16
        cfg.self_play_devices = "auto"
        cfg.data_parallel = False
        cfg.eval_interval = 5
        cfg.eval_games = 32
        cfg.eval_simulations = 192
        cfg.promotion_threshold = 0.55
        return cfg
    if name == "a100-fast":
        cfg.iterations = 40
        cfg.games_per_iteration = 128
        cfg.simulations = 96
        cfg.mcts_batch_size = 24
        cfg.epochs = 2
        cfg.train_steps_per_iteration = 64
        cfg.batch_size = 4096
        cfg.replay_size = 300_000
        cfg.learning_rate = 2.5e-4
        cfg.min_learning_rate = 2.5e-5
        cfg.lr_schedule = "cosine"
        cfg.warmup_iterations = 3
        cfg.temperature_moves = 12
        cfg.mcts_c_puct = 1.25
        cfg.mcts_dirichlet_alpha = 0.15
        cfg.channels = 96
        cfg.residual_blocks = 6
        cfg.policy_channels = 8
        cfg.value_channels = 4
        cfg.value_hidden = 256
        cfg.checkpoint_dir = "alphazero_gomoku/outputs/checkpoints/a100-fast"
        cfg.replay_path = "alphazero_gomoku/outputs/replay/a100-fast_replay.pt"
        cfg.metrics_path = "alphazero_gomoku/outputs/metrics/a100-fast.jsonl"
        cfg.self_play_workers = 16
        cfg.self_play_devices = "auto"
        cfg.data_parallel = False
        return cfg
    if name == "a100-turbo":
        cfg.iterations = 12
        cfg.games_per_iteration = 128
        cfg.simulations = 32
        cfg.mcts_batch_size = 16
        cfg.epochs = 2
        cfg.train_steps_per_iteration = 16
        cfg.batch_size = 8192
        cfg.replay_size = 150_000
        cfg.learning_rate = 3.0e-4
        cfg.min_learning_rate = 3.0e-5
        cfg.lr_schedule = "cosine"
        cfg.warmup_iterations = 2
        cfg.temperature_moves = 10
        cfg.mcts_c_puct = 1.25
        cfg.mcts_dirichlet_alpha = 0.15
        cfg.channels = 64
        cfg.residual_blocks = 4
        cfg.policy_channels = 4
        cfg.value_channels = 2
        cfg.value_hidden = 192
        cfg.checkpoint_dir = "alphazero_gomoku/outputs/checkpoints/a100-turbo"
        cfg.replay_path = "alphazero_gomoku/outputs/replay/a100-turbo_replay.pt"
        cfg.metrics_path = "alphazero_gomoku/outputs/metrics/a100-turbo.jsonl"
        cfg.self_play_workers = 16
        cfg.self_play_devices = "auto"
        cfg.data_parallel = False
        return cfg
    raise ValueError(f"unknown preset: {name}")


def split_devices(device_spec: str, fallback_device: str) -> list[str]:
    if device_spec == "auto":
        if torch.cuda.is_available():
            return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        return [fallback_device]
    devices = [device.strip() for device in device_spec.split(",") if device.strip()]
    return devices or [fallback_device]


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in unwrap_model(model).state_dict().items()
    }


def make_model(cfg: TrainConfig) -> PolicyValueNet:
    return build_model_from_config(asdict(cfg))


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def architecture_from_config(cfg: TrainConfig) -> dict[str, object]:
    return model_kwargs_from_config(asdict(cfg))


def load_resume_checkpoint(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> int:
    if not cfg.resume:
        return 0

    path = Path(cfg.resume)
    checkpoint = load_torch(path, map_location=cfg.device)
    checkpoint_cfg = checkpoint.get("config", {})
    expected = architecture_from_config(cfg)
    found = model_kwargs_from_config(checkpoint_cfg)
    if found != expected:
        raise RuntimeError(
            "resume checkpoint architecture does not match current config: "
            f"checkpoint={found} current={expected}"
        )

    model.load_state_dict(checkpoint["model"])
    optimizer_loaded = False
    if "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            move_optimizer_state(optimizer, cfg.device)
            optimizer_loaded = True
        except ValueError as exc:
            print(f"resume_warning optimizer_state_not_loaded error={exc}", flush=True)

    iteration = int(checkpoint.get("iteration", 0))
    print(
        f"resume_loaded path={path} checkpoint_iteration={iteration} "
        f"optimizer_loaded={optimizer_loaded}",
        flush=True,
    )
    return iteration


def save_replay(replay: deque[Example], path: str) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(replay), target)
    print(f"replay_saved path={target} examples={len(replay)}", flush=True)


def load_replay(cfg: TrainConfig) -> deque[Example]:
    replay: deque[Example] = deque(maxlen=cfg.replay_size)
    if not cfg.replay_path:
        return replay
    path = Path(cfg.replay_path)
    if not path.exists():
        return replay
    loaded = load_torch(path, map_location="cpu")
    replay.extend(loaded[-cfg.replay_size:])
    print(f"replay_loaded path={path} examples={len(replay)}", flush=True)
    return replay


def sample_action(policy: np.ndarray) -> int:
    total = float(policy.sum())
    if total <= 0:
        raise ValueError("cannot sample from an empty policy")
    return int(np.random.choice(len(policy), p=policy / total))


def policy_entropy(policy: np.ndarray) -> float:
    probs = policy[policy > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs + 1.0e-12)).sum())


def augment_examples(examples: list[Example], board_size: int) -> list[Example]:
    """Apply all 8 dihedral symmetries to each example.

    Vectorised: stacks all examples into two big arrays, applies each of the 8
    transforms with a single numpy call, then returns a flat list.  This is
    ~5-10x faster than the previous per-example Python loop.
    """
    if not examples:
        return []
    n = len(examples)
    # (n, 2, H, W) and (n, H*W)
    states = np.stack([e[0] for e in examples]).astype(np.float32)   # (n,2,H,W)
    policies = np.stack([e[1] for e in examples]).astype(np.float32) # (n,H*W)
    values = [e[2] for e in examples]

    pol2d = policies.reshape(n, board_size, board_size)
    augmented: list[Example] = []
    for k in range(8):
        # spatial transform (axes 2,3 for states; axes 1,2 for pol2d)
        if k < 4:
            ts = np.rot90(states,  k, axes=(2, 3))
            tp = np.rot90(pol2d,   k, axes=(1, 2))
        else:
            ts = np.flip(np.rot90(states,  k - 4, axes=(2, 3)), axis=3)
            tp = np.flip(np.rot90(pol2d,   k - 4, axes=(1, 2)), axis=2)
        ts = np.ascontiguousarray(ts, dtype=np.float32)
        tp = np.ascontiguousarray(tp.reshape(n, -1), dtype=np.float32)
        for i, v in enumerate(values):
            augmented.append((ts[i], tp[i], v))
    return augmented


def play_self_game(
    model: PolicyValueNet,
    cfg: TrainConfig,
    mcts_cfg: MCTSConfig,
) -> tuple[list[Example], dict[str, float]]:
    mcts = MCTS(model, mcts_cfg, device=cfg.device)
    state = GomokuState.new(size=cfg.board_size, win_length=cfg.win_length)
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    entropies: list[float] = []

    while not state.is_terminal:
        temperature = 1.0 if state.moves_played < cfg.temperature_moves else 0.0
        root = mcts.search(state, add_exploration_noise=True)
        policy = visit_count_policy(root, state.action_size, temperature)
        if policy.sum() <= 0:
            legal = state.legal_actions()
            policy[legal] = 1.0 / len(legal)
        policy = policy.astype(np.float32)
        entropies.append(policy_entropy(policy))
        history.append((state.encode(), policy, state.current_player))
        state = state.apply(sample_action(policy))

    examples: list[Example] = []
    for encoded, policy, player in history:
        if state.winner == 0:
            value = 0.0
        else:
            value = 1.0 if state.winner == player else -1.0
        examples.append((encoded.astype(np.float32), policy, value))
    stats = {
        "policy_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "winner": float(state.winner if state.winner is not None else 0),
    }
    return examples, stats


def play_self_games_worker(
    cfg_data: dict,
    mcts_data: dict,
    model_state_path: str,
    games: int,
    device: str,
    worker_id: int,
    iteration: int = 0,
    progress_queue: object | None = None,
) -> tuple[list[Example], list[int], dict[str, float], str]:
    limit_worker_threads()
    enable_fast_math()
    cfg = TrainConfig(**cfg_data)
    cfg.device = device
    set_seed(cfg.seed + worker_id * 10_000 + iteration)
    model = make_model(cfg).to(device)
    model_state = load_torch(model_state_path, map_location="cpu")
    model.load_state_dict(model_state)
    mcts_cfg = MCTSConfig(**mcts_data)

    examples: list[Example] = []
    lengths: list[int] = []
    entropies: list[float] = []
    winners: list[float] = []
    worker_start = time.monotonic()
    for game_index in range(1, games + 1):
        game_start = time.monotonic()
        game_examples, game_stats = play_self_game(model, cfg, mcts_cfg)
        examples.extend(game_examples)
        lengths.append(len(game_examples))
        entropies.append(game_stats["policy_entropy"])
        winners.append(game_stats["winner"])
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "selfplay_game",
                    "iteration": iteration,
                    "worker_id": worker_id,
                    "device": device,
                    "game_index": game_index,
                    "games": games,
                    "moves": len(game_examples),
                    "examples": len(game_examples),
                    "policy_entropy": game_stats["policy_entropy"],
                    "game_seconds": time.monotonic() - game_start,
                    "worker_seconds": time.monotonic() - worker_start,
                }
            )
    stats = {
        "policy_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "black_win_rate": float(np.mean([winner == 1 for winner in winners])) if winners else 0.0,
        "white_win_rate": float(np.mean([winner == -1 for winner in winners])) if winners else 0.0,
        "draw_rate": float(np.mean([winner == 0 for winner in winners])) if winners else 0.0,
    }
    return examples, lengths, stats, device


def distribute_games(total_games: int, worker_count: int) -> list[int]:
    base = total_games // worker_count
    remainder = total_games % worker_count
    return [base + (1 if index < remainder else 0) for index in range(worker_count)]


def merge_stats(stats: list[dict[str, float]]) -> dict[str, float]:
    if not stats:
        return {
            "policy_entropy": 0.0,
            "black_win_rate": 0.0,
            "white_win_rate": 0.0,
            "draw_rate": 0.0,
        }
    if "winner" in stats[0]:
        winners = [item["winner"] for item in stats]
        return {
            "policy_entropy": float(np.mean([item["policy_entropy"] for item in stats])),
            "black_win_rate": float(np.mean([winner == 1 for winner in winners])),
            "white_win_rate": float(np.mean([winner == -1 for winner in winners])),
            "draw_rate": float(np.mean([winner == 0 for winner in winners])),
        }
    keys = stats[0].keys()
    return {key: float(np.mean([item[key] for item in stats])) for key in keys}


def collect_self_play_examples(
    model: torch.nn.Module,
    cfg: TrainConfig,
    mcts_cfg: MCTSConfig,
    iteration: int,
) -> tuple[list[Example], list[int], dict[str, float]]:
    if cfg.self_play_workers <= 1:
        examples: list[Example] = []
        lengths: list[int] = []
        stats: list[dict[str, float]] = []
        start = time.monotonic()
        for game_index in range(1, cfg.games_per_iteration + 1):
            game_start = time.monotonic()
            game_examples, game_stats = play_self_game(unwrap_model(model), cfg, mcts_cfg)
            examples.extend(game_examples)
            lengths.append(len(game_examples))
            stats.append(game_stats)
            elapsed = time.monotonic() - start
            avg = elapsed / game_index
            eta = avg * (cfg.games_per_iteration - game_index)
            print(
                f"selfplay_progress iter={iteration} games={game_index}/"
                f"{cfg.games_per_iteration} elapsed={format_duration(elapsed)} "
                f"eta={format_duration(eta)} last_moves={len(game_examples)} "
                f"policy_entropy={game_stats['policy_entropy']:.3f} "
                f"last_time={format_duration(time.monotonic() - game_start)} "
                f"examples={len(examples)}",
                flush=True,
            )
        return examples, lengths, merge_stats(stats)

    devices = split_devices(cfg.self_play_devices, cfg.device)
    worker_devices = [
        devices[index % len(devices)] for index in range(cfg.self_play_workers)
    ]
    game_counts = distribute_games(cfg.games_per_iteration, cfg.self_play_workers)
    jobs = [
        (worker_id, device, games)
        for worker_id, (device, games) in enumerate(zip(worker_devices, game_counts))
        if games > 0
    ]

    all_examples: list[Example] = []
    all_lengths: list[int] = []
    all_stats: list[dict[str, float]] = []
    cfg_data = asdict(cfg)
    mcts_data = asdict(mcts_cfg)
    context = mp.get_context("spawn")
    total_games = sum(game_counts)
    done_games = 0
    start = time.monotonic()
    last_heartbeat = start
    with tempfile.TemporaryDirectory(prefix="az_selfplay_") as temp_dir, context.Manager() as manager:
        model_state_path = str(Path(temp_dir) / f"iter_{iteration:04d}_model.pt")
        torch.save(cpu_state_dict(model), model_state_path)
        progress_queue = manager.Queue()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(jobs), mp_context=context
        ) as executor:
            futures = [
                executor.submit(
                    play_self_games_worker,
                    cfg_data,
                    mcts_data,
                    model_state_path,
                    games,
                    device,
                    worker_id,
                    iteration,
                    progress_queue,
                )
                for worker_id, device, games in jobs
            ]
            pending = set(futures)
            while pending:
                try:
                    event = progress_queue.get(timeout=1.0)
                except queue.Empty:
                    event = None

                now = time.monotonic()
                if event is not None and event.get("type") == "selfplay_game":
                    done_games += 1
                    elapsed = now - start
                    avg = elapsed / max(1, done_games)
                    eta = avg * (total_games - done_games)
                    print(
                        f"selfplay_progress iter={iteration} games={done_games}/"
                        f"{total_games} elapsed={format_duration(elapsed)} "
                        f"eta={format_duration(eta)} worker={event['worker_id']} "
                        f"device={event['device']} worker_game={event['game_index']}/"
                        f"{event['games']} moves={event['moves']} "
                        f"policy_entropy={event['policy_entropy']:.3f} "
                        f"game_time={format_duration(event['game_seconds'])}",
                        flush=True,
                    )
                    last_heartbeat = now
                elif now - last_heartbeat >= 30:
                    elapsed = now - start
                    if done_games:
                        avg = elapsed / done_games
                        eta_text = format_duration(avg * (total_games - done_games))
                    else:
                        eta_text = "warming_up"
                    print(
                        f"selfplay_wait iter={iteration} games={done_games}/"
                        f"{total_games} elapsed={format_duration(elapsed)} "
                        f"eta={eta_text}",
                        flush=True,
                    )
                    last_heartbeat = now

                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    examples, lengths, stats, device = future.result()
                    all_examples.extend(examples)
                    all_lengths.extend(lengths)
                    all_stats.append(stats)
                    print(
                        f"selfplay_worker_done iter={iteration} device={device} "
                        f"games={len(lengths)} examples={len(examples)} "
                        f"policy_entropy={stats['policy_entropy']:.3f}",
                        flush=True,
                    )
    return all_examples, all_lengths, merge_stats(all_stats)


def replay_to_dataset(replay: deque[Example]) -> TensorDataset:
    states = tensor_from_array(np.stack([item[0] for item in replay]), dtype=torch.float32)
    policies = tensor_from_array(np.stack([item[1] for item in replay]), dtype=torch.float32)
    values = torch.tensor([item[2] for item in replay], dtype=torch.float32)
    return TensorDataset(states, policies, values)


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: deque[Example],
    cfg: TrainConfig,
    scaler: object | None,
    dataset: TensorDataset | None = None,
) -> TrainStats:
    if dataset is None:
        dataset = replay_to_dataset(replay)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=cfg.device.startswith("cuda"),
        num_workers=0,
        drop_last=cfg.train_steps_per_iteration > 0,
    )

    model.train()
    totals = TrainStats(lr=float(optimizer.param_groups[0]["lr"]))
    autocast_ctx = autocast_context(cfg.device, cfg.amp, cfg.amp_dtype)

    steps_to_run = cfg.train_steps_per_iteration if cfg.train_steps_per_iteration > 0 else len(loader)
    loader_iter = iter(loader)
    for _ in range(steps_to_run):
        try:
            batch_states, batch_policies, batch_values = next(loader_iter)
        except StopIteration:
            if cfg.train_steps_per_iteration <= 0:
                break
            loader_iter = iter(loader)
            batch_states, batch_policies, batch_values = next(loader_iter)

        batch_states = batch_states.to(cfg.device, non_blocking=True)
        batch_policies = batch_policies.to(cfg.device, non_blocking=True)
        batch_values = batch_values.to(cfg.device, non_blocking=True)

        if batch_values.min().item() < -1.001 or batch_values.max().item() > 1.001:
            raise RuntimeError(
                f"value labels out of range: min={batch_values.min().item()} "
                f"max={batch_values.max().item()}"
            )
        policy_sums = batch_policies.sum(dim=1)
        if not torch.isfinite(batch_policies).all() or not torch.isfinite(batch_values).all():
            raise RuntimeError("non-finite policy or value labels in replay")
        if torch.any(torch.abs(policy_sums - 1.0) > 1.0e-3):
            batch_policies = batch_policies / policy_sums.clamp_min(1.0e-8).unsqueeze(1)

        with autocast_ctx:
            logits, predicted_values = model(batch_states)
        logits = logits.float()
        predicted_values = predicted_values.float()
        if not torch.isfinite(logits).all():
            totals.skipped += 1
            print("train_warning skipped_batch reason=nonfinite_logits", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue
        invalid_values = (
            (~torch.isfinite(predicted_values))
            | (predicted_values < -1.0)
            | (predicted_values > 1.0)
        )
        if invalid_values.any():
            totals.value_clamps += int(invalid_values.sum().item())
            print(
                "train_warning clamped_values "
                f"count={int(invalid_values.sum().item())} "
                f"pred_value_min={float(torch.nan_to_num(predicted_values.detach()).min().cpu())} "
                f"pred_value_max={float(torch.nan_to_num(predicted_values.detach()).max().cpu())}",
                flush=True,
            )
            predicted_values = torch.nan_to_num(
                predicted_values, nan=0.0, posinf=1.0, neginf=-1.0
            ).clamp(-1.0, 1.0)
        log_probs = F.log_softmax(logits, dim=1)
        pred_probs = F.softmax(logits, dim=1)
        target_entropy = -(batch_policies * torch.log(batch_policies.clamp_min(1.0e-8))).sum(dim=1)
        pred_entropy = -(pred_probs * torch.log(pred_probs.clamp_min(1.0e-8))).sum(dim=1)
        policy_loss = -(batch_policies * log_probs).sum(dim=1).mean()
        policy_kl = (batch_policies * (torch.log(batch_policies.clamp_min(1.0e-8)) - log_probs)).sum(dim=1).mean()
        value_loss = F.mse_loss(predicted_values, batch_values)
        loss = cfg.policy_loss_weight * policy_loss + cfg.value_loss_weight * value_loss

        if (not torch.isfinite(loss)) or loss.item() > cfg.max_loss:
            totals.skipped += 1
            print(
                "train_warning skipped_batch "
                f"loss={float(loss.detach().cpu())} "
                f"policy={float(policy_loss.detach().cpu())} "
                f"value={float(value_loss.detach().cpu())} "
                f"logits_min={float(logits.detach().min().cpu())} "
                f"logits_max={float(logits.detach().max().cpu())} "
                f"pred_value_min={float(predicted_values.detach().min().cpu())} "
                f"pred_value_max={float(predicted_values.detach().max().cpu())} "
                f"label_min={float(batch_values.detach().min().cpu())} "
                f"label_max={float(batch_values.detach().max().cpu())}",
                flush=True,
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        grad_norm_value = float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)
        if not math.isfinite(grad_norm_value):
            totals.skipped += 1
            print(
                "train_warning skipped_batch "
                f"reason=nonfinite_grad grad_norm={grad_norm_value} "
                f"loss={float(loss.detach().cpu())} "
                f"policy={float(policy_loss.detach().cpu())} "
                f"value={float(value_loss.detach().cpu())}",
                flush=True,
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.update()
            continue

        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        target_best = batch_policies.argmax(dim=1)
        pred_best = logits.argmax(dim=1)
        value_acc = (torch.sign(predicted_values.detach()) == torch.sign(batch_values)).float()
        non_draw = batch_values.abs() > 1.0e-6
        if non_draw.any():
            value_acc_mean = value_acc[non_draw].mean()
        else:
            value_acc_mean = torch.tensor(1.0, device=batch_values.device)

        batch_size = batch_states.shape[0]
        totals.loss += float(loss.item()) * batch_size
        totals.policy_loss += float(policy_loss.item()) * batch_size
        totals.value_loss += float(value_loss.item()) * batch_size
        totals.policy_kl += float(policy_kl.item()) * batch_size
        totals.target_entropy += float(target_entropy.mean().item()) * batch_size
        totals.pred_entropy += float(pred_entropy.mean().item()) * batch_size
        totals.policy_top1 += float((target_best == pred_best).float().mean().item()) * batch_size
        totals.value_mae += float((predicted_values.detach() - batch_values).abs().mean().item()) * batch_size
        totals.value_acc += float(value_acc_mean.item()) * batch_size
        totals.grad_norm += grad_norm_value * batch_size
        totals.batches += batch_size
        totals.optimizer_steps += 1

    denom = max(1, totals.batches)
    totals.loss /= denom
    totals.policy_loss /= denom
    totals.value_loss /= denom
    totals.policy_kl /= denom
    totals.target_entropy /= denom
    totals.pred_entropy /= denom
    totals.policy_top1 /= denom
    totals.value_mae /= denom
    totals.value_acc /= denom
    totals.grad_norm /= denom
    return totals


def select_mcts_action(
    model: PolicyValueNet,
    state: GomokuState,
    simulations: int,
    device: str,
    cfg: TrainConfig,
) -> int:
    root = MCTS(
        model,
        MCTSConfig(
            simulations=simulations,
            c_puct=cfg.mcts_c_puct,
            dirichlet_alpha=cfg.mcts_dirichlet_alpha,
            dirichlet_fraction=cfg.mcts_dirichlet_fraction,
            eval_batch_size=min(cfg.mcts_batch_size, max(1, simulations)),
            amp_dtype=cfg.mcts_amp_dtype,
        ),
        device=device,
    ).search(state, add_exploration_noise=False)
    policy = visit_count_policy(root, state.action_size, temperature=0.0)
    if policy.sum() <= 0:
        legal = state.legal_actions()
        return int(legal[0])
    return int(policy.argmax())


def evaluate_candidate(
    candidate: PolicyValueNet,
    baseline: PolicyValueNet | None,
    cfg: TrainConfig,
) -> dict[str, float]:
    if baseline is None or cfg.eval_games <= 0:
        return {"win_rate": 1.0, "wins": 0.0, "losses": 0.0, "draws": 0.0}

    candidate.eval()
    baseline.eval()
    wins = losses = draws = 0
    for game_index in range(cfg.eval_games):
        state = GomokuState.new(size=cfg.board_size, win_length=cfg.win_length)
        candidate_player = 1 if game_index % 2 == 0 else -1
        while not state.is_terminal:
            if state.current_player == candidate_player:
                action = select_mcts_action(candidate, state, cfg.eval_simulations, cfg.device, cfg)
            else:
                action = select_mcts_action(baseline, state, cfg.eval_simulations, cfg.device, cfg)
            state = state.apply(action)
        if state.winner == 0:
            draws += 1
        elif state.winner == candidate_player:
            wins += 1
        else:
            losses += 1
    win_rate = (wins + 0.5 * draws) / max(1, cfg.eval_games)
    return {
        "win_rate": float(win_rate),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
    }


def save_checkpoint(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    iteration: int,
    stats: TrainStats | None = None,
) -> Path:
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"gomoku10_iter_{iteration:04d}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "model_kwargs": architecture_from_config(cfg),
            "iteration": iteration,
            "stats": asdict(stats) if stats is not None else None,
        },
        path,
    )
    return path


def best_checkpoint_path(cfg: TrainConfig) -> Path:
    return Path(cfg.checkpoint_dir) / "gomoku10_best.pt"


def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    enable_fast_math()
    if cfg.self_play_workers > 1:
        limit_worker_threads()
    cfg.device = resolve_device(cfg.device)
    print(f"device={cfg.device}")
    base_model = make_model(cfg).to(cfg.device)
    model: torch.nn.Module = base_model
    if cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("model_compiled=true", flush=True)
        except Exception as exc:
            print(f"model_compile_warning error={exc}", flush=True)
            model = base_model
    if cfg.data_parallel and cfg.device.startswith("cuda") and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print(f"training_data_parallel={device_ids}")
    print(
        f"self_play_workers={cfg.self_play_workers} "
        f"self_play_devices={split_devices(cfg.self_play_devices, cfg.device)}"
    )
    print(
        f"config preset={cfg.preset} iterations={cfg.iterations} "
        f"games_per_iteration={cfg.games_per_iteration} simulations={cfg.simulations} "
        f"mcts_batch_size={cfg.mcts_batch_size} epochs={cfg.epochs} "
        f"train_steps_per_iteration={cfg.train_steps_per_iteration} "
        f"batch_size={cfg.batch_size} replay_size={cfg.replay_size} "
        f"lr={cfg.learning_rate} min_lr={cfg.min_learning_rate} "
        f"lr_schedule={cfg.lr_schedule} warmup_iterations={cfg.warmup_iterations} "
        f"mcts_c_puct={cfg.mcts_c_puct} "
        f"dirichlet_alpha={cfg.mcts_dirichlet_alpha} "
        f"dirichlet_fraction={cfg.mcts_dirichlet_fraction} "
        f"channels={cfg.channels} residual_blocks={cfg.residual_blocks} "
        f"policy_channels={cfg.policy_channels} value_channels={cfg.value_channels} "
        f"value_hidden={cfg.value_hidden} use_se={cfg.use_se} "
        f"augment_symmetries={cfg.augment_symmetries} "
        f"amp={cfg.amp} amp_dtype={cfg.amp_dtype} mcts_amp_dtype={cfg.mcts_amp_dtype} "
        f"metrics_path={cfg.metrics_path or 'none'}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        unwrap_model(model).parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    start_iteration = load_resume_checkpoint(base_model, optimizer, cfg)
    end_iteration = start_iteration + cfg.iterations
    replay = load_replay(cfg)
    mcts_cfg = MCTSConfig(
        simulations=cfg.simulations,
        c_puct=cfg.mcts_c_puct,
        dirichlet_alpha=cfg.mcts_dirichlet_alpha,
        dirichlet_fraction=cfg.mcts_dirichlet_fraction,
        eval_batch_size=cfg.mcts_batch_size,
        amp_dtype=cfg.mcts_amp_dtype,
    )
    scaler = make_grad_scaler(cfg.amp and cfg.device.startswith("cuda"), cfg.amp_dtype)
    champion = copy.deepcopy(base_model).to(cfg.device) if cfg.eval_interval and cfg.eval_games else None
    run_start = time.monotonic()

    for iteration in range(start_iteration + 1, end_iteration + 1):
        completed_this_run = iteration - start_iteration
        iteration_start = time.monotonic()
        set_optimizer_lr(optimizer, scheduled_learning_rate(cfg, iteration, end_iteration))
        print(f"iter={iteration}/{end_iteration} phase=selfplay_start", flush=True)
        examples, lengths, selfplay_stats = collect_self_play_examples(base_model, cfg, mcts_cfg, iteration)
        raw_examples = len(examples)
        if cfg.augment_symmetries:
            examples = augment_examples(examples, cfg.board_size)
        replay.extend(examples)
        new_examples = len(examples)
        print(
            f"iter={iteration}/{end_iteration} phase=selfplay_done "
            f"raw_examples={raw_examples} examples={new_examples} "
            f"avg_moves={np.mean(lengths):.1f} replay={len(replay)} "
            f"target_policy_entropy={selfplay_stats['policy_entropy']:.4f} "
            f"black_win_rate={selfplay_stats['black_win_rate']:.3f} "
            f"white_win_rate={selfplay_stats['white_win_rate']:.3f} "
            f"draw_rate={selfplay_stats['draw_rate']:.3f} "
            f"elapsed={format_duration(time.monotonic() - iteration_start)}",
            flush=True,
        )

        stats = TrainStats()
        train_dataset = replay_to_dataset(replay)
        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.monotonic()
            stats = train_epoch(model, optimizer, replay, cfg, scaler, train_dataset)
            print(
                f"train_progress iter={iteration}/{end_iteration} "
                f"epoch={epoch}/{cfg.epochs} elapsed={format_duration(time.monotonic() - epoch_start)} "
                f"loss={stats.loss:.4f} policy={stats.policy_loss:.4f} "
                f"value={stats.value_loss:.4f} policy_kl={stats.policy_kl:.4f} "
                f"target_entropy={stats.target_entropy:.4f} pred_entropy={stats.pred_entropy:.4f} "
                f"policy_top1={stats.policy_top1:.4f} value_mae={stats.value_mae:.4f} "
                f"value_acc={stats.value_acc:.4f} grad_norm={stats.grad_norm:.4f} "
                f"steps={stats.optimizer_steps} lr={stats.lr:.6g} "
                f"skipped={stats.skipped} value_clamps={stats.value_clamps}",
                flush=True,
            )

        eval_stats = {"win_rate": 1.0, "wins": 0.0, "losses": 0.0, "draws": 0.0}
        promoted = True
        if champion is not None and cfg.eval_interval > 0 and iteration % cfg.eval_interval == 0:
            eval_stats = evaluate_candidate(base_model, champion, cfg)
            promoted = eval_stats["win_rate"] >= cfg.promotion_threshold
            print(
                f"eval_progress iter={iteration}/{end_iteration} "
                f"candidate_score={eval_stats['win_rate']:.3f} "
                f"wins={int(eval_stats['wins'])} losses={int(eval_stats['losses'])} "
                f"draws={int(eval_stats['draws'])} threshold={cfg.promotion_threshold:.3f} "
                f"promoted={promoted}",
                flush=True,
            )
            if promoted:
                champion.load_state_dict(base_model.state_dict())
            elif cfg.gate_evaluation:
                base_model.load_state_dict(champion.state_dict())
                print(f"eval_reverted iter={iteration} reason=below_threshold", flush=True)

        checkpoint = save_checkpoint(base_model, optimizer, cfg, iteration, stats)
        if promoted:
            best_path = best_checkpoint_path(cfg)
            best_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkpoint, best_path)
        if cfg.replay_path and cfg.replay_save_interval > 0 and iteration % cfg.replay_save_interval == 0:
            save_replay(replay, cfg.replay_path)

        total_elapsed = time.monotonic() - run_start
        avg_iteration = total_elapsed / completed_this_run
        total_eta = avg_iteration * (end_iteration - iteration)
        iteration_elapsed = time.monotonic() - iteration_start
        append_metrics(
            cfg.metrics_path,
            {
                "iteration": iteration,
                "end_iteration": end_iteration,
                "raw_examples": raw_examples,
                "examples": new_examples,
                "avg_moves": float(np.mean(lengths)),
                "replay": len(replay),
                "loss": stats.loss,
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
                "policy_kl": stats.policy_kl,
                "target_entropy": stats.target_entropy,
                "pred_entropy": stats.pred_entropy,
                "policy_top1": stats.policy_top1,
                "value_mae": stats.value_mae,
                "value_acc": stats.value_acc,
                "grad_norm": stats.grad_norm,
                "lr": stats.lr,
                "optimizer_steps": stats.optimizer_steps,
                "skipped": stats.skipped,
                "value_clamps": stats.value_clamps,
                "selfplay_entropy": selfplay_stats["policy_entropy"],
                "black_win_rate": selfplay_stats["black_win_rate"],
                "white_win_rate": selfplay_stats["white_win_rate"],
                "draw_rate": selfplay_stats["draw_rate"],
                "eval_score": eval_stats["win_rate"],
                "eval_wins": eval_stats["wins"],
                "eval_losses": eval_stats["losses"],
                "eval_draws": eval_stats["draws"],
                "promoted": promoted,
                "iter_seconds": iteration_elapsed,
                "total_seconds": total_elapsed,
                "checkpoint": str(checkpoint),
            },
        )
        print(
            f"iter={iteration}/{end_iteration} raw_examples={raw_examples} examples={new_examples} "
            f"avg_moves={np.mean(lengths):.1f} loss={stats.loss:.4f} "
            f"policy={stats.policy_loss:.4f} value={stats.value_loss:.4f} "
            f"policy_kl={stats.policy_kl:.4f} target_entropy={stats.target_entropy:.4f} "
            f"pred_entropy={stats.pred_entropy:.4f} policy_top1={stats.policy_top1:.4f} "
            f"value_mae={stats.value_mae:.4f} value_acc={stats.value_acc:.4f} "
            f"selfplay_entropy={selfplay_stats['policy_entropy']:.4f} "
            f"eval_score={eval_stats['win_rate']:.3f} steps={stats.optimizer_steps} "
            f"skipped={stats.skipped} value_clamps={stats.value_clamps} "
            f"iter_time={format_duration(iteration_elapsed)} "
            f"total_elapsed={format_duration(total_elapsed)} "
            f"total_eta={format_duration(total_eta)} saved={checkpoint}",
            flush=True,
        )

    save_replay(replay, cfg.replay_path)


def build_parser(defaults: TrainConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a 10x10 Gomoku AI with AlphaZero-style self-play."
    )
    parser.add_argument(
        "--preset",
        choices=["local", "a100-4", "a100-fast", "a100-turbo", "a100-prod"],
        default=defaults.preset,
    )
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--games-per-iteration", type=int, default=defaults.games_per_iteration)
    parser.add_argument("--simulations", type=int, default=defaults.simulations)
    parser.add_argument("--mcts-batch-size", type=int, default=defaults.mcts_batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--train-steps-per-iteration", type=int, default=defaults.train_steps_per_iteration)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--replay-size", type=int, default=defaults.replay_size)
    parser.add_argument("--channels", type=int, default=defaults.channels)
    parser.add_argument("--residual-blocks", type=int, default=defaults.residual_blocks)
    parser.add_argument("--policy-channels", type=int, default=defaults.policy_channels)
    parser.add_argument("--value-channels", type=int, default=defaults.value_channels)
    parser.add_argument("--value-hidden", type=int, default=defaults.value_hidden)
    parser.add_argument("--use-se", dest="use_se", action="store_true")
    parser.add_argument("--no-use-se", dest="use_se", action="store_false")
    parser.add_argument("--policy-loss-weight", type=float, default=defaults.policy_loss_weight)
    parser.add_argument("--value-loss-weight", type=float, default=defaults.value_loss_weight)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--min-learning-rate", type=float, default=defaults.min_learning_rate)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default=defaults.lr_schedule)
    parser.add_argument("--warmup-iterations", type=int, default=defaults.warmup_iterations)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--max-loss", type=float, default=defaults.max_loss)
    parser.add_argument("--temperature-moves", type=int, default=defaults.temperature_moves)
    parser.add_argument("--mcts-c-puct", type=float, default=defaults.mcts_c_puct)
    parser.add_argument("--mcts-dirichlet-alpha", type=float, default=defaults.mcts_dirichlet_alpha)
    parser.add_argument("--mcts-dirichlet-fraction", type=float, default=defaults.mcts_dirichlet_fraction)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--checkpoint-dir", default=defaults.checkpoint_dir)
    parser.add_argument("--metrics-path", default=defaults.metrics_path)
    parser.add_argument("--replay-path", default=defaults.replay_path)
    parser.add_argument("--replay-save-interval", type=int, default=defaults.replay_save_interval)
    parser.add_argument("--resume", default=defaults.resume)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--self-play-workers", type=int, default=defaults.self_play_workers)
    parser.add_argument("--self-play-devices", default=defaults.self_play_devices)
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    parser.add_argument("--eval-games", type=int, default=defaults.eval_games)
    parser.add_argument("--eval-simulations", type=int, default=defaults.eval_simulations)
    parser.add_argument("--promotion-threshold", type=float, default=defaults.promotion_threshold)
    parser.add_argument("--gate-evaluation", dest="gate_evaluation", action="store_true")
    parser.add_argument("--no-gate-evaluation", dest="gate_evaluation", action="store_false")
    parser.add_argument("--data-parallel", dest="data_parallel", action="store_true")
    parser.add_argument("--no-data-parallel", dest="data_parallel", action="store_false")
    parser.add_argument("--augment-symmetries", dest="augment_symmetries", action="store_true")
    parser.add_argument("--no-augment-symmetries", dest="augment_symmetries", action="store_false")
    parser.add_argument("--compile-model", dest="compile_model", action="store_true")
    parser.add_argument("--no-compile-model", dest="compile_model", action="store_false")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default=defaults.amp_dtype)
    parser.add_argument(
        "--mcts-amp-dtype",
        choices=["bf16", "fp16", "none"],
        default=defaults.mcts_amp_dtype,
    )
    parser.set_defaults(
        data_parallel=defaults.data_parallel,
        augment_symmetries=defaults.augment_symmetries,
        use_se=defaults.use_se,
        gate_evaluation=defaults.gate_evaluation,
        compile_model=defaults.compile_model,
        amp=defaults.amp,
    )
    return parser


def main() -> None:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument(
        "--preset",
        choices=["local", "a100-4", "a100-fast", "a100-turbo", "a100-prod"],
        default="local",
    )
    preset_args, _ = preset_parser.parse_known_args()
    args = build_parser(preset_config(preset_args.preset)).parse_args()
    cfg = TrainConfig(**vars(args))
    run_training(cfg)


if __name__ == "__main__":
    main()
