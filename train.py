from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import multiprocessing as mp
import random
import shutil
import sys
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

# (encoded_state, mcts_policy, value, policy_weight). policy_weight is 0 for
# positions from cheap searches (playout cap randomization): they only train
# the value head.
Example = tuple[np.ndarray, np.ndarray, float, float]


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
    train_data_workers: int = 0
    train_prefetch_factor: int = 2
    replay_size: int = 10_000
    min_replay_size: int = 0
    max_train_replay_passes: float = 0.0
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
    use_global_pool: bool = False       # KataGo: global context pooling in each residual block
    use_soft_policy: bool = False       # KataGo: auxiliary soft policy head (target = π^(1/T))
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    soft_policy_loss_weight: float = 0.0   # KataGo: weight for soft policy loss (use ~8.0)
    soft_policy_temp: float = 4.0          # temperature T for soft target: π^(1/T)
    surprise_weighting: bool = False    # KataGo: weight replay samples by KL(prior||MCTS)
    mcts_value_weight: float = 0.0      # KataGo: mix MCTS root value into value target (0=off,0.5=half)
    # KataGo MCTS improvements
    mcts_root_policy_temp: float = 1.0  # >1 flattens root priors for better early exploration
    mcts_shaped_dirichlet: bool = False # shaped Dirichlet noise by prior rank
    mcts_dynamic_cpuct: bool = False    # scale c_puct by empirical value variance
    mcts_fpu_reduction: float = 0.0     # KataGo FPU: unvisited child Q = parent Q - fpu * sqrt(visited mass)
    mcts_forced_playouts: bool = False  # KataGo: forced playouts + policy target pruning at the root
    mcts_forced_playout_k: float = 2.0  # n_forced = sqrt(k * prior * root_visits)
    playout_cap_randomization: bool = False  # KataGo: mix cheap (value-only) and full (policy) searches
    full_search_prob: float = 0.25      # probability of a full search per move when PCR is on
    fast_simulations: int = 0           # simulations for cheap searches (0 = simulations // 4)
    selfplay_tree_reuse: bool = True    # reuse the chosen subtree between self-play moves
    augment_symmetries: bool = True     # random dihedral symmetry per sample at train time
    replay_path: str = ""
    replay_save_interval: int = 5
    eval_interval: int = 0
    eval_games: int = 0
    eval_simulations: int = 128
    eval_opening_moves: int = 2         # random opening plies per eval game pair (colors swapped)
    eval_progress_interval: int = 1
    eval_workers: int = 1
    eval_devices: str = "auto"
    eval_early_cutoff: bool = False
    promotion_threshold: float = 0.55
    gate_evaluation: bool = False
    early_stop_evals: int = 0           # stop after N consecutive failed evaluations (0 = off)
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
    soft_policy_loss: float = 0.0
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
    if name == "v2":
        cfg.iterations = 60
        cfg.games_per_iteration = 128
        cfg.simulations = 512
        cfg.mcts_batch_size = 64
        cfg.epochs = 3
        cfg.train_steps_per_iteration = 128
        cfg.batch_size = 2048
        cfg.replay_size = 120_000
        cfg.learning_rate = 8.0e-5
        cfg.min_learning_rate = 8.0e-6
        cfg.lr_schedule = "cosine"
        cfg.warmup_iterations = 2
        cfg.weight_decay = 1.0e-4
        cfg.max_grad_norm = 10.0
        cfg.temperature_moves = 12
        cfg.mcts_c_puct = 1.25
        cfg.mcts_dirichlet_alpha = 0.15
        cfg.channels = 192
        cfg.residual_blocks = 12
        cfg.policy_channels = 16
        cfg.value_channels = 8
        cfg.value_hidden = 512
        cfg.use_se = False
        cfg.use_global_pool = True
        cfg.use_soft_policy = True
        cfg.soft_policy_loss_weight = 4.0
        cfg.value_loss_weight = 1.5
        cfg.surprise_weighting = True
        cfg.mcts_value_weight = 0.5
        cfg.mcts_root_policy_temp = 1.1
        cfg.mcts_shaped_dirichlet = True
        cfg.mcts_dynamic_cpuct = True
        cfg.mcts_fpu_reduction = 0.2
        cfg.mcts_forced_playouts = True
        cfg.playout_cap_randomization = True
        cfg.full_search_prob = 0.33
        cfg.fast_simulations = 128
        cfg.checkpoint_dir = "alphazero_gomoku/outputs/checkpoints/v2"
        cfg.replay_path = "alphazero_gomoku/outputs/replay/v2_replay.pt"
        cfg.replay_save_interval = 2
        cfg.metrics_path = "alphazero_gomoku/outputs/metrics/v2.jsonl"
        cfg.resume = "alphazero_gomoku/outputs/checkpoints/a100-4-prod-v3/gomoku10_best.pt"
        cfg.self_play_workers = 32
        cfg.self_play_devices = "auto"
        cfg.data_parallel = False
        cfg.eval_interval = 5
        cfg.eval_games = 32
        cfg.eval_simulations = 512
        cfg.eval_opening_moves = 4
        cfg.eval_progress_interval = 1
        cfg.eval_workers = 8
        cfg.train_data_workers = 2
        cfg.eval_early_cutoff = True
        cfg.promotion_threshold = 0.55
        cfg.early_stop_evals = 4
        return cfg
    if name == "v3-local":
        cfg.iterations = 40
        cfg.games_per_iteration = 48
        cfg.simulations = 256
        cfg.mcts_batch_size = 32
        cfg.epochs = 3
        cfg.train_steps_per_iteration = 64
        cfg.batch_size = 1024
        cfg.replay_size = 50_000
        cfg.min_replay_size = 20_000
        cfg.max_train_replay_passes = 2.0
        cfg.learning_rate = 4.0e-5
        cfg.min_learning_rate = 8.0e-6
        cfg.lr_schedule = "cosine"
        cfg.warmup_iterations = 2
        cfg.weight_decay = 1.0e-4
        cfg.max_grad_norm = 10.0
        cfg.temperature_moves = 12
        cfg.mcts_c_puct = 1.25
        cfg.mcts_dirichlet_alpha = 0.15
        cfg.channels = 192
        cfg.residual_blocks = 12
        cfg.policy_channels = 16
        cfg.value_channels = 8
        cfg.value_hidden = 512
        cfg.use_global_pool = True
        cfg.use_soft_policy = True
        cfg.soft_policy_loss_weight = 4.0
        cfg.value_loss_weight = 1.5
        cfg.surprise_weighting = True
        cfg.mcts_value_weight = 0.5
        cfg.mcts_root_policy_temp = 1.1
        cfg.mcts_shaped_dirichlet = True
        cfg.mcts_dynamic_cpuct = True
        cfg.mcts_fpu_reduction = 0.2
        cfg.mcts_forced_playouts = True
        cfg.playout_cap_randomization = True
        cfg.full_search_prob = 0.50
        cfg.fast_simulations = 64
        cfg.checkpoint_dir = "alphazero_gomoku/outputs/checkpoints/v3-local"
        cfg.replay_path = "alphazero_gomoku/outputs/replay/v3-local_replay.pt"
        cfg.replay_save_interval = 2
        cfg.metrics_path = "alphazero_gomoku/outputs/metrics/v3-local.jsonl"
        cfg.resume = "alphazero_gomoku/outputs/checkpoints/a100-4-prod-v3/gomoku10_best.pt"
        cfg.self_play_workers = 4
        cfg.self_play_devices = "auto"
        cfg.data_parallel = False
        cfg.eval_interval = 5
        cfg.eval_games = 16
        cfg.eval_simulations = 256
        cfg.eval_opening_moves = 4
        cfg.eval_progress_interval = 1
        cfg.eval_workers = 4
        cfg.eval_early_cutoff = True
        cfg.promotion_threshold = 0.55
        cfg.gate_evaluation = True
        cfg.early_stop_evals = 3
        return cfg
    if name == "v3-student-local":
        cfg.iterations = 80
        cfg.games_per_iteration = 96
        cfg.simulations = 256
        cfg.mcts_batch_size = 32
        cfg.epochs = 2
        cfg.train_steps_per_iteration = 96
        cfg.batch_size = 2048
        cfg.replay_size = 80_000
        cfg.min_replay_size = 25_000
        cfg.max_train_replay_passes = 2.0
        cfg.learning_rate = 6.0e-5
        cfg.min_learning_rate = 8.0e-6
        cfg.lr_schedule = "cosine"
        cfg.warmup_iterations = 3
        cfg.weight_decay = 1.0e-4
        cfg.max_grad_norm = 10.0
        cfg.temperature_moves = 12
        cfg.mcts_c_puct = 1.25
        cfg.mcts_dirichlet_alpha = 0.15
        cfg.channels = 128
        cfg.residual_blocks = 8
        cfg.policy_channels = 12
        cfg.value_channels = 6
        cfg.value_hidden = 384
        cfg.use_global_pool = True
        cfg.use_soft_policy = True
        cfg.soft_policy_loss_weight = 2.0
        cfg.value_loss_weight = 1.25
        cfg.surprise_weighting = False
        cfg.mcts_value_weight = 0.25
        cfg.mcts_root_policy_temp = 1.1
        cfg.mcts_shaped_dirichlet = True
        cfg.mcts_dynamic_cpuct = True
        cfg.mcts_fpu_reduction = 0.2
        cfg.mcts_forced_playouts = True
        cfg.playout_cap_randomization = True
        cfg.full_search_prob = 0.35
        cfg.fast_simulations = 64
        cfg.checkpoint_dir = "alphazero_gomoku/outputs/checkpoints/v3-student-local"
        cfg.replay_path = "alphazero_gomoku/outputs/replay/v3-student-local_replay.pt"
        cfg.replay_save_interval = 2
        cfg.metrics_path = "alphazero_gomoku/outputs/metrics/v3-student-local.jsonl"
        cfg.resume = "alphazero_gomoku/outputs/checkpoints/distill-oldbest-128x8/gomoku10_student_best.pt"
        cfg.self_play_workers = 1
        cfg.self_play_devices = "auto"
        cfg.eval_interval = 5
        cfg.eval_games = 20
        cfg.eval_simulations = 256
        cfg.eval_opening_moves = 4
        cfg.eval_progress_interval = 1
        cfg.eval_workers = 8
        cfg.train_data_workers = 2
        cfg.eval_early_cutoff = True
        cfg.promotion_threshold = 0.55
        cfg.gate_evaluation = True
        cfg.early_stop_evals = 3
        return cfg
    if name == "a100-4":
        cfg.iterations = 100
        cfg.games_per_iteration = 96
        cfg.simulations = 384
        cfg.mcts_batch_size = 48
        cfg.epochs = 3
        cfg.train_steps_per_iteration = 96
        cfg.batch_size = 4096
        # Raw positions only (symmetry augmentation happens at train time), so
        # 80k ≈ a ~50-iteration window. The old 500k was sized for 8x-augmented
        # entries; keeping it would mean never evicting stale early-run data,
        # whose bootstrapped value labels (mcts_value_weight) decay into noise.
        cfg.replay_size = 80_000
        cfg.learning_rate = 2.0e-4
        cfg.min_learning_rate = 2.0e-5
        cfg.lr_schedule = "cosine"
        cfg.warmup_iterations = 5
        cfg.weight_decay = 1.0e-4
        # The 8x soft policy term keeps healthy grad norms around 10; clipping
        # at 5 would permanently halve the effective LR instead of catching spikes.
        cfg.max_grad_norm = 10.0
        cfg.temperature_moves = 12
        cfg.mcts_c_puct = 1.25
        cfg.mcts_dirichlet_alpha = 0.15
        cfg.channels = 192
        cfg.residual_blocks = 12
        cfg.policy_channels = 16
        cfg.value_channels = 8
        cfg.value_hidden = 512
        cfg.use_se = False                  # global pooling replaces SE (both are channel-wise context)
        cfg.use_global_pool = True          # KataGo: global context injection (additive pooling bias)
        cfg.use_soft_policy = True          # KataGo: auxiliary soft policy head
        cfg.soft_policy_loss_weight = 8.0   # KataGo: 8x weight on soft policy loss
        cfg.surprise_weighting = True       # KataGo: prioritise high-KL replay samples
        cfg.mcts_value_weight = 0.5         # KataGo: mix MCTS value into value target
        cfg.mcts_root_policy_temp = 1.1     # KataGo: flatten root priors slightly
        cfg.mcts_shaped_dirichlet = True    # KataGo: shaped Dirichlet by prior rank
        cfg.mcts_dynamic_cpuct = True       # KataGo: variance-scaled exploration
        cfg.mcts_fpu_reduction = 0.2        # KataGo: first-play urgency reduction
        cfg.mcts_forced_playouts = True     # KataGo: forced playouts + policy target pruning
        cfg.playout_cap_randomization = True  # KataGo: cheap value-only searches most moves
        cfg.full_search_prob = 0.25
        cfg.fast_simulations = 96
        cfg.checkpoint_dir = "alphazero_gomoku/outputs/checkpoints/a100-4"
        cfg.replay_path = "alphazero_gomoku/outputs/replay/a100-4_replay.pt"
        cfg.metrics_path = "alphazero_gomoku/outputs/metrics/a100-4.jsonl"
        cfg.self_play_workers = 16
        cfg.self_play_devices = "auto"
        cfg.data_parallel = False
        cfg.eval_interval = 5
        cfg.eval_games = 16
        cfg.eval_simulations = 128
        cfg.eval_workers = 8
        cfg.train_data_workers = 2
        cfg.promotion_threshold = 0.55
        cfg.early_stop_evals = 3            # stop after 3 failed evals (15 stagnant iterations)
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
        cfg.eval_workers = 16
        cfg.train_data_workers = 2
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
        cfg.train_data_workers = 2
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
        cfg.train_data_workers = 2
        return cfg
    raise ValueError(f"unknown preset: {name}")


def split_devices(device_spec: str, fallback_device: str) -> list[str]:
    if device_spec == "auto":
        if fallback_device.startswith("cuda") and torch.cuda.is_available():
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


def save_replay(replay: deque[Example], kl_buffer: deque[float], path: str) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp")
    torch.save({"version": 2, "examples": list(replay), "kl": list(kl_buffer)}, tmp)
    tmp.replace(target)
    print(f"replay_saved path={target} examples={len(replay)}", flush=True)


def load_replay(cfg: TrainConfig) -> tuple[deque[Example], deque[float]]:
    """Load the replay buffer and its aligned per-sample KL surprises.

    The KL buffer is persisted alongside the examples so surprise weighting
    keeps working across resumes instead of silently turning off until the
    buffers refill.
    """
    replay: deque[Example] = deque(maxlen=cfg.replay_size)
    kl_buffer: deque[float] = deque(maxlen=cfg.replay_size)
    if not cfg.replay_path:
        return replay, kl_buffer
    path = Path(cfg.replay_path)
    if not path.exists():
        return replay, kl_buffer
    loaded = load_torch(path, map_location="cpu")
    if isinstance(loaded, dict):
        examples = loaded.get("examples", [])
        kls = loaded.get("kl", [])
    else:  # legacy v1 format: bare list of (state, policy, value)
        examples = loaded
        kls = []
    examples = examples[-cfg.replay_size:]
    examples = [
        item if len(item) == 4 else (item[0], item[1], item[2], 1.0)
        for item in examples
    ]
    replay.extend(examples)
    if len(kls) >= len(examples):
        kl_buffer.extend(kls[-len(examples):])
    else:
        kl_buffer.extend([0.0] * len(replay))
        if cfg.surprise_weighting:
            print(
                "replay_warning kl_buffer_missing surprise weights reset to neutral "
                "for loaded examples",
                flush=True,
            )
    print(f"replay_loaded path={path} examples={len(replay)}", flush=True)
    return replay, kl_buffer


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


def apply_random_symmetries(
    states: torch.Tensor, policies: torch.Tensor, board_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independent random dihedral symmetry to each sample in a batch.

    Replaces the old 8x replay expansion: the buffer now stores raw examples
    and the transform is applied at sampling time, so the buffer holds 8x more
    unique positions for the same memory and the augmentation is effectively
    infinite.
    """
    n = states.shape[0]
    pol2d = policies.view(n, board_size, board_size)
    out_states = states.clone()
    out_pol = pol2d.clone()
    ks = torch.randint(0, 8, (n,), device=states.device)
    for k in range(1, 8):
        idx = (ks == k).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        s = states[idx]
        p = pol2d[idx]
        if k < 4:
            s = torch.rot90(s, k, dims=(2, 3))
            p = torch.rot90(p, k, dims=(1, 2))
        else:
            s = torch.flip(torch.rot90(s, k - 4, dims=(2, 3)), dims=(3,))
            p = torch.flip(torch.rot90(p, k - 4, dims=(1, 2)), dims=(2,))
        out_states[idx] = s
        out_pol[idx] = p
    return out_states, out_pol.reshape(n, -1)


def play_self_game(
    model: PolicyValueNet,
    cfg: TrainConfig,
    mcts_cfg: MCTSConfig,
) -> tuple[list[Example], list[float], dict[str, float]]:
    """Play one self-play game.

    KataGo playout cap randomization: with probability full_search_prob a move
    gets a full search (noise + forced playouts) and produces a policy target;
    other moves get a cheap search and only train the value head
    (policy_weight=0). The policy target is always the τ=1 visit distribution
    (pruned of forced playouts); the sampling temperature only affects which
    move is played.

    Returns:
        examples: (encoded_state, policy_target, value, policy_weight) per move
        kl_surprises: KL(raw_network_prior || MCTS_target) per move — used for
            surprise-weighted replay sampling (KataGo); 0 for cheap searches
        stats: scalar diagnostics
    """
    mcts = MCTS(model, mcts_cfg, device=cfg.device)
    state = GomokuState.new(size=cfg.board_size, win_length=cfg.win_length)
    history: list[tuple[np.ndarray, np.ndarray, int, float, float]] = []
    entropies: list[float] = []
    kl_surprises: list[float] = []
    next_root = None
    full_moves = 0

    while not state.is_terminal:
        is_full = (not cfg.playout_cap_randomization) or (
            random.random() < cfg.full_search_prob
        )
        simulations = cfg.simulations if is_full else max(1, cfg.fast_simulations)
        root = mcts.search(
            state,
            add_exploration_noise=is_full,
            reuse_root=next_root if cfg.selfplay_tree_reuse else None,
            simulations=simulations,
        )

        target = mcts.policy_target(root, state.action_size, pruned=is_full)
        if target.sum() <= 0:
            legal = state.legal_actions()
            target = np.zeros(state.action_size, dtype=np.float32)
            target[legal] = 1.0 / len(legal)
        target = target.astype(np.float32)

        if is_full:
            full_moves += 1
            entropies.append(policy_entropy(target))
            # KL(raw prior || MCTS target) on the pre-noise priors, so the
            # surprise weight measures actual network blind spots, not noise
            actions = sorted(root.children)
            prior = np.array([root.children[a].raw_prior for a in actions], dtype=np.float32)
            mcts_p = np.array([target[a] for a in actions], dtype=np.float32)
            kl = float(np.sum(mcts_p * np.log((mcts_p + 1e-12) / (prior + 1e-12))))
            kl_surprises.append(max(0.0, kl))
        else:
            kl_surprises.append(0.0)

        # MCTS root value estimate (root.value is already from the current
        # player's perspective: backprop negates per ply up to the root)
        mcts_value = float(root.value) if root.visit_count > 0 else 0.0

        history.append(
            (state.encode(), target, state.current_player, mcts_value, 1.0 if is_full else 0.0)
        )

        temperature = 1.0 if state.moves_played < cfg.temperature_moves else 0.0
        move_policy = visit_count_policy(root, state.action_size, temperature)
        if move_policy.sum() <= 0:
            move_policy = target
        action = sample_action(move_policy)
        next_root = root.children.get(action)
        state = state.apply(action)

    examples: list[Example] = []
    w = cfg.mcts_value_weight
    for encoded, policy, player, mcts_val, policy_weight in history:
        if state.winner == 0:
            terminal_v = 0.0
        else:
            terminal_v = 1.0 if state.winner == player else -1.0
        # KataGo short-term value: blend terminal result with MCTS value estimate
        value = (1.0 - w) * terminal_v + w * mcts_val if w > 0 else terminal_v
        examples.append((encoded.astype(np.float32), policy, float(value), policy_weight))

    stats = {
        "policy_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "winner": float(state.winner if state.winner is not None else 0),
        "full_search_rate": full_moves / max(1, len(history)),
    }
    return examples, kl_surprises, stats


def play_self_games_worker(
    cfg_data: dict,
    mcts_data: dict,
    model_state_path: str,
    games: int,
    device: str,
    worker_id: int,
    iteration: int = 0,
) -> tuple[list[Example], list[float], list[int], dict[str, float], str]:
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
    kl_surprises: list[float] = []
    lengths: list[int] = []
    entropies: list[float] = []
    winners: list[float] = []
    full_search_rates: list[float] = []
    for game_index in range(1, games + 1):
        game_examples, game_kls, game_stats = play_self_game(model, cfg, mcts_cfg)
        examples.extend(game_examples)
        kl_surprises.extend(game_kls)
        lengths.append(len(game_examples))
        entropies.append(game_stats["policy_entropy"])
        winners.append(game_stats["winner"])
        full_search_rates.append(game_stats["full_search_rate"])
    stats = {
        "policy_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "black_win_rate": float(np.mean([winner == 1 for winner in winners])) if winners else 0.0,
        "white_win_rate": float(np.mean([winner == -1 for winner in winners])) if winners else 0.0,
        "draw_rate": float(np.mean([winner == 0 for winner in winners])) if winners else 0.0,
        "full_search_rate": float(np.mean(full_search_rates)) if full_search_rates else 0.0,
        "games": float(len(winners)),
    }
    return examples, kl_surprises, lengths, stats, device


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
            "full_search_rate": 0.0,
        }
    if "winner" in stats[0]:
        # per-game stats from the single-worker path
        winners = [item["winner"] for item in stats]
        return {
            "policy_entropy": float(np.mean([item["policy_entropy"] for item in stats])),
            "black_win_rate": float(np.mean([winner == 1 for winner in winners])),
            "white_win_rate": float(np.mean([winner == -1 for winner in winners])),
            "draw_rate": float(np.mean([winner == 0 for winner in winners])),
            "full_search_rate": float(
                np.mean([item.get("full_search_rate", 1.0) for item in stats])
            ),
        }
    # per-worker stats: weight by the number of games each worker played
    games = np.array([item.get("games", 1.0) for item in stats], dtype=np.float64)
    keys = [key for key in stats[0] if key != "games"]
    return {
        key: float(np.average([item[key] for item in stats], weights=games))
        for key in keys
    }


def collect_self_play_examples(
    model: torch.nn.Module,
    cfg: TrainConfig,
    mcts_cfg: MCTSConfig,
    iteration: int,
) -> tuple[list[Example], list[float], list[int], dict[str, float]]:
    if cfg.self_play_workers <= 1:
        examples: list[Example] = []
        kl_surprises: list[float] = []
        lengths: list[int] = []
        stats: list[dict[str, float]] = []
        start = time.monotonic()
        for game_index in range(1, cfg.games_per_iteration + 1):
            game_start = time.monotonic()
            game_examples, game_kls, game_stats = play_self_game(unwrap_model(model), cfg, mcts_cfg)
            examples.extend(game_examples)
            kl_surprises.extend(game_kls)
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
        return examples, kl_surprises, lengths, merge_stats(stats)

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
    all_kl_surprises: list[float] = []
    all_lengths: list[int] = []
    all_stats: list[dict[str, float]] = []
    cfg_data = asdict(cfg)
    mcts_data = asdict(mcts_cfg)
    context = mp.get_context("spawn")
    total_games = sum(game_counts)
    done_games = 0
    start = time.monotonic()
    last_heartbeat = start
    # On Windows, tempfile roots can be blocked by process-handle races. Store
    # these small per-iteration snapshots next to regular checkpoints instead.
    snapshot_dir = (
        Path(cfg.checkpoint_dir)
        / "_selfplay_tmp"
        / f"iter_{iteration:04d}_{int(time.time() * 1000)}"
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    model_state_path = str(snapshot_dir / "model.pt")
    torch.save(cpu_state_dict(model), model_state_path)
    if sys.platform == "win32":
        executor_context = concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs))
    else:
        executor_context = concurrent.futures.ProcessPoolExecutor(
            max_workers=len(jobs), mp_context=context
        )
    try:
        with executor_context as executor:
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
                )
                for worker_id, device, games in jobs
            ]
            pending = set(futures)
            while pending:
                now = time.monotonic()
                if now - last_heartbeat >= 30:
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
                    examples, kl_surprises, lengths, stats, device = future.result()
                    all_examples.extend(examples)
                    all_kl_surprises.extend(kl_surprises)
                    all_lengths.extend(lengths)
                    all_stats.append(stats)
                    done_games += len(lengths)
                    print(
                        f"selfplay_worker_done iter={iteration} device={device} "
                        f"games={done_games}/{total_games} worker_games={len(lengths)} "
                        f"examples={len(examples)} policy_entropy={stats['policy_entropy']:.3f}",
                        flush=True,
                    )
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    return all_examples, all_kl_surprises, all_lengths, merge_stats(all_stats)


def replay_to_dataset(
    replay: deque[Example],
    kl_buffer: deque[float] | None = None,
) -> tuple[TensorDataset, torch.Tensor | None]:
    """Convert replay buffer to TensorDataset.

    Returns (dataset, weights) where weights is a 1-D tensor of per-sample
    surprise weights for WeightedRandomSampler, or None if kl_buffer is absent.
    """
    states = tensor_from_array(np.stack([item[0] for item in replay]), dtype=torch.float32)
    policies = tensor_from_array(np.stack([item[1] for item in replay]), dtype=torch.float32)
    values = torch.tensor([item[2] for item in replay], dtype=torch.float32)
    policy_weights = torch.tensor([item[3] for item in replay], dtype=torch.float32)
    dataset = TensorDataset(states, policies, values, policy_weights)
    weights = None
    if kl_buffer is not None and len(kl_buffer) == len(replay):
        kl = np.array(list(kl_buffer), dtype=np.float32)
        mean_kl = float(kl.mean()) + 1e-8
        w = np.clip(0.5 + 0.5 * kl / mean_kl, 0.1, 10.0)
        weights = torch.tensor(w, dtype=torch.float32)
    return dataset, weights


def effective_train_steps(cfg: TrainConfig, examples: int) -> int:
    if cfg.train_steps_per_iteration <= 0:
        return cfg.train_steps_per_iteration
    if cfg.max_train_replay_passes <= 0:
        return cfg.train_steps_per_iteration
    batches = max(1, math.ceil(examples / max(1, cfg.batch_size)))
    capped_steps = max(1, math.ceil(cfg.max_train_replay_passes * batches))
    return min(cfg.train_steps_per_iteration, capped_steps)


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: deque[Example],
    cfg: TrainConfig,
    scaler: object | None,
    dataset: TensorDataset | None = None,
    sample_weights: torch.Tensor | None = None,
    train_steps_to_run: int | None = None,
) -> TrainStats:
    if dataset is None:
        dataset, sample_weights = replay_to_dataset(
            replay, kl_buffer=None
        )
    use_weighted = cfg.surprise_weighting and sample_weights is not None
    # drop_last keeps batch sizes uniform under fixed step counts, but with a
    # buffer smaller than one batch it would yield an empty loader and skip the
    # iteration entirely — fall back to whatever batch the data can fill.
    drop_last = cfg.train_steps_per_iteration > 0 and len(dataset) >= cfg.batch_size
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "pin_memory": cfg.device.startswith("cuda"),
        "num_workers": max(0, cfg.train_data_workers),
        "drop_last": drop_last,
    }
    if cfg.train_data_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, cfg.train_prefetch_factor)
    if use_weighted:
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        loader = DataLoader(
            dataset,
            sampler=sampler,
            **loader_kwargs,
        )
    else:
        loader = DataLoader(
            dataset,
            shuffle=True,
            **loader_kwargs,
        )

    model.train()
    totals = TrainStats(lr=float(optimizer.param_groups[0]["lr"]))
    autocast_ctx = autocast_context(cfg.device, cfg.amp, cfg.amp_dtype)

    if train_steps_to_run is not None:
        steps_to_run = train_steps_to_run
    elif cfg.train_steps_per_iteration > 0:
        steps_to_run = cfg.train_steps_per_iteration
    else:
        steps_to_run = len(loader)
    loader_iter = iter(loader)
    policy_samples = 0.0
    for _ in range(steps_to_run):
        try:
            batch_states, batch_policies, batch_values, batch_policy_weights = next(loader_iter)
        except StopIteration:
            if cfg.train_steps_per_iteration <= 0:
                break
            loader_iter = iter(loader)
            try:
                batch_states, batch_policies, batch_values, batch_policy_weights = next(loader_iter)
            except StopIteration:
                # drop_last=True with fewer examples than one batch: nothing to train on
                print(
                    "train_warning empty_loader "
                    f"examples={len(dataset)} batch_size={cfg.batch_size}",
                    flush=True,
                )
                break

        batch_states = batch_states.to(cfg.device, non_blocking=True)
        batch_policies = batch_policies.to(cfg.device, non_blocking=True)
        batch_values = batch_values.to(cfg.device, non_blocking=True)
        batch_policy_weights = batch_policy_weights.to(cfg.device, non_blocking=True)

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

        if cfg.augment_symmetries:
            batch_states, batch_policies = apply_random_symmetries(
                batch_states, batch_policies, cfg.board_size
            )

        with autocast_ctx:
            logits, soft_logits, predicted_values = model(batch_states)
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
        # Policy losses and metrics only count full-search samples (policy_weight=1);
        # value-only samples from cheap searches contribute zero weight.
        pw = batch_policy_weights
        policy_count = pw.sum().clamp_min(1.0)
        log_probs = F.log_softmax(logits, dim=1)
        pred_probs = F.softmax(logits, dim=1)
        target_entropy_rows = -(batch_policies * torch.log(batch_policies.clamp_min(1.0e-8))).sum(dim=1)
        pred_entropy_rows = -(pred_probs * torch.log(pred_probs.clamp_min(1.0e-8))).sum(dim=1)
        target_entropy = (target_entropy_rows * pw).sum() / policy_count
        pred_entropy = (pred_entropy_rows * pw).sum() / policy_count
        policy_ce_rows = -(batch_policies * log_probs).sum(dim=1)
        policy_loss = (policy_ce_rows * pw).sum() / policy_count
        policy_kl_rows = (
            batch_policies * (torch.log(batch_policies.clamp_min(1.0e-8)) - log_probs)
        ).sum(dim=1)
        policy_kl = (policy_kl_rows * pw).sum() / policy_count
        value_loss = F.mse_loss(predicted_values, batch_values)
        loss = cfg.policy_loss_weight * policy_loss + cfg.value_loss_weight * value_loss

        # KataGo auxiliary soft policy loss: target = π^(1/T), teaches the network
        # to discriminate among non-top moves and speeds up learning
        soft_policy_loss = torch.tensor(0.0, device=cfg.device)
        if soft_logits is not None and cfg.soft_policy_loss_weight > 0:
            soft_logits_f = soft_logits.float()
            T = max(1e-3, cfg.soft_policy_temp)
            # soft target: π^(1/T) renormalised
            soft_target = batch_policies.pow(1.0 / T)
            soft_target = soft_target / soft_target.sum(dim=1, keepdim=True).clamp_min(1e-8)
            soft_log_probs = F.log_softmax(soft_logits_f, dim=1)
            soft_ce_rows = -(soft_target * soft_log_probs).sum(dim=1)
            soft_policy_loss = (soft_ce_rows * pw).sum() / policy_count
            loss = loss + cfg.soft_policy_loss_weight * soft_policy_loss

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
        pc = float(pw.sum().item())
        totals.loss += float(loss.item()) * batch_size
        totals.policy_loss += float(policy_loss.item()) * pc
        totals.soft_policy_loss += float(soft_policy_loss.item()) * pc
        totals.value_loss += float(value_loss.item()) * batch_size
        totals.policy_kl += float(policy_kl.item()) * pc
        totals.target_entropy += float(target_entropy.item()) * pc
        totals.pred_entropy += float(pred_entropy.item()) * pc
        totals.policy_top1 += float(((target_best == pred_best).float() * pw).sum().item())
        totals.value_mae += float((predicted_values.detach() - batch_values).abs().mean().item()) * batch_size
        totals.value_acc += float(value_acc_mean.item()) * batch_size
        totals.grad_norm += grad_norm_value * batch_size
        totals.batches += batch_size
        policy_samples += pc
        totals.optimizer_steps += 1

    denom = max(1, totals.batches)
    policy_denom = max(1.0, policy_samples)
    totals.loss /= denom
    totals.policy_loss /= policy_denom
    totals.soft_policy_loss /= policy_denom
    totals.value_loss /= denom
    totals.policy_kl /= policy_denom
    totals.target_entropy /= policy_denom
    totals.pred_entropy /= policy_denom
    totals.policy_top1 /= policy_denom
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
            root_policy_temp=cfg.mcts_root_policy_temp,
            shaped_dirichlet=cfg.mcts_shaped_dirichlet,
            dynamic_cpuct=cfg.mcts_dynamic_cpuct,
            fpu_reduction=cfg.mcts_fpu_reduction,
        ),
        device=device,
    ).search(state, add_exploration_noise=False)
    policy = visit_count_policy(root, state.action_size, temperature=0.0)
    if policy.sum() <= 0:
        legal = state.legal_actions()
        return int(legal[0])
    return int(policy.argmax())


def random_opening(cfg: TrainConfig, rng: random.Random) -> list[int]:
    """Uniform random legal opening moves for evaluation games."""
    state = GomokuState.new(size=cfg.board_size, win_length=cfg.win_length)
    opening: list[int] = []
    for _ in range(max(0, cfg.eval_opening_moves)):
        if state.is_terminal:
            break
        action = int(rng.choice(list(state.legal_actions())))
        opening.append(action)
        state = state.apply(action)
    return opening


def play_eval_game_worker(
    cfg_data: dict,
    candidate_state_path: str,
    baseline_state_path: str,
    jobs: list[tuple[int, list[int], int]],
    device: str,
    iteration: int = 0,
) -> list[dict[str, object]]:
    limit_worker_threads()
    enable_fast_math()
    cfg = TrainConfig(**cfg_data)
    cfg.device = device

    candidate = make_model(cfg).to(device)
    baseline = make_model(cfg).to(device)
    candidate.load_state_dict(load_torch(candidate_state_path, map_location="cpu"))
    baseline.load_state_dict(load_torch(baseline_state_path, map_location="cpu"))
    candidate.eval()
    baseline.eval()

    results: list[dict[str, object]] = []
    for game_index, opening, candidate_player in jobs:
        set_seed(cfg.seed + 50_000 + iteration * 1_000 + game_index)
        state = GomokuState.new(size=cfg.board_size, win_length=cfg.win_length)
        for action in opening:
            state = state.apply(action)
        while not state.is_terminal:
            if state.current_player == candidate_player:
                action = select_mcts_action(candidate, state, cfg.eval_simulations, device, cfg)
            else:
                action = select_mcts_action(baseline, state, cfg.eval_simulations, device, cfg)
            state = state.apply(action)

        if state.winner == 0:
            outcome = "draw"
        elif state.winner == candidate_player:
            outcome = "candidate"
        else:
            outcome = "baseline"
        results.append(
            {
                "game_index": game_index,
                "candidate_player": candidate_player,
                "outcome": outcome,
                "moves": state.moves_played,
                "device": device,
            }
        )
    return results


def evaluate_candidate(
    candidate: PolicyValueNet,
    baseline: PolicyValueNet | None,
    cfg: TrainConfig,
    rng: random.Random | None = None,
    iteration: int | None = None,
    end_iteration: int | None = None,
) -> dict[str, float]:
    """Pit candidate against baseline on randomised paired openings.

    Without openings the matchup is deterministic (greedy MCTS, no noise), so
    eval_games would collapse into 2 distinct games. Each random opening is
    played twice with colors swapped, cancelling first-move advantage.
    """
    if baseline is None or cfg.eval_games <= 0:
        return {
            "win_rate": 1.0,
            "wins": 0.0,
            "losses": 0.0,
            "draws": 0.0,
            "games": 0.0,
            "early_cutoff": 0.0,
        }

    rng = rng or random.Random(random.getrandbits(64))
    openings = [random_opening(cfg, rng) for _ in range((cfg.eval_games + 1) // 2)]
    iter_label = (
        f" iter={iteration}/{end_iteration}"
        if iteration is not None and end_iteration is not None
        else ""
    )
    eval_start = time.monotonic()
    print(
        f"eval_start{iter_label} games={cfg.eval_games} "
        f"simulations={cfg.eval_simulations} opening_moves={cfg.eval_opening_moves} "
        f"threshold={cfg.promotion_threshold:.3f} "
        f"early_cutoff={cfg.eval_early_cutoff} "
        f"workers={cfg.eval_workers} devices={split_devices(cfg.eval_devices, cfg.device)}",
        flush=True,
    )
    candidate.eval()
    baseline.eval()

    if cfg.eval_workers > 1:
        worker_count = min(max(1, cfg.eval_workers), cfg.eval_games)
        devices = split_devices(cfg.eval_devices, cfg.device)
        game_counts = distribute_games(cfg.eval_games, worker_count)
        game_jobs: list[list[tuple[int, list[int], int]]] = []
        next_game = 0
        for count in game_counts:
            jobs: list[tuple[int, list[int], int]] = []
            for _ in range(count):
                jobs.append(
                    (
                        next_game,
                        openings[next_game // 2],
                        1 if next_game % 2 == 0 else -1,
                    )
                )
                next_game += 1
            game_jobs.append(jobs)
        worker_devices = [devices[index % len(devices)] for index in range(worker_count)]
        snapshot_dir = (
            Path(cfg.checkpoint_dir)
            / "_eval_tmp"
            / f"iter_{iteration or 0:04d}_{int(time.time() * 1000)}"
        )
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        candidate_state_path = str(snapshot_dir / "candidate.pt")
        baseline_state_path = str(snapshot_dir / "baseline.pt")
        torch.save(cpu_state_dict(candidate), candidate_state_path)
        torch.save(cpu_state_dict(baseline), baseline_state_path)
        cfg_data = asdict(cfg)
        context = mp.get_context("spawn")
        wins = losses = draws = played = 0
        if cfg.eval_early_cutoff:
            print(
                f"eval_parallel{iter_label} early_cutoff_disabled=true "
                f"reason=all_games_already_dispatched",
                flush=True,
            )
        if sys.platform == "win32":
            executor_context = concurrent.futures.ThreadPoolExecutor(max_workers=worker_count)
        else:
            executor_context = concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count, mp_context=context
            )
        try:
            with executor_context as executor:
                futures = [
                    executor.submit(
                        play_eval_game_worker,
                        cfg_data,
                        candidate_state_path,
                        baseline_state_path,
                        jobs,
                        worker_devices[worker_id],
                        iteration or 0,
                    )
                    for worker_id, jobs in enumerate(game_jobs)
                    if jobs
                ]
                pending = set(futures)
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=30,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        print(
                            f"eval_wait{iter_label} games={played}/{cfg.eval_games} "
                            f"elapsed={format_duration(time.monotonic() - eval_start)}",
                            flush=True,
                        )
                        continue
                    for future in done:
                        for result in future.result():
                            outcome = str(result["outcome"])
                            if outcome == "draw":
                                draws += 1
                            elif outcome == "candidate":
                                wins += 1
                            else:
                                losses += 1
                            played += 1
                            score = wins + 0.5 * draws
                            candidate_player = int(result["candidate_player"])
                            if cfg.eval_progress_interval > 0 and (
                                played % cfg.eval_progress_interval == 0
                                or played == cfg.eval_games
                            ):
                                print(
                                    f"eval_game{iter_label} game={played}/{cfg.eval_games} "
                                    f"source_game={int(result['game_index']) + 1} "
                                    f"device={result['device']} "
                                    f"candidate_color={'black' if candidate_player == 1 else 'white'} "
                                    f"outcome={outcome} moves={int(result['moves'])} "
                                    f"score={score / max(1, cfg.eval_games):.3f} "
                                    f"wins={wins} losses={losses} draws={draws} "
                                    f"elapsed={format_duration(time.monotonic() - eval_start)}",
                                    flush=True,
                                )
            win_rate = (wins + 0.5 * draws) / max(1, cfg.eval_games)
            print(
                f"eval_done{iter_label} played={played}/{cfg.eval_games} "
                f"score={win_rate:.3f} wins={wins} losses={losses} draws={draws} "
                f"early_cutoff=False elapsed={format_duration(time.monotonic() - eval_start)}",
                flush=True,
            )
            return {
                "win_rate": float(win_rate),
                "wins": float(wins),
                "losses": float(losses),
                "draws": float(draws),
                "games": float(played),
                "early_cutoff": 0.0,
            }
        finally:
            shutil.rmtree(snapshot_dir, ignore_errors=True)

    wins = losses = draws = 0
    played = 0
    early_cutoff = False
    cutoff_reason = ""
    threshold_score = cfg.promotion_threshold * cfg.eval_games
    for game_index in range(cfg.eval_games):
        state = GomokuState.new(size=cfg.board_size, win_length=cfg.win_length)
        for action in openings[game_index // 2]:
            state = state.apply(action)
        candidate_player = 1 if game_index % 2 == 0 else -1
        while not state.is_terminal:
            if state.current_player == candidate_player:
                action = select_mcts_action(candidate, state, cfg.eval_simulations, cfg.device, cfg)
            else:
                action = select_mcts_action(baseline, state, cfg.eval_simulations, cfg.device, cfg)
            state = state.apply(action)
        if state.winner == 0:
            draws += 1
            outcome = "draw"
        elif state.winner == candidate_player:
            wins += 1
            outcome = "candidate"
        else:
            losses += 1
            outcome = "baseline"
        played += 1
        score = wins + 0.5 * draws
        remaining = cfg.eval_games - played
        if cfg.eval_progress_interval > 0 and (
            played % cfg.eval_progress_interval == 0 or played == cfg.eval_games
        ):
            print(
                f"eval_game{iter_label} game={played}/{cfg.eval_games} "
                f"candidate_color={'black' if candidate_player == 1 else 'white'} "
                f"outcome={outcome} moves={state.moves_played} "
                f"score={score / max(1, cfg.eval_games):.3f} "
                f"wins={wins} losses={losses} draws={draws} "
                f"elapsed={format_duration(time.monotonic() - eval_start)}",
                flush=True,
            )
        if cfg.eval_early_cutoff:
            if score + remaining < threshold_score:
                early_cutoff = True
                cutoff_reason = "cannot_reach_threshold"
            elif score >= threshold_score:
                early_cutoff = True
                cutoff_reason = "already_above_threshold"
            if early_cutoff:
                print(
                    f"eval_cutoff{iter_label} reason={cutoff_reason} "
                    f"played={played}/{cfg.eval_games} "
                    f"score={score / max(1, cfg.eval_games):.3f} "
                    f"threshold={cfg.promotion_threshold:.3f} "
                    f"wins={wins} losses={losses} draws={draws} "
                    f"elapsed={format_duration(time.monotonic() - eval_start)}",
                    flush=True,
                )
                break
    win_rate = (wins + 0.5 * draws) / max(1, cfg.eval_games)
    print(
        f"eval_done{iter_label} played={played}/{cfg.eval_games} "
        f"score={win_rate:.3f} wins={wins} losses={losses} draws={draws} "
        f"early_cutoff={early_cutoff} elapsed={format_duration(time.monotonic() - eval_start)}",
        flush=True,
    )
    return {
        "win_rate": float(win_rate),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "games": float(played),
        "early_cutoff": float(early_cutoff),
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
    tmp = path.with_name(f"{path.name}.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "model_kwargs": architecture_from_config(cfg),
            "iteration": iteration,
            "stats": asdict(stats) if stats is not None else None,
        },
        tmp,
    )
    tmp.replace(path)
    return path


def best_checkpoint_path(cfg: TrainConfig) -> Path:
    return Path(cfg.checkpoint_dir) / "gomoku10_best.pt"


def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    enable_fast_math()
    if cfg.self_play_workers > 1 or cfg.eval_workers > 1:
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
        f"train_data_workers={cfg.train_data_workers} "
        f"train_prefetch_factor={cfg.train_prefetch_factor} "
        f"lr={cfg.learning_rate} min_lr={cfg.min_learning_rate} "
        f"lr_schedule={cfg.lr_schedule} warmup_iterations={cfg.warmup_iterations} "
        f"mcts_c_puct={cfg.mcts_c_puct} "
        f"dirichlet_alpha={cfg.mcts_dirichlet_alpha} "
        f"dirichlet_fraction={cfg.mcts_dirichlet_fraction} "
        f"channels={cfg.channels} residual_blocks={cfg.residual_blocks} "
        f"policy_channels={cfg.policy_channels} value_channels={cfg.value_channels} "
        f"value_hidden={cfg.value_hidden} use_se={cfg.use_se} "
        f"augment_symmetries={cfg.augment_symmetries} "
        f"eval_workers={cfg.eval_workers} "
        f"eval_devices={split_devices(cfg.eval_devices, cfg.device)} "
        f"amp={cfg.amp} amp_dtype={cfg.amp_dtype} mcts_amp_dtype={cfg.mcts_amp_dtype} "
        f"metrics_path={cfg.metrics_path or 'none'}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        unwrap_model(model).parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    start_iteration = load_resume_checkpoint(base_model, optimizer, cfg)
    end_iteration = start_iteration + cfg.iterations
    replay, kl_buffer = load_replay(cfg)
    if cfg.playout_cap_randomization and cfg.fast_simulations <= 0:
        cfg.fast_simulations = max(1, cfg.simulations // 4)
    mcts_cfg = MCTSConfig(
        simulations=cfg.simulations,
        c_puct=cfg.mcts_c_puct,
        dirichlet_alpha=cfg.mcts_dirichlet_alpha,
        dirichlet_fraction=cfg.mcts_dirichlet_fraction,
        eval_batch_size=cfg.mcts_batch_size,
        amp_dtype=cfg.mcts_amp_dtype,
        root_policy_temp=cfg.mcts_root_policy_temp,
        shaped_dirichlet=cfg.mcts_shaped_dirichlet,
        dynamic_cpuct=cfg.mcts_dynamic_cpuct,
        fpu_reduction=cfg.mcts_fpu_reduction,
        forced_playouts=cfg.mcts_forced_playouts,
        forced_playout_k=cfg.mcts_forced_playout_k,
    )
    scaler = make_grad_scaler(cfg.amp and cfg.device.startswith("cuda"), cfg.amp_dtype)
    champion = copy.deepcopy(base_model).to(cfg.device) if cfg.eval_interval and cfg.eval_games else None
    if cfg.early_stop_evals > 0 and champion is None:
        print(
            "config_warning early_stop_disabled reason=evaluation_not_enabled "
            "(set eval_interval and eval_games)",
            flush=True,
        )
    failed_evals = 0
    run_start = time.monotonic()

    for iteration in range(start_iteration + 1, end_iteration + 1):
        completed_this_run = iteration - start_iteration
        iteration_start = time.monotonic()
        set_optimizer_lr(optimizer, scheduled_learning_rate(cfg, iteration, end_iteration))
        print(f"iter={iteration}/{end_iteration} phase=selfplay_start", flush=True)
        examples, kl_surprises, lengths, selfplay_stats = collect_self_play_examples(base_model, cfg, mcts_cfg, iteration)
        raw_examples = len(examples)
        policy_examples = int(sum(example[3] for example in examples))
        replay.extend(examples)
        kl_buffer.extend(kl_surprises)
        new_examples = len(examples)
        print(
            f"iter={iteration}/{end_iteration} phase=selfplay_done "
            f"raw_examples={raw_examples} policy_examples={policy_examples} "
            f"avg_moves={np.mean(lengths):.1f} replay={len(replay)} "
            f"target_policy_entropy={selfplay_stats['policy_entropy']:.4f} "
            f"black_win_rate={selfplay_stats['black_win_rate']:.3f} "
            f"white_win_rate={selfplay_stats['white_win_rate']:.3f} "
            f"draw_rate={selfplay_stats['draw_rate']:.3f} "
            f"full_search_rate={selfplay_stats['full_search_rate']:.3f} "
            f"elapsed={format_duration(time.monotonic() - iteration_start)}",
            flush=True,
        )

        stats = TrainStats(lr=float(optimizer.param_groups[0]["lr"]))
        train_skipped_min_replay = len(replay) < cfg.min_replay_size
        train_steps_to_run = cfg.train_steps_per_iteration
        if train_skipped_min_replay:
            print(
                "train_skip "
                f"iter={iteration}/{end_iteration} reason=min_replay "
                f"replay={len(replay)} min_replay_size={cfg.min_replay_size}",
                flush=True,
            )
        else:
            train_dataset, train_weights = replay_to_dataset(
                replay, kl_buffer if cfg.surprise_weighting else None
            )
            train_steps_to_run = effective_train_steps(cfg, len(train_dataset))
            if train_steps_to_run != cfg.train_steps_per_iteration:
                print(
                    "train_steps_capped "
                    f"iter={iteration}/{end_iteration} requested={cfg.train_steps_per_iteration} "
                    f"effective={train_steps_to_run} replay={len(train_dataset)} "
                    f"batch_size={cfg.batch_size} max_train_replay_passes={cfg.max_train_replay_passes}",
                    flush=True,
                )
            if cfg.surprise_weighting and train_weights is None:
                print(
                    "train_warning surprise_weighting_disabled reason=kl_buffer_mismatch "
                    f"kl={len(kl_buffer)} replay={len(replay)}",
                    flush=True,
                )
            for epoch in range(1, cfg.epochs + 1):
                epoch_start = time.monotonic()
                stats = train_epoch(
                    model,
                    optimizer,
                    replay,
                    cfg,
                    scaler,
                    train_dataset,
                    train_weights,
                    train_steps_to_run=train_steps_to_run,
                )
                print(
                    f"train_progress iter={iteration}/{end_iteration} "
                    f"epoch={epoch}/{cfg.epochs} elapsed={format_duration(time.monotonic() - epoch_start)} "
                    f"loss={stats.loss:.4f} policy={stats.policy_loss:.4f} "
                    f"soft_policy={stats.soft_policy_loss:.4f} "
                    f"value={stats.value_loss:.4f} policy_kl={stats.policy_kl:.4f} "
                    f"target_entropy={stats.target_entropy:.4f} pred_entropy={stats.pred_entropy:.4f} "
                    f"policy_top1={stats.policy_top1:.4f} value_mae={stats.value_mae:.4f} "
                    f"value_acc={stats.value_acc:.4f} grad_norm={stats.grad_norm:.4f} "
                    f"steps={stats.optimizer_steps} lr={stats.lr:.6g} "
                    f"skipped={stats.skipped} value_clamps={stats.value_clamps}",
                    flush=True,
                )

        eval_stats = {
            "win_rate": 1.0,
            "wins": 0.0,
            "losses": 0.0,
            "draws": 0.0,
            "games": 0.0,
            "early_cutoff": 0.0,
        }
        promoted = True
        evaluated = (
            champion is not None
            and not train_skipped_min_replay
            and cfg.eval_interval > 0
            and iteration % cfg.eval_interval == 0
        )
        stop_early = False
        if evaluated:
            eval_stats = evaluate_candidate(
                base_model,
                champion,
                cfg,
                rng=random.Random(cfg.seed + iteration * 7919),
                iteration=iteration,
                end_iteration=end_iteration,
            )
            promoted = eval_stats["win_rate"] >= cfg.promotion_threshold
            print(
                f"eval_progress iter={iteration}/{end_iteration} "
                f"candidate_score={eval_stats['win_rate']:.3f} "
                f"wins={int(eval_stats['wins'])} losses={int(eval_stats['losses'])} "
                f"draws={int(eval_stats['draws'])} games={int(eval_stats['games'])}/{cfg.eval_games} "
                f"early_cutoff={bool(eval_stats['early_cutoff'])} "
                f"threshold={cfg.promotion_threshold:.3f} "
                f"promoted={promoted}",
                flush=True,
            )
            if promoted:
                champion.load_state_dict(base_model.state_dict())
                failed_evals = 0
            else:
                failed_evals += 1
                if cfg.gate_evaluation:
                    base_model.load_state_dict(champion.state_dict())
                    print(f"eval_reverted iter={iteration} reason=below_threshold", flush=True)
            if cfg.early_stop_evals > 0 and failed_evals >= cfg.early_stop_evals:
                stop_early = True

        checkpoint = save_checkpoint(base_model, optimizer, cfg, iteration, stats)
        # With gating enabled, "best" means the champion line: only update it on a
        # real promotion. Without evaluation there is no champion, so best tracks
        # the latest checkpoint as before.
        update_best = (evaluated and promoted) if champion is not None else True
        if update_best:
            best_path = best_checkpoint_path(cfg)
            best_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_best = best_path.with_name(f"{best_path.name}.tmp")
            shutil.copyfile(checkpoint, tmp_best)
            tmp_best.replace(best_path)
        if cfg.replay_path and cfg.replay_save_interval > 0 and iteration % cfg.replay_save_interval == 0:
            save_replay(replay, kl_buffer, cfg.replay_path)

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
                "policy_examples": policy_examples,
                "full_search_rate": selfplay_stats["full_search_rate"],
                "avg_moves": float(np.mean(lengths)),
                "replay": len(replay),
                "loss": stats.loss,
                "policy_loss": stats.policy_loss,
                "soft_policy_loss": stats.soft_policy_loss,
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
                "train_skipped_min_replay": train_skipped_min_replay,
                "train_steps_effective": train_steps_to_run,
                "min_replay_size": cfg.min_replay_size,
                "max_train_replay_passes": cfg.max_train_replay_passes,
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
                "eval_games": eval_stats["games"],
                "eval_early_cutoff": bool(eval_stats["early_cutoff"]),
                "evaluated": evaluated,
                "promoted": promoted,
                "failed_evals": failed_evals,
                "early_stopped": stop_early,
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
        if stop_early:
            print(
                f"early_stop iter={iteration}/{end_iteration} "
                f"consecutive_failed_evals={failed_evals} "
                f"threshold={cfg.promotion_threshold:.3f} "
                f"reason=candidate_not_beating_champion",
                flush=True,
            )
            break

    save_replay(replay, kl_buffer, cfg.replay_path)


def build_parser(defaults: TrainConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a 10x10 Gomoku AI with AlphaZero-style self-play."
    )
    parser.add_argument(
        "--preset",
        choices=[
            "local",
            "v2",
            "v3-local",
            "v3-student-local",
            "a100-4",
            "a100-fast",
            "a100-turbo",
            "a100-prod",
        ],
        default=defaults.preset,
    )
    parser.add_argument("--board-size", type=int, default=defaults.board_size)
    parser.add_argument("--win-length", type=int, default=defaults.win_length)
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--games-per-iteration", type=int, default=defaults.games_per_iteration)
    parser.add_argument("--simulations", type=int, default=defaults.simulations)
    parser.add_argument("--mcts-batch-size", type=int, default=defaults.mcts_batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--train-steps-per-iteration", type=int, default=defaults.train_steps_per_iteration)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--train-data-workers", type=int, default=defaults.train_data_workers)
    parser.add_argument("--train-prefetch-factor", type=int, default=defaults.train_prefetch_factor)
    parser.add_argument("--replay-size", type=int, default=defaults.replay_size)
    parser.add_argument("--min-replay-size", type=int, default=defaults.min_replay_size)
    parser.add_argument("--max-train-replay-passes", type=float, default=defaults.max_train_replay_passes)
    parser.add_argument("--channels", type=int, default=defaults.channels)
    parser.add_argument("--residual-blocks", type=int, default=defaults.residual_blocks)
    parser.add_argument("--policy-channels", type=int, default=defaults.policy_channels)
    parser.add_argument("--value-channels", type=int, default=defaults.value_channels)
    parser.add_argument("--value-hidden", type=int, default=defaults.value_hidden)
    parser.add_argument("--use-se", dest="use_se", action="store_true")
    parser.add_argument("--no-use-se", dest="use_se", action="store_false")
    parser.add_argument("--use-global-pool", dest="use_global_pool", action="store_true")
    parser.add_argument("--no-use-global-pool", dest="use_global_pool", action="store_false")
    parser.add_argument("--use-soft-policy", dest="use_soft_policy", action="store_true")
    parser.add_argument("--no-use-soft-policy", dest="use_soft_policy", action="store_false")
    parser.add_argument("--soft-policy-loss-weight", type=float, default=defaults.soft_policy_loss_weight)
    parser.add_argument("--soft-policy-temp", type=float, default=defaults.soft_policy_temp)
    parser.add_argument("--surprise-weighting", dest="surprise_weighting", action="store_true")
    parser.add_argument("--no-surprise-weighting", dest="surprise_weighting", action="store_false")
    parser.add_argument("--mcts-value-weight", type=float, default=defaults.mcts_value_weight)
    parser.add_argument("--mcts-root-policy-temp", type=float, default=defaults.mcts_root_policy_temp)
    parser.add_argument("--mcts-shaped-dirichlet", dest="mcts_shaped_dirichlet", action="store_true")
    parser.add_argument("--no-mcts-shaped-dirichlet", dest="mcts_shaped_dirichlet", action="store_false")
    parser.add_argument("--mcts-dynamic-cpuct", dest="mcts_dynamic_cpuct", action="store_true")
    parser.add_argument("--no-mcts-dynamic-cpuct", dest="mcts_dynamic_cpuct", action="store_false")
    parser.add_argument("--mcts-fpu-reduction", type=float, default=defaults.mcts_fpu_reduction)
    parser.add_argument("--mcts-forced-playouts", dest="mcts_forced_playouts", action="store_true")
    parser.add_argument("--no-mcts-forced-playouts", dest="mcts_forced_playouts", action="store_false")
    parser.add_argument("--mcts-forced-playout-k", type=float, default=defaults.mcts_forced_playout_k)
    parser.add_argument(
        "--playout-cap-randomization", dest="playout_cap_randomization", action="store_true"
    )
    parser.add_argument(
        "--no-playout-cap-randomization", dest="playout_cap_randomization", action="store_false"
    )
    parser.add_argument("--full-search-prob", type=float, default=defaults.full_search_prob)
    parser.add_argument("--fast-simulations", type=int, default=defaults.fast_simulations)
    parser.add_argument("--selfplay-tree-reuse", dest="selfplay_tree_reuse", action="store_true")
    parser.add_argument("--no-selfplay-tree-reuse", dest="selfplay_tree_reuse", action="store_false")
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
    parser.add_argument("--eval-opening-moves", type=int, default=defaults.eval_opening_moves)
    parser.add_argument("--eval-progress-interval", type=int, default=defaults.eval_progress_interval)
    parser.add_argument("--eval-workers", type=int, default=defaults.eval_workers)
    parser.add_argument("--eval-devices", default=defaults.eval_devices)
    parser.add_argument("--eval-early-cutoff", dest="eval_early_cutoff", action="store_true")
    parser.add_argument("--no-eval-early-cutoff", dest="eval_early_cutoff", action="store_false")
    parser.add_argument("--early-stop-evals", type=int, default=defaults.early_stop_evals)
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
        use_global_pool=defaults.use_global_pool,
        use_soft_policy=defaults.use_soft_policy,
        surprise_weighting=defaults.surprise_weighting,
        mcts_shaped_dirichlet=defaults.mcts_shaped_dirichlet,
        mcts_dynamic_cpuct=defaults.mcts_dynamic_cpuct,
        mcts_forced_playouts=defaults.mcts_forced_playouts,
        playout_cap_randomization=defaults.playout_cap_randomization,
        selfplay_tree_reuse=defaults.selfplay_tree_reuse,
        gate_evaluation=defaults.gate_evaluation,
        eval_early_cutoff=defaults.eval_early_cutoff,
        compile_model=defaults.compile_model,
        amp=defaults.amp,
    )
    return parser


def main() -> None:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument(
        "--preset",
        choices=[
            "local",
            "v2",
            "v3-local",
            "v3-student-local",
            "a100-4",
            "a100-fast",
            "a100-turbo",
            "a100-prod",
        ],
        default="local",
    )
    preset_args, _ = preset_parser.parse_known_args()
    args = build_parser(preset_config(preset_args.preset)).parse_args()
    cfg = TrainConfig(**vars(args))
    run_training(cfg)


if __name__ == "__main__":
    main()
