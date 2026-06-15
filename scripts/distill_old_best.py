from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from alphazero_gomoku.game import GomokuState
from alphazero_gomoku.inference import mcts_config_from_cfg
from alphazero_gomoku.model import build_model_from_config, model_kwargs_from_config
from alphazero_gomoku.torch_compat import tensor_from_array
from alphazero_gomoku.train import TrainConfig
from alphazero_gomoku.utils import cpu_state_dict, format_duration, load_model, resolve_device


def default_path(*parts: str) -> Path:
    return REPO_DIR.joinpath(*parts)


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = max(1.0e-6, float(temperature))
    masked = logits.float() / temp
    masked = masked.masked_fill(~legal_mask, -1.0e9)
    return F.softmax(masked, dim=-1)


def numpy_policy_from_logits(
    logits: torch.Tensor,
    legal_mask: np.ndarray,
    temperature: float,
) -> np.ndarray:
    legal = torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device)
    policy = masked_softmax(logits, legal, temperature)
    return policy.detach().cpu().numpy().astype(np.float32)


def sample_action(policy: np.ndarray, rng: random.Random) -> int:
    total = float(policy.sum())
    if total <= 0:
        raise ValueError("cannot sample from an empty policy")
    return int(rng.choices(range(len(policy)), weights=policy, k=1)[0])


def teacher_raw_target(
    teacher: torch.nn.Module,
    state: GomokuState,
    *,
    device: str,
    policy_temperature: float,
) -> tuple[np.ndarray, float]:
    encoded = tensor_from_array(state.encode()[None, ...], dtype=torch.float32, device=device)
    with torch.inference_mode():
        logits, _, value = teacher(encoded)
    policy = numpy_policy_from_logits(logits[0], state.legal_mask(), policy_temperature)
    return policy, float(value[0].detach().cpu().item())


def teacher_mcts_target(
    teacher: torch.nn.Module,
    teacher_cfg: dict,
    state: GomokuState,
    *,
    device: str,
    simulations: int,
) -> tuple[np.ndarray, float]:
    search = MCTS(
        teacher,
        mcts_config_from_cfg(teacher_cfg, simulations),
        device=device,
    )
    root = search.search(state, add_exploration_noise=False)
    policy = search.policy_target(root, state.action_size, pruned=True)
    if policy.sum() <= 0:
        policy, value = teacher_raw_target(
            teacher, state, device=device, policy_temperature=1.0
        )
        return policy, value
    return policy.astype(np.float32), float(root.value)


def generate_distill_examples(
    teacher: torch.nn.Module,
    teacher_cfg: dict,
    args: argparse.Namespace,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    rng = random.Random(args.seed)
    states: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    mcts_targets = 0
    start = time.monotonic()

    for game_index in range(args.games):
        state = GomokuState.new(size=args.board_size, win_length=args.win_length)
        for _ in range(args.random_opening_moves):
            if state.is_terminal:
                break
            action = int(rng.choice(list(state.legal_actions())))
            state = state.apply(action)

        while not state.is_terminal and state.moves_played < args.max_moves:
            use_mcts = args.teacher_sims > 0 and rng.random() < args.mcts_target_prob
            if use_mcts:
                policy, value = teacher_mcts_target(
                    teacher,
                    teacher_cfg,
                    state,
                    device=device,
                    simulations=args.teacher_sims,
                )
                mcts_targets += 1
            else:
                policy, value = teacher_raw_target(
                    teacher,
                    state,
                    device=device,
                    policy_temperature=args.teacher_policy_temp,
                )

            states.append(state.encode().astype(np.float32))
            legal_masks.append(state.legal_mask())
            policies.append(policy)
            values.append(float(np.clip(value, -1.0, 1.0)))

            move_policy = policy.copy()
            if args.move_temperature != args.teacher_policy_temp:
                legal = state.legal_mask()
                logits = np.full_like(move_policy, -1.0e9, dtype=np.float32)
                logits[legal] = np.log(np.clip(move_policy[legal], 1.0e-12, 1.0))
                logits[legal] /= max(1.0e-6, args.move_temperature)
                logits[legal] -= logits[legal].max()
                exp = np.exp(logits[legal])
                move_policy[:] = 0.0
                move_policy[legal] = exp / max(float(exp.sum()), 1.0e-12)
            action = sample_action(move_policy, rng)
            state = state.apply(action)

        print(
            "distill_generate_game "
            f"game={game_index + 1}/{args.games} moves={state.moves_played} "
            f"examples={len(states)} mcts_targets={mcts_targets} "
            f"elapsed={format_duration(time.monotonic() - start)}",
            flush=True,
        )

    metadata = {
        "examples": float(len(states)),
        "mcts_targets": float(mcts_targets),
        "seconds": time.monotonic() - start,
    }
    return (
        np.stack(states),
        np.stack(legal_masks),
        np.stack(policies),
        np.asarray(values, dtype=np.float32),
        metadata,
    )


def student_train_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig(preset="distill-oldbest-light")
    cfg.board_size = args.board_size
    cfg.win_length = args.win_length
    cfg.channels = args.channels
    cfg.residual_blocks = args.residual_blocks
    cfg.policy_channels = args.policy_channels
    cfg.value_channels = args.value_channels
    cfg.value_hidden = args.value_hidden
    cfg.use_global_pool = args.use_global_pool
    cfg.use_soft_policy = args.use_soft_policy
    cfg.checkpoint_dir = str(args.checkpoint_dir)
    cfg.learning_rate = args.learning_rate
    cfg.batch_size = args.batch_size
    cfg.weight_decay = args.weight_decay
    cfg.max_grad_norm = args.max_grad_norm
    cfg.device = args.device
    return cfg


def save_student_checkpoint(
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    args: argparse.Namespace,
    *,
    epoch: int,
    stats: dict[str, float],
    best: bool,
) -> Path:
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = args.checkpoint_dir / f"gomoku10_student_epoch_{epoch:04d}.pt"
    config = asdict(cfg)
    config["distillation"] = {
        "teacher": str(args.teacher),
        "student_resume": str(args.student_resume) if args.student_resume else "",
        "teacher_sims": args.teacher_sims,
        "mcts_target_prob": args.mcts_target_prob,
        "teacher_policy_temp": args.teacher_policy_temp,
        "move_temperature": args.move_temperature,
    }
    torch.save(
        {
            "model": cpu_state_dict(student),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "model_kwargs": model_kwargs_from_config(asdict(cfg)),
            "iteration": epoch,
            "stats": stats,
        },
        path,
    )
    final_path = args.checkpoint_dir / "gomoku10_student_final.pt"
    shutil.copyfile(path, final_path)
    if best:
        best_path = args.checkpoint_dir / "gomoku10_student_best.pt"
        shutil.copyfile(path, best_path)
    return path


def append_metrics(path: Path, payload: dict[str, object]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def train_student(
    student: torch.nn.Module,
    args: argparse.Namespace,
    cfg: TrainConfig,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    device: str,
) -> None:
    states_np, masks_np, policies_np, values_np = arrays
    states = tensor_from_array(states_np, dtype=torch.float32)
    masks = torch.as_tensor(masks_np, dtype=torch.bool)
    policies = tensor_from_array(policies_np, dtype=torch.float32)
    values = torch.as_tensor(values_np, dtype=torch.float32)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_loss = math.inf
    start = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        student.train()
        permutation = torch.randperm(states.shape[0])
        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "soft_policy_loss": 0.0,
            "value_loss": 0.0,
            "policy_kl": 0.0,
            "policy_top1": 0.0,
            "value_mae": 0.0,
        }
        examples_seen = 0
        batches = 0

        for start_index in range(0, states.shape[0], args.batch_size):
            batch_indices = permutation[start_index : start_index + args.batch_size]
            batch_states = states[batch_indices].to(device, non_blocking=True)
            batch_masks = masks[batch_indices].to(device, non_blocking=True)
            batch_policies = policies[batch_indices].to(device, non_blocking=True)
            batch_values = values[batch_indices].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            policy_logits, soft_logits, predicted_values = student(batch_states)
            legal_logits = policy_logits.float().masked_fill(~batch_masks, -1.0e9)
            log_probs = F.log_softmax(legal_logits, dim=1)
            policy_loss_rows = -(batch_policies * log_probs).sum(dim=1)
            policy_loss = policy_loss_rows.mean()
            value_loss = F.mse_loss(predicted_values.float(), batch_values)
            loss = args.policy_loss_weight * policy_loss + args.value_loss_weight * value_loss

            soft_policy_loss = torch.tensor(0.0, device=device)
            if soft_logits is not None and args.soft_policy_loss_weight > 0:
                soft_base = batch_policies.clamp_min(1.0e-8).masked_fill(~batch_masks, 0.0)
                soft_target = soft_base.pow(1.0 / max(1.0e-6, args.soft_policy_temp))
                soft_target = soft_target / soft_target.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
                soft_log_probs = F.log_softmax(
                    soft_logits.float().masked_fill(~batch_masks, -1.0e9),
                    dim=1,
                )
                soft_policy_loss = -(soft_target * soft_log_probs).sum(dim=1).mean()
                loss = loss + args.soft_policy_loss_weight * soft_policy_loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise RuntimeError(f"non-finite grad norm: {grad_norm}")
            optimizer.step()

            with torch.no_grad():
                probs = log_probs.exp()
                policy_kl_rows = (
                    batch_policies
                    * (batch_policies.clamp_min(1.0e-8).log() - log_probs)
                ).sum(dim=1)
                target_best = batch_policies.argmax(dim=1)
                pred_best = probs.argmax(dim=1)
                count = int(batch_states.shape[0])
                totals["loss"] += float(loss.detach().cpu()) * count
                totals["policy_loss"] += float(policy_loss.detach().cpu()) * count
                totals["soft_policy_loss"] += float(soft_policy_loss.detach().cpu()) * count
                totals["value_loss"] += float(value_loss.detach().cpu()) * count
                totals["policy_kl"] += float(policy_kl_rows.mean().detach().cpu()) * count
                totals["policy_top1"] += float((target_best == pred_best).float().mean().detach().cpu()) * count
                totals["value_mae"] += float((predicted_values - batch_values).abs().mean().detach().cpu()) * count
                examples_seen += count
                batches += 1

        stats = {name: value / max(1, examples_seen) for name, value in totals.items()}
        stats.update(
            {
                "epoch": epoch,
                "examples": int(states.shape[0]),
                "batches": batches,
                "seconds": time.monotonic() - start,
                "lr": args.learning_rate,
            }
        )
        is_best = stats["loss"] < best_loss
        if is_best:
            best_loss = stats["loss"]
        checkpoint = save_student_checkpoint(
            student, optimizer, cfg, args, epoch=epoch, stats=stats, best=is_best
        )
        append_metrics(args.metrics_path, stats | {"checkpoint": str(checkpoint), "best": is_best})
        print(
            "distill_epoch "
            f"epoch={epoch}/{args.epochs} loss={stats['loss']:.4f} "
            f"policy={stats['policy_loss']:.4f} soft_policy={stats['soft_policy_loss']:.4f} "
            f"value={stats['value_loss']:.4f} policy_kl={stats['policy_kl']:.4f} "
            f"policy_top1={stats['policy_top1']:.4f} value_mae={stats['value_mae']:.4f} "
            f"best={is_best} saved={checkpoint} elapsed={format_duration(stats['seconds'])}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill the old best Gomoku checkpoint into a lighter student model."
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=default_path("outputs", "checkpoints", "v1-old-best", "gomoku10_best.pt"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=default_path("outputs", "checkpoints", "distill-oldbest-128x8"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=default_path("outputs", "metrics", "distill-oldbest-128x8.jsonl"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--student-resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--board-size", type=int, default=10)
    parser.add_argument("--win-length", type=int, default=5)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--max-moves", type=int, default=100)
    parser.add_argument("--random-opening-moves", type=int, default=8)
    parser.add_argument("--teacher-policy-temp", type=float, default=1.0)
    parser.add_argument("--move-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-sims", type=int, default=0)
    parser.add_argument("--mcts-target-prob", type=float, default=0.0)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--residual-blocks", type=int, default=8)
    parser.add_argument("--policy-channels", type=int, default=12)
    parser.add_argument("--value-channels", type=int, default=6)
    parser.add_argument("--value-hidden", type=int, default=384)
    parser.add_argument("--use-global-pool", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-soft-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--soft-policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--soft-policy-temp", type=float, default=4.0)
    args = parser.parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be positive")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not (0.0 <= args.mcts_target_prob <= 1.0):
        raise SystemExit("--mcts-target-prob must be in [0, 1]")
    if args.teacher_sims < 0:
        raise SystemExit("--teacher-sims must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    teacher, teacher_cfg = load_model(args.teacher, device)
    teacher.eval()
    cfg = student_train_config(args)
    cfg.device = device
    student = build_model_from_config(asdict(cfg)).to(device)
    if args.student_resume:
        payload = torch.load(args.student_resume, map_location=device, weights_only=False)
        student.load_state_dict(payload["model"])
        print(f"distill_student_resume path={args.student_resume}", flush=True)

    print(
        "distill_start "
        f"teacher={args.teacher} device={device} games={args.games} "
        f"student_arch={model_kwargs_from_config(asdict(cfg))}",
        flush=True,
    )
    states, masks, policies, values, metadata = generate_distill_examples(
        teacher, teacher_cfg, args, device=device
    )
    print("distill_dataset " + json.dumps(metadata, sort_keys=True), flush=True)
    train_student(student, args, cfg, (states, masks, policies, values), device=device)


if __name__ == "__main__":
    main()
