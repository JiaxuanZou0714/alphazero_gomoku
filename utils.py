from __future__ import annotations

from pathlib import Path

import torch

from .model import PolicyValueNet, build_model_from_config


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this Python environment cannot see it")
    return device


def load_torch(path: Path | str, *, map_location: str | torch.device = "cpu") -> object:
    """``torch.load`` with a ``weights_only`` fallback for older PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(checkpoint: Path | str, device: str) -> tuple[PolicyValueNet, dict]:
    path = Path(checkpoint)
    payload = load_torch(path, map_location=device)
    cfg = payload.get("config", {})
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in unwrap_model(model).state_dict().items()
    }


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"
