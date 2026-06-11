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


def load_model(checkpoint: Path | str, device: str) -> tuple[PolicyValueNet, dict]:
    path = Path(checkpoint)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    cfg = payload.get("config", {})
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg
