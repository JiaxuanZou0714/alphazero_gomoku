from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def autocast_for(device: str | torch.device, amp_dtype: str) -> object:
    """Return the autocast context for ``amp_dtype`` on ``device``.

    Single source of truth shared by training (``train.py``) and search
    (``mcts.py``). CUDA only; ``"none"`` (or a non-CUDA device) disables AMP.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    if amp_dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp_dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"unknown amp_dtype: {amp_dtype}")


def tensor_from_array(
    value: Any,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Create a tensor without requiring PyTorch's NumPy bridge.

    Some lab images combine older PyTorch wheels with NumPy 2.x. In that setup
    torch.from_numpy and torch.as_tensor(numpy_array) can fail even though NumPy
    itself works. The fallback keeps training runnable without changing the
    server's global Python environment.
    """

    try:
        return torch.as_tensor(value, dtype=dtype, device=device)
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc) and "Could not infer dtype" not in str(exc):
            raise
        if not hasattr(value, "tolist"):
            raise
        return torch.tensor(value.tolist(), dtype=dtype, device=device)
