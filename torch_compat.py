from __future__ import annotations

from typing import Any

import torch


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
