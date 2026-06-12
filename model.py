from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        use_se: bool = False,
        se_ratio: int = 16,
        use_global_pool: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        hidden = max(1, channels // se_ratio)
        self.se = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1),
                nn.Sigmoid(),
            )
            if use_se
            else None
        )
        # KataGo global pooling: broadcast global board context into every cell.
        # avg+max pool -> Linear -> additive per-channel bias. KataGo adds the
        # pooled features as biases rather than multiplying by a sigmoid gate,
        # which would attenuate the signal across many stacked blocks.
        self.gp_fc = nn.Linear(channels * 2, channels) if use_global_pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.se is not None:
            x = x * self.se(x)
        if self.gp_fc is not None:
            g_avg = x.mean(dim=(2, 3))
            g_max = x.amax(dim=(2, 3))
            bias = self.gp_fc(torch.cat([g_avg, g_max], dim=1))
            x = x + bias.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(x + residual)
        return x


class PolicyValueNet(nn.Module):
    """Residual policy/value network for 10x10 Gomoku.

    No hand-coded patterns. forward() always returns (policy_logits, soft_logits, value)
    where soft_logits is None when use_soft_policy=False.
    """

    def __init__(
        self,
        board_size: int = 10,
        input_channels: int = 2,
        channels: int = 64,
        residual_blocks: int = 4,
        policy_channels: int = 2,
        value_channels: int = 1,
        value_hidden: int = 128,
        use_se: bool = False,
        se_ratio: int = 16,
        use_global_pool: bool = False,
        use_soft_policy: bool = False,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.use_soft_policy = use_soft_policy

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual_tower = nn.Sequential(
            *[
                ResidualBlock(channels, use_se=use_se, se_ratio=se_ratio, use_global_pool=use_global_pool)
                for _ in range(residual_blocks)
            ]
        )

        def _policy_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, policy_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(policy_channels),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(policy_channels * board_size * board_size, self.action_size),
            )

        self.policy_head = _policy_head()
        self.soft_policy_head: nn.Module | None = _policy_head() if use_soft_policy else None

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(value_channels * board_size * board_size, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
        )

        nn.init.normal_(self.policy_head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.policy_head[-1].bias)
        nn.init.normal_(self.value_head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.value_head[-1].bias)
        if self.soft_policy_head is not None:
            nn.init.normal_(self.soft_policy_head[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.soft_policy_head[-1].bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        x = self.stem(x)
        x = self.residual_tower(x)
        policy_logits = self.policy_head(x)
        soft_logits = self.soft_policy_head(x) if self.soft_policy_head is not None else None
        value = torch.tanh(self.value_head(self.value_conv(x)).squeeze(-1).float()).clamp(-1.0, 1.0)
        return policy_logits, soft_logits, value


def model_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "board_size": int(cfg.get("board_size", 10)),
        "channels": int(cfg.get("channels", 64)),
        "residual_blocks": int(cfg.get("residual_blocks", 4)),
        "policy_channels": int(cfg.get("policy_channels", 2)),
        "value_channels": int(cfg.get("value_channels", 1)),
        "value_hidden": int(cfg.get("value_hidden", 128)),
        "use_se": bool(cfg.get("use_se", False)),
        "se_ratio": int(cfg.get("se_ratio", 16)),
        "use_global_pool": bool(cfg.get("use_global_pool", False)),
        "use_soft_policy": bool(cfg.get("use_soft_policy", False)),
    }


def build_model_from_config(cfg: dict[str, Any]) -> PolicyValueNet:
    return PolicyValueNet(**model_kwargs_from_config(cfg))
