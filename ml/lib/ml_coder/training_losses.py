"""Reusable loss functions for ML-coder training scripts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AsymmetricLossConfig:
    gamma_neg: float = 4.0
    gamma_pos: float = 1.0
    clip: float = 0.05
    eps: float = 1e-8


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for multi-label classification.

    Returns an element-wise loss matrix of shape [batch, labels] when called.
    """

    def __init__(
        self,
        *,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        xs_pos = xs_pos.clamp(min=self.eps, max=1.0)
        xs_neg = xs_neg.clamp(min=self.eps, max=1.0)

        # Base CE terms (note: logs are negative).
        loss_pos = targets * torch.log(xs_pos)
        if self.pos_weight is not None:
            loss_pos = loss_pos * self.pos_weight
        loss_neg = (1.0 - targets) * torch.log(xs_neg)

        loss = loss_pos + loss_neg

        # Asymmetric focusing.
        gamma_pos = self.gamma_pos
        gamma_neg = self.gamma_neg
        if gamma_pos > 0 or gamma_neg > 0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
            weight = torch.pow(1.0 - pt, gamma)
            loss = loss * weight

        return -loss

