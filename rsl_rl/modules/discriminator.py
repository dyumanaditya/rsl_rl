# Copyright (c) 2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from amp-rsl-rl (ami-iit/amp-rsl-rl) Discriminator.
#
# Key difference from the original: this version takes a SINGLE disc_obs
# tensor as input (not a concatenated state+next_state pair).  This matches
# the G1ImitationEnv disc_obs format where:
#   - AMP mode: disc_obs is already a multi-frame history (n_steps * feat)
#   - ADD mode: disc_obs is (demo - agent) difference feature (single frame)

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F

from rsl_rl.modules.normalizer import EmpiricalNormalization


class Discriminator(nn.Module):
    """AMP/ADD discriminator for single-obs input.

    Architecture mirrors amp-rsl-rl's Discriminator (trunk + linear head,
    BCEWithLogitsLoss, optional R1 gradient penalty, optional empirical
    normalisation) but the ``forward()`` method takes a single ``disc_obs``
    tensor rather than a concatenated (state, next_state) pair.

    Args:
        input_dim: Dimension of disc_obs (disc_obs_size from G1ImitationEnv).
        hidden_layer_sizes: Hidden widths of the MLP trunk.
        reward_scale: Multiplicative scale applied to predicted rewards.
        reward_clamp_epsilon: Unused – kept for API compatibility.
        device: Torch device.
        empirical_normalization: If True, maintain an EmpiricalNormalization
            layer that is updated via ``update_normalization()``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list[int],
        reward_scale: float,
        reward_clamp_epsilon: float = 1.0e-4,
        device: str | torch.device = "cpu",
        empirical_normalization: bool = False,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.reward_scale = reward_scale

        layers: list[nn.Module] = []
        curr_in = input_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in, h))
            layers.append(nn.ReLU())
            curr_in = h

        self.trunk = nn.Sequential(*layers)
        self.linear = nn.Linear(curr_in, 1)

        self.empirical_normalization = empirical_normalization
        if empirical_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(shape=[input_dim])
        else:
            self.obs_normalizer = nn.Identity()

        self.loss_fun = nn.BCEWithLogitsLoss()

        self.to(self.device)
        self.train()

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        return self.linear(self.trunk(x))

    @torch.no_grad()
    def predict_reward(self, disc_obs: torch.Tensor) -> torch.Tensor:
        """Compute adversarial reward from a batch of disc_obs.

        Uses softplus(logit) == -log(1 - sigmoid(logit)) which matches both
        the amp-rsl-rl reward formula and the existing AMPDiscriminator.

        Returns:
            Tensor of shape (B,).
        """
        logit = self.forward(disc_obs)
        return (self.reward_scale * F.softplus(logit)).squeeze(-1)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        policy_d: torch.Tensor,
        expert_d: torch.Tensor,
        expert_obs: torch.Tensor,
        lambda_: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (amp_loss, grad_pen_loss) for combining with PPO loss.

        Args:
            policy_d: Discriminator logits for policy (agent) samples.
            expert_d: Discriminator logits for expert (demo) samples.
            expert_obs: Raw (pre-normalised) expert obs – used for the R1
                gradient penalty.
            lambda_: R1 penalty coefficient.
        """
        expert_loss = self.loss_fun(expert_d, torch.ones_like(expert_d))
        policy_loss = self.loss_fun(policy_d, torch.zeros_like(policy_d))
        amp_loss = 0.5 * (expert_loss + policy_loss)
        grad_pen = self.compute_grad_pen(expert_obs, lambda_)
        return amp_loss, grad_pen

    def compute_grad_pen(self, expert_obs: torch.Tensor, lambda_: float = 10.0) -> torch.Tensor:
        """R1 gradient penalty evaluated on the expert (real) data.

        Matches the BCEWithLogits branch of amp-rsl-rl's compute_grad_pen.
        """
        data = self.obs_normalizer(expert_obs.detach()).requires_grad_(True)
        scores = self.linear(self.trunk(data))
        grad = autograd.grad(
            outputs=scores.sum(),
            inputs=data,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return 0.5 * lambda_ * grad.pow(2).sum(dim=1).mean()

    def update_normalization(self, *batches: torch.Tensor) -> None:
        """Update empirical normaliser statistics (no-op when disabled)."""
        if not self.empirical_normalization:
            return
        with torch.no_grad():
            for batch in batches:
                self.obs_normalizer.update(batch)
