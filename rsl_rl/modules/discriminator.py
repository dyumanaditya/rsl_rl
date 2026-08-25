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

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F

from rsl_rl.modules.normalizer import EmpiricalNormalization


class DiffNormalization(nn.Module):
    """ADD-style normalizer: scale inputs by running mean absolute value.

    Zeros (the expert class in ADD) normalize to exactly zero — unlike
    EmpiricalNormalization which shifts them by the running mean.
    """

    def __init__(self, shape: list[int], min_abs: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("mean_abs", torch.ones(shape))
        self.register_buffer("count", torch.tensor(0.0))
        self.min_abs = min_abs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.clamp_min(self.mean_abs, self.min_abs)

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        x_flat = x.reshape(-1, self.mean_abs.shape[-1])
        n = float(x_flat.shape[0])
        if n == 0:
            return
        total = self.count + n
        self.mean_abs += (x_flat.abs().mean(0) - self.mean_abs) * n / total.clamp(min=1)
        self.count = total


class Discriminator(nn.Module):
    """AMP/ADD discriminator for single-obs input.

    Architecture mirrors amp-rsl-rl's Discriminator (trunk + linear head,
    BCEWithLogitsLoss, optional R1 gradient penalty, optional empirical
    normalisation) but the ``forward()`` method takes a single ``disc_obs``
    tensor rather than a concatenated (state, next_state) pair.

    External root tracking (``imitation.use_aux_root_tracking``, ADD only) mirrors
    the FoRL-SHAC ``ADDDiscriminator``: the global root pos/rot dims are REMOVED
    from the differential the classifier judges, and a separate smooth POSITIVE
    root pose-tracking reward is added on top of the adversarial style reward.
    ``input_dim`` then reports the REDUCED width, which is what sizes the trunk,
    the normalizer, the AMP_PPO replay buffer and the ADD zero-positive class.

    Args:
        input_dim: Dimension of disc_obs (disc_obs_size from G1ImitationEnv).
        hidden_layer_sizes: Hidden widths of the MLP trunk.
        reward_scale: Multiplicative scale applied to predicted rewards.
        reward_clamp_epsilon: Unused – kept for API compatibility.
        device: Torch device.
        empirical_normalization: If True, maintain an EmpiricalNormalization
            layer that is updated via ``update_normalization()``.
        feature_groups: ``{name: (start, end)}`` slices of the full differential
            (``base_env.disc_feature_groups``). Required for external root tracking
            so ``root_pos`` / ``root_rot`` can be located.
        add_cfg: Resolved ADD config (``imitation.wadd.resolve_add_cfg``) carrying
            ``use_aux_root_tracking`` and the ``aux_root_*`` weights / kernel.
            ``None`` disables external root tracking (classic behaviour).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list[int],
        reward_scale: float,
        reward_clamp_epsilon: float = 1.0e-4,
        device: str | torch.device = "cpu",
        empirical_normalization: bool = False,
        diff_normalization: bool = False,
        feature_groups: Optional[Dict[str, Tuple[int, int]]] = None,
        add_cfg=None,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.reward_scale = reward_scale

        # ------------------------------------------------------------------
        # External root tracking (anchor-body style; ADD only)
        # ------------------------------------------------------------------
        # The global root pose lives in the root_pos/root_rot slots of the
        # differential for the AUX reward, but is dropped from what the classifier
        # sees. Precompute the keep indices BEFORE building the trunk so the net is
        # sized to the reduced differential — exactly as ADDDiscriminator does.
        self.add = add_cfg
        self.full_dim = input_dim
        self.feature_groups = dict(feature_groups) if feature_groups else {}
        self.aux_root = bool(
            add_cfg is not None and getattr(add_cfg, "use_aux_root_tracking", False)
        )
        self._aux_pos_slice = self.feature_groups.get("root_pos") if self.aux_root else None
        self._aux_ori_slice = self.feature_groups.get("root_rot") if self.aux_root else None
        self._keep_idx: Optional[torch.Tensor] = None
        self.reduced_feature_groups = self.feature_groups
        self._aux_reward_fn = None
        self._root_term_fn = None
        if self.aux_root and (self._aux_pos_slice or self._aux_ori_slice):
            # External root tracking shares its reward kernel and group-reindex
            # helper with the FoRL-SHAC discriminator (imitation/wadd.py +
            # imitation/discriminator.py) so the ADD reward is numerically
            # IDENTICAL under both learning algorithms. Imported lazily and bound
            # to the instance: rsl_rl stays importable without the app package on
            # sys.path, and the per-step reward path pays no import lookup.
            from imitation.discriminator import _reindex_feature_groups
            from imitation.wadd import aux_root_tracking_reward, root_tracking_term

            self._aux_reward_fn = aux_root_tracking_reward
            self._root_term_fn = root_tracking_term

            drop = set()
            for sl in (self._aux_pos_slice, self._aux_ori_slice):
                if sl is not None:
                    drop.update(range(sl[0], sl[1]))
            keep = [i for i in range(input_dim) if i not in drop]
            self._keep_idx = torch.tensor(keep, device=self.device, dtype=torch.long)
            input_dim = len(keep)
            self.reduced_feature_groups = _reindex_feature_groups(
                self.feature_groups, drop, self.full_dim
            )
        # Per-env breakdown of the LAST aux reward, stashed by _aux_reward so the
        # runner can log the style-vs-aux split (the aux is otherwise folded into
        # the disc reward and invisible). Detached; no host sync in the hot loop.
        self.last_aux_pos: Optional[torch.Tensor] = None
        self.last_aux_ori: Optional[torch.Tensor] = None
        self.last_aux_total: Optional[torch.Tensor] = None

        # Width the classifier / replay / zero-positive class all use.
        self.input_dim = input_dim

        layers: list[nn.Module] = []
        curr_in = input_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in, h))
            layers.append(nn.ReLU())
            curr_in = h

        self.trunk = nn.Sequential(*layers)
        self.linear = nn.Linear(curr_in, 1)

        if diff_normalization:
            self.obs_normalizer: nn.Module = DiffNormalization(shape=[input_dim])
        elif empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[input_dim])
        else:
            self.obs_normalizer = nn.Identity()

        self.loss_fun = nn.BCEWithLogitsLoss()

        self.to(self.device)
        if self._keep_idx is not None:
            self._keep_idx = self._keep_idx.to(self.device)
        self.train()

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is already REDUCED (see ``reduce``) — the replay buffer and the
        ADD zero-positive class both store reduced differentials, so the training
        path never needs to reduce again. Only ``predict_reward`` takes the full
        differential (it needs the raw root slots for the aux reward)."""
        x = self.obs_normalizer(x)
        return self.linear(self.trunk(x))

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Drop the global root pos/rot dims from a full disc-obs / differential so
        only the local pose/limb features remain. No-op unless external root
        tracking is on. Mirrors ``ADDDiscriminator._reduce``."""
        return x if self._keep_idx is None else x.index_select(-1, self._keep_idx)

    def _aux_reward(self, raw_diff: torch.Tensor) -> Optional[torch.Tensor]:
        """Separate smooth global-root tracking reward (position + orientation),
        read from the un-reduced ``raw_diff = demo - agent`` whose root slots keep
        the GLOBAL root pose. ``None`` when external root tracking is off.

        Also stashes the per-env position / orientation / total breakdown on
        ``self.last_aux_*`` so the runner can log the style-vs-aux split.
        Byte-for-byte the same computation as ``ADDDiscriminator._aux_reward``
        (js branch: fixed ``aux_root_clip`` bound)."""
        if not self.aux_root or self._aux_reward_fn is None:
            self.last_aux_pos = self.last_aux_ori = self.last_aux_total = None
            return None
        clip = getattr(self.add, "aux_root_clip", 0.0)
        bound = float(clip) if clip and clip > 0 else None
        aux = self._aux_reward_fn(
            raw_diff, self._aux_pos_slice, self._aux_ori_slice,
            self.add.aux_root_weight, self.add.aux_root_sigma,
            self.add.aux_root_ori_weight, self.add.aux_root_ori_sigma,
            kind=self.add.aux_root_reward_kind,
            bound=bound,
        )
        self._stash_aux_breakdown(raw_diff)
        return aux

    def _stash_aux_breakdown(self, raw_diff: torch.Tensor) -> None:
        """Record the per-env position/orientation/total aux reward for logging
        (no host sync — the runner reduces once per iteration). Detached like
        ``ADDDiscriminator._stash_aux_breakdown`` so the stashed tensors never
        pin a graph when ``_aux_reward`` is called outside ``predict_reward``."""
        with torch.no_grad():
            pos = ori = None
            if self.add.aux_root_weight != 0.0 and self._aux_pos_slice is not None:
                sl = self._aux_pos_slice
                pos = self.add.aux_root_weight * self._root_term_fn(
                    raw_diff[..., sl[0]:sl[1]], self.add.aux_root_sigma,
                    self.add.aux_root_reward_kind)
            if self.add.aux_root_ori_weight != 0.0 and self._aux_ori_slice is not None:
                sl = self._aux_ori_slice
                ori = self.add.aux_root_ori_weight * self._root_term_fn(
                    raw_diff[..., sl[0]:sl[1]], self.add.aux_root_ori_sigma,
                    self.add.aux_root_reward_kind)
            self.last_aux_pos = pos
            self.last_aux_ori = ori
            if pos is None and ori is None:
                self.last_aux_total = None
            elif pos is None:
                self.last_aux_total = ori
            elif ori is None:
                self.last_aux_total = pos
            else:
                self.last_aux_total = pos + ori

    @torch.no_grad()
    def predict_reward(self, disc_obs: torch.Tensor) -> torch.Tensor:
        """Compute the policy-facing reward from a batch of disc_obs.

        ``disc_obs`` is the FULL-width input: the agent disc obs for AMP, or the
        full ``demo - agent`` differential for ADD. Uses softplus(logit) ==
        -log(1 - sigmoid(logit)), matching both the amp-rsl-rl reward formula and
        the existing AMPDiscriminator.

        With external root tracking on, the returned reward is
        ``style(reduced diff) + aux_root(raw diff)`` — the same sum
        ``ADDDiscriminator._js_reward`` returns, so the ``disc_reward_weight``
        applied by the caller scales both terms exactly as it does under SHAC.

        Returns:
            Tensor of shape (B,).
        """
        logit = self.forward(self.reduce(disc_obs))
        reward = (self.reward_scale * F.softplus(logit)).squeeze(-1)
        aux = self._aux_reward(disc_obs)
        if aux is not None:
            reward = reward + aux
        return reward

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        policy_d: torch.Tensor,
        expert_d: torch.Tensor,
        expert_obs: torch.Tensor,
        lambda_: float = 10.0,
        policy_obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (amp_loss, grad_pen_loss) for combining with PPO loss.

        Args:
            policy_d: Discriminator logits for policy (agent) samples.
            expert_d: Discriminator logits for expert (demo) samples.
            expert_obs: Raw (pre-normalised) expert obs – R1 gradient penalty anchor.
            lambda_: R1 penalty coefficient.
            policy_obs: If provided, also compute gradient penalty at policy samples
                (two-sided GP matching MimicKit ADD; pass for ADD mode, omit for AMP).
        """
        expert_loss = self.loss_fun(expert_d, torch.ones_like(expert_d))
        policy_loss = self.loss_fun(policy_d, torch.zeros_like(policy_d))
        amp_loss = 0.5 * (expert_loss + policy_loss)
        grad_pen = self.compute_grad_pen(expert_obs, lambda_)
        if policy_obs is not None:
            grad_pen = grad_pen + self.compute_grad_pen(policy_obs, lambda_)
        return amp_loss, grad_pen

    def compute_grad_pen(self, expert_obs: torch.Tensor, lambda_: float = 10.0) -> torch.Tensor:
        """R1 gradient penalty evaluated on the expert (real) data.

        Matches the BCEWithLogits branch of amp-rsl-rl's compute_grad_pen.

        Prefer ``compute_loss_fused`` in the training loop: this entry point runs
        its OWN forward through the trunk, so calling it once per class (the ADD
        two-sided penalty) costs two extra discriminator forwards on top of the
        one already used for the logits.
        """
        if lambda_ <= 0.0:
            # The penalty is scaled to zero — skip the (expensive) double-backward
            # graph entirely instead of building it and multiplying by 0.
            return expert_obs.new_zeros(())
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

    def compute_loss_fused(
        self,
        combined_obs: torch.Tensor,
        num_policy: int,
        lambda_: float = 10.0,
        two_sided: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-forward version of ``forward`` + ``compute_loss``.

        The unfused path runs the trunk THREE times per minibatch: once for the
        logits, then once inside each ``compute_grad_pen`` call (expert, and for
        ADD also policy). The R1 penalty only needs d(score)/d(normalised input),
        which is available from the SAME graph as the logits, so a single forward
        plus a single ``autograd.grad`` reproduces every term exactly.

        Args:
            combined_obs: ``cat([policy_obs, expert_obs], dim=0)`` — the reduced
                disc observations, already detached (they come from storage).
            num_policy: number of leading rows belonging to the policy class.
            lambda_: R1 penalty coefficient. ``<= 0`` skips the penalty graph.
            two_sided: also penalise at the policy samples (MimicKit ADD).

        Returns:
            ``(amp_loss, grad_pen, policy_d, expert_d)`` — numerically identical
            to ``forward`` + ``compute_loss`` on the same inputs.
        """
        need_gp = lambda_ > 0.0

        # EmpiricalNormalization.forward UPDATES its running statistics, so the
        # unfused path (which normalises once per forward, i.e. 3x per minibatch)
        # advances those stats differently than a single fused call. Fusing would
        # then silently change AMP's normalisation schedule, so fall back to the
        # legacy path there. ADD's DiffNorm and the nn.Identity default are pure
        # functions of their input (DiffNorm is advanced explicitly via
        # update_normalization), so fusing is exact for them.
        if self.training and isinstance(self.obs_normalizer, EmpiricalNormalization):
            scores = self.linear(self.trunk(self.obs_normalizer(combined_obs)))
            policy_d, expert_d = scores[:num_policy], scores[num_policy:]
            amp_loss, grad_pen = self.compute_loss(
                policy_d=policy_d,
                expert_d=expert_d,
                expert_obs=combined_obs[num_policy:],
                lambda_=lambda_,
                policy_obs=combined_obs[:num_policy] if two_sided else None,
            )
            return amp_loss, grad_pen, policy_d, expert_d

        data = self.obs_normalizer(combined_obs.detach())
        if need_gp:
            data = data.requires_grad_(True)
        scores = self.linear(self.trunk(data))
        policy_d, expert_d = scores[:num_policy], scores[num_policy:]

        expert_loss = self.loss_fun(expert_d, torch.ones_like(expert_d))
        policy_loss = self.loss_fun(policy_d, torch.zeros_like(policy_d))
        amp_loss = 0.5 * (expert_loss + policy_loss)

        if not need_gp:
            return amp_loss, scores.new_zeros(()), policy_d, expert_d

        grad = autograd.grad(
            outputs=scores.sum(),
            inputs=data,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        sq = grad.pow(2).sum(dim=1)
        # Same per-class means the unfused two calls produce.
        grad_pen = 0.5 * lambda_ * sq[num_policy:].mean()
        if two_sided:
            grad_pen = grad_pen + 0.5 * lambda_ * sq[:num_policy].mean()
        return amp_loss, grad_pen, policy_d, expert_d

    def update_normalization(self, *batches: torch.Tensor) -> None:
        """Update normaliser statistics (no-op when using nn.Identity)."""
        if isinstance(self.obs_normalizer, nn.Identity):
            return
        with torch.no_grad():
            for batch in batches:
                self.obs_normalizer.update(batch)
