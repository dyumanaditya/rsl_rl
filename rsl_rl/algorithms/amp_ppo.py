# Copyright (c) 2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from amp-rsl-rl (ami-iit/amp-rsl-rl) AMP_PPO.
#
# Key differences from the original:
#  - Works with flat tensors (not TensorDicts).
#  - Single disc_obs per entry in the replay buffer (not state/next_state pairs).
#  - ``demo_fn``: callable(n) -> Tensor replaces AMPLoader.
#    AMP mode: returns sampled demo disc_obs from the motion library.
#    ADD mode: returns zeros (the positive / "perfect match" class).
#  - Supports both 'amp' and 'add' imitation modes.

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer


class AMP_PPO:
    """PPO with an AMP / ADD discriminator trained on MimicKit's schedule.

    The discriminator is trained in its OWN optimisation loop after the PPO
    epochs, with its own optimizer, on MimicKit's step count::

        batch  = ceil(disc_batch_size * num_envs)          # disc_batch_size is per-env
        steps  = ceil(rollout_samples / batch) * disc_epochs

    (MimicKit ``amp_agent._update_model``/``_update_disc``; the same formula the
    FoRL-SHAC path uses in ``imitation.discriminator.Discriminator.update``.)

    This replaces the inherited amp-rsl-rl behaviour of folding the
    discriminator loss into the PPO minibatch loss and taking a single backward
    pass. That coupling pinned the discriminator to ``num_learning_epochs *
    num_mini_batches`` steps at the PPO minibatch size and left
    ``imitation.disc_epochs`` / ``disc_batch_size`` / ``disc_replay_samples``
    dead, so PPO and FoRL-SHAC trained the same discriminator on different
    schedules.

    Parameters
    ----------
    actor_critic:
        Policy network.
    discriminator:
        ``Discriminator`` instance (from ``rsl_rl.modules.discriminator``).
    demo_fn:
        Callable ``(n: int) -> Tensor[n, disc_obs_size]`` that returns
        expert / demo observations.
        - AMP mode: samples from the motion library.
        - ADD mode: returns zeros (perfect-match target class).
    mode:
        ``'amp'`` or ``'add'``.
    disc_lr:
        Learning rate used for discriminator parameters.  Defaults to
        ``learning_rate`` when ``None``.
    amp_replay_buffer_size:
        Capacity of the disc_obs replay buffer.
    grad_penalty_coeff:
        R1 gradient-penalty coefficient (λ).
    disc_epochs:
        Passes over the rollout's disc observations per update (MimicKit
        ``disc_epochs``).
    disc_batch_size:
        Discriminator minibatch size PER ENV; scaled by ``num_envs`` at update
        time, matching MimicKit's ``ceil(disc_batch_size * num_envs)``.
    disc_replay_samples:
        Number of replay observations mixed into each discriminator batch
        alongside the current rollout (MimicKit ``disc_replay_samples``).
    """

    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic: ActorCritic,
        discriminator: Discriminator,
        demo_fn: Callable[[int], torch.Tensor],
        mode: str = "amp",
        # PPO hyper-parameters
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        action_bound_weight: float = 0.0,
        learning_rate: float = 1e-3,
        critic_learning_rate: Optional[float] = None,
        actor_optimizer_class_name: str = "Adam",
        critic_optimizer_class_name: Optional[str] = None,
        disc_optimizer_class_name: Optional[str] = None,
        optimizer_momentum: float = 0.9,
        disc_weight_decay: float = 1.0e-3,
        disc_head_weight_decay: float = 1.0e-1,
        max_grad_norm: float = 1.0,
        critic_max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = True,
        value_clip_scale: float = 0.0,
        norm_adv_clip: float = 0.0,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        desired_kl_per_action_dim: bool = False,
        min_learning_rate: float = 1e-5,
        max_learning_rate: float = 1e-2,
        # Discriminator hyper-parameters
        disc_lr: Optional[float] = None,
        amp_replay_buffer_size: int = 100_000,
        grad_penalty_coeff: float = 10.0,
        disc_epochs: int = 2,
        disc_batch_size: int = 2,
        disc_replay_samples: int = 1000,
        disc_logit_reg: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_learning_rate = float(min_learning_rate)
        self.max_learning_rate = float(max_learning_rate)
        self.norm_adv_clip = float(norm_adv_clip)
        self.value_clip_scale = float(value_clip_scale)
        self.mode = mode

        # The adaptive-KL target is a SUM over action dimensions, so the tuned
        # rsl_rl default (0.01, calibrated on 12-DoF quadrupeds) is a ~2.4x
        # tighter trust region on a 29-DoF humanoid. With
        # ``desired_kl_per_action_dim`` the config value is read PER DIMENSION and
        # scaled by num_actions here, so the same number means the same policy
        # displacement regardless of the robot.
        self._num_actions = int(actor_critic.std.numel()) if hasattr(actor_critic, "std") else 1
        self.desired_kl = (
            float(desired_kl) * self._num_actions
            if desired_kl is not None and desired_kl_per_action_dim
            else desired_kl
        )

        # Discriminator
        self.discriminator = discriminator.to(self.device)
        disc_obs_size = discriminator.input_dim
        self.amp_storage = ReplayBuffer(disc_obs_size, amp_replay_buffer_size, device)
        self.demo_fn = demo_fn
        self.grad_penalty_coeff = grad_penalty_coeff
        self.disc_epochs = int(disc_epochs)
        self.disc_batch_size = int(disc_batch_size)   # per-env, scaled at update time
        self.disc_replay_samples = int(disc_replay_samples)
        # MimicKit `disc_logit_reg` (add_agent._compute_disc_loss) — an explicit
        # L2 on the classifier HEAD weights, which FoRL-SHAC also applies
        # (imitation/discriminator.py _train_step). This path had neither, only
        # amp-rsl-rl's optimizer weight decay, so the same `imitation.disc_logit_reg`
        # config value shaped the discriminator under SHAC and was dead under PPO.
        self.disc_logit_reg = float(disc_logit_reg)
        # Current iteration's disc observations, appended per rollout step by
        # process_disc_step and consumed (then pushed to the replay) by
        # _update_disc. Kept separate from amp_storage so the update can mix the
        # FRESH rollout with replay exactly as MimicKit / FoRL-SHAC do.
        self._rollout_disc_obs: list[torch.Tensor] = []

        # Actor-critic
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage: Optional[RolloutStorage] = None

        # SEPARATE optimizers, as MimicKit has (its disc has its own
        # ``_disc_optimizer``). Previously both lived in one Adam because the two
        # losses shared a backward pass; now that the discriminator has its own
        # update loop, a shared optimizer would step the policy parameters on the
        # disc iterations too (Adam moves a parameter whenever its group is
        # stepped and it still holds momentum), which is exactly what we must not
        # do. Discriminator trunk and head keep the amp-rsl-rl weight decays.
        #
        # The ACTOR and the CRITIC also get their own optimizers, matching
        # MimicKit (``_actor_optimizer`` / ``_critic_optimizer``) and FoRL-SHAC
        # (``actor_optimizer`` / ``critic_optimizer``). Sharing one was actively
        # harmful here: ``clip_grad_norm_`` was applied to the CONCATENATED
        # actor+critic gradient, and the ADD reward is dense and unnormalised
        # (returns ~60), so the value loss — and with it the critic gradient —
        # dominates the joint norm. Measured on this env: |g_critic| ~ 11 vs
        # |g_actor| ~ 1.8, so a max_grad_norm of 1.0 rescaled BOTH by ~0.16, and
        # the factor swung between 0.03 and 0.31 from iteration to iteration
        # purely as a function of how well the critic happened to fit. The actor's
        # step size was therefore set by the critic's loss magnitude. Clipping the
        # two groups separately decouples them, and a separate critic LR keeps the
        # adaptive-KL schedule (a POLICY trust region) off the value function.
        _disc_lr = disc_lr if disc_lr is not None else learning_rate
        _critic_lr = (
            float(critic_learning_rate) if critic_learning_rate is not None else learning_rate
        )
        self.critic_learning_rate = _critic_lr
        # Split by NAME so ``std`` (an actor-side Parameter living directly on the
        # module) lands with the actor, and ActorCriticRecurrent's ``memory_c``
        # encoder lands with the critic it feeds.
        _critic_prefixes = ("critic.", "memory_c.")
        self.actor_params = [
            p for n, p in self.actor_critic.named_parameters()
            if not n.startswith(_critic_prefixes)
        ]
        self.critic_params = [
            p for n, p in self.actor_critic.named_parameters()
            if n.startswith(_critic_prefixes)
        ]
        assert len(self.actor_params) + len(self.critic_params) == len(
            list(self.actor_critic.parameters())
        ), "actor/critic parameter split did not cover every parameter"
        def make_optimizer(name, param_groups, lr):
            name = str(name).lower()
            if name == "adam":
                return optim.Adam(param_groups, lr=lr)
            if name == "adamw":
                return optim.AdamW(param_groups, lr=lr)
            if name == "sgd":
                return optim.SGD(param_groups, lr=lr, momentum=float(optimizer_momentum))
            raise ValueError(
                f"Unsupported AMP_PPO optimizer {name!r}; expected Adam, AdamW, or SGD."
            )

        critic_optimizer_class_name = (
            critic_optimizer_class_name or actor_optimizer_class_name
        )
        disc_optimizer_class_name = disc_optimizer_class_name or actor_optimizer_class_name
        self.optimizer = make_optimizer(
            actor_optimizer_class_name,
            [{"params": self.actor_params, "lr": learning_rate}],
            learning_rate,
        )
        self.critic_optimizer = make_optimizer(
            critic_optimizer_class_name,
            [{"params": self.critic_params, "lr": _critic_lr}],
            _critic_lr,
        )
        self.disc_optimizer = make_optimizer(
            disc_optimizer_class_name,
            [
                {
                    "params": self.discriminator.trunk.parameters(),
                    "lr": _disc_lr,
                    "weight_decay": float(disc_weight_decay),
                },
                {
                    "params": self.discriminator.linear.parameters(),
                    "lr": _disc_lr,
                    "weight_decay": float(disc_head_weight_decay),
                },
            ],
            _disc_lr,
        )
        # The adaptive-KL schedule may only move the ACTOR LR; the critic and the
        # discriminator keep their fixed rates. That is now structural (separate
        # optimizers) rather than a param-group carve-out, but the attribute is
        # kept so the schedule code below reads the same.
        self._policy_param_group = self.optimizer.param_groups[0]

        # Per-update diagnostics the runner mirrors to tensorboard/wandb. Without
        # them a starved actor is invisible: the reported losses look healthy
        # while the actor is taking near-zero steps.
        self.last_actor_grad_norm = 0.0
        self.last_critic_grad_norm = 0.0
        self.last_clip_frac = 0.0

        self.transition = RolloutStorage.Transition()

        # PPO hyper-parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.action_bound_weight = float(action_bound_weight)
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        # FoRL-SHAC gives the critic 10x the actor's clip (forl_cfg.py:39) and
        # MimicKit clips neither. Now that the two groups are clipped separately,
        # a shared 1.0 would be a much TIGHTER bound on the critic than it ever
        # was under the joint clip, so follow SHAC's ratio by default.
        self.critic_max_grad_norm = (
            float(critic_max_grad_norm)
            if critic_max_grad_norm is not None
            else 10.0 * max_grad_norm
        )
        self.use_clipped_value_loss = use_clipped_value_loss

    # ------------------------------------------------------------------
    # Storage initialisation
    # ------------------------------------------------------------------

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ) -> None:
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def test_mode(self) -> None:
        self.actor_critic.eval()

    def train_mode(self) -> None:
        self.actor_critic.train()

    # ------------------------------------------------------------------
    # Rollout interface
    # ------------------------------------------------------------------

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = (
            self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        )
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
        timeout_values: Optional[torch.Tensor] = None,
    ) -> None:
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            # A timeout must bootstrap V(s_{t+1}), i.e. the final state captured
            # before the environment resets. The legacy fallback below uses
            # V(s_t) only for callers that have not supplied terminal values.
            # AMPOnPolicyRunner supplies the correct values for feed-forward
            # policies through PPOEnv's terminal_critic_obs field.
            bootstrap_values = (
                timeout_values
                if timeout_values is not None
                else self.transition.values.squeeze(-1)
            )
            self.transition.rewards += (
                self.gamma * bootstrap_values * infos["time_outs"].to(self.device)
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def process_disc_step(
        self,
        disc_obs: torch.Tensor,
        demo_disc_obs: Optional[torch.Tensor] = None,
    ) -> None:
        """Insert a disc observation into the replay buffer.

        AMP mode: inserts ``disc_obs`` (agent state representation).
        ADD mode: inserts the *difference* ``demo_disc_obs - disc_obs`` so
            that the replay buffer always stores the same quantity the
            discriminator is trained on (actual diff = negative class).

        With external root tracking the stored differential is REDUCED (global
        root pos/rot dropped) so the classifier gets no root signal at all —
        root tracking is the aux reward's job. ``discriminator.reduce`` is a
        no-op otherwise, and the buffer is sized to ``discriminator.input_dim``,
        which already reports the reduced width. Mirrors the SHAC
        ``ADDDiscriminator._update_js`` reduce-then-push order.
        """
        if self.mode == "add" and demo_disc_obs is not None:
            entry = self.discriminator.reduce((demo_disc_obs - disc_obs).detach())
        else:
            entry = self.discriminator.reduce(disc_obs.detach())
        # Held for this iteration's disc update, which pushes the whole rollout
        # into ``amp_storage`` in one go (mirrors ADDDiscriminator._update_js:
        # record -> replay.push -> sample). Inserting here instead would make the
        # "fresh rollout" and "replay" batches indistinguishable.
        self._rollout_disc_obs.append(entry)

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> Tuple[float, ...]:
        """PPO update, then the discriminator update on MimicKit's schedule.

        The two are separate optimisation loops with separate optimizers (see the
        class docstring); ``_update_disc`` runs after the PPO epochs.

        Returns
        -------
        Tuple of mean losses:
            (value_loss, surrogate_loss, amp_loss, grad_pen_loss,
             policy_pred, expert_pred, acc_policy, acc_expert, kl)
        """
        # Per-minibatch statistics are accumulated on-device and read back with a
        # single .item() after the loop. Calling .item() per minibatch forces a
        # GPU->CPU sync each iteration (~8 syncs x total_updates), which serializes
        # the optimizer and grows with env count; batching keeps the numbers
        # identical while issuing one sync per scalar at the end.
        dev = self.device
        sum_value_loss = torch.zeros((), device=dev)
        sum_surrogate_loss = torch.zeros((), device=dev)
        sum_actor_grad_norm = torch.zeros((), device=dev)
        sum_critic_grad_norm = torch.zeros((), device=dev)
        sum_clip_frac = torch.zeros((), device=dev)
        mean_kl_divergence = 0.0

        total_updates = self.num_learning_epochs * self.num_mini_batches

        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for sample in generator:
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                _rnd_state_batch,
            ) = sample

            hid_a, hid_c = (None, None)
            if hid_states_batch is not None:
                hid_a, hid_c = hid_states_batch

            # ----------------------------------------------------------
            # Policy forward
            # ----------------------------------------------------------
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_a)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_c
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # ----------------------------------------------------------
            # Adaptive KL / learning-rate schedule
            # ----------------------------------------------------------
            # The KL is always MEASURED (it is the one number that says whether the
            # policy is actually moving); only the learning-rate reaction is gated
            # on the adaptive schedule. It used to be computed inside that gate, so
            # a `schedule: fixed` run logged Loss/kl_divergence == 0.
            if self.desired_kl is not None:
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    mean_kl_divergence += kl_mean.item()

                    if self.schedule == "adaptive":
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(
                                self.min_learning_rate, self.learning_rate / 1.5
                            )
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(
                                self.max_learning_rate, self.learning_rate * 1.5
                            )
                        # Update only the policy group's LR; the critic and the
                        # disc keep their own fixed rates (separate optimizers).
                        self._policy_param_group["lr"] = self.learning_rate

            # ----------------------------------------------------------
            # PPO surrogate loss
            # ----------------------------------------------------------
            advantages_batch = torch.squeeze(advantages_batch)
            if self.norm_adv_clip > 0.0:
                # MimicKit ``norm_adv_clip``: the advantages arrive already
                # standardised (RolloutStorage.compute_returns), and a single
                # heavy-tailed transition otherwise dominates a minibatch's
                # surrogate gradient — which shows up as an outsized KL and, under
                # the adaptive schedule, as a collapsed learning rate.
                advantages_batch = advantages_batch.clamp(
                    -self.norm_adv_clip, self.norm_adv_clip
                )
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            action_bound_loss = (
                torch.relu(mu_batch.abs() - 1.0).pow(2).sum(dim=-1).mean()
            )

            # ----------------------------------------------------------
            # Value loss
            # ----------------------------------------------------------
            if self.use_clipped_value_loss:
                # The PPO value clip is an ABSOLUTE bound in return units, so it
                # only makes sense when returns are O(1). Imitation returns here
                # are ~60 (a dense ~0.65/step style reward over a ~230-step clip),
                # and a +-0.2 clip then freezes the critic's gradient after it has
                # moved 0.2 per iteration -- the value loss in the 5.5k-iteration
                # jumps_15s run never dropped below ~35. ``value_clip_scale``
                # makes the bound RELATIVE to the batch's return scale; 0 keeps
                # the legacy absolute ``clip_param``.
                if self.value_clip_scale > 0.0:
                    v_clip = self.value_clip_scale * (
                        returns_batch.std().detach() + 1.0e-8
                    )
                else:
                    v_clip = self.clip_param
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-v_clip, v_clip)
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            ppo_loss = (
                surrogate_loss
                + self.action_bound_weight * action_bound_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # ----------------------------------------------------------
            # PPO backward pass (policy/value only — the discriminator has its
            # own loop and optimizer, see _update_disc)
            #
            # ONE backward over the summed loss (the actor and critic MLPs are
            # disjoint subgraphs, so each parameter still receives exactly its own
            # loss's gradient), then clip and step the two groups SEPARATELY so
            # the critic's gradient magnitude cannot rescale the actor's step.
            # ----------------------------------------------------------
            self.optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            ppo_loss.backward()
            a_norm = nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
            c_norm = nn.utils.clip_grad_norm_(self.critic_params, self.critic_max_grad_norm)
            self.optimizer.step()
            self.critic_optimizer.step()

            # ----------------------------------------------------------
            # Statistics
            # ----------------------------------------------------------
            with torch.no_grad():
                sum_value_loss += value_loss.detach()
                sum_surrogate_loss += surrogate_loss.detach()
                sum_actor_grad_norm += a_norm.detach()
                sum_critic_grad_norm += c_norm.detach()
                sum_clip_frac += (
                    (ratio - 1.0).abs() > self.clip_param
                ).float().mean()

        # Read the accumulated device-side statistics back to host in one shot.
        mean_value_loss = sum_value_loss.item() / total_updates
        mean_surrogate_loss = sum_surrogate_loss.item() / total_updates
        mean_kl_divergence /= total_updates
        self.last_actor_grad_norm = sum_actor_grad_norm.item() / total_updates
        self.last_critic_grad_norm = sum_critic_grad_norm.item() / total_updates
        self.last_clip_frac = sum_clip_frac.item() / total_updates

        disc_stats = self._update_disc()

        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            disc_stats["amp_loss"],
            disc_stats["grad_pen_loss"],
            disc_stats["policy_pred"],
            disc_stats["expert_pred"],
            disc_stats["acc_policy"],
            disc_stats["acc_expert"],
            mean_kl_divergence,
        )

    def _update_disc(self) -> dict:
        """Train the discriminator on MimicKit's schedule.

        Step count and batch size follow MimicKit ``amp_agent._update_model``::

            batch = ceil(disc_batch_size * num_envs)
            steps = ceil(rollout_samples / batch) * disc_epochs

        and each batch mixes this iteration's disc observations with
        ``disc_replay_samples`` drawn from the replay buffer, exactly as
        ``ADDDiscriminator._update_js`` does on the FoRL-SHAC side (record ->
        replay.push -> sample fresh+replay with replacement).

        For ADD the "policy" rows are the reduced residuals ``demo - agent`` and
        ``demo_fn`` returns the zero vector (the perfect-match positive class);
        for AMP they are agent disc observations and ``demo_fn`` samples the
        motion library.
        """
        dev = self.device
        zero = torch.zeros((), device=dev)
        empty = {
            "amp_loss": 0.0, "grad_pen_loss": 0.0, "policy_pred": 0.0,
            "expert_pred": 0.0, "acc_policy": 0.0, "acc_expert": 0.0,
            "num_steps": 0, "batch_size": 0,
        }
        if not self._rollout_disc_obs:
            return empty

        # Fresh rollout observations. MimicKit fills the replay buffer with the
        # complete rollout until capacity, then admits only
        # `disc_replay_samples` fresh rows per iteration. This prevents a large
        # vectorized rollout from replacing nearly the whole replay history.
        agent_obs = torch.cat(self._rollout_disc_obs, dim=0)
        self._rollout_disc_obs = []
        if len(self.amp_storage) < self.amp_storage.buffer_size:
            replay_insert = agent_obs
        else:
            insert_n = min(agent_obs.shape[0], self.disc_replay_samples)
            insert_idx = torch.randperm(agent_obs.shape[0], device=dev)[:insert_n]
            replay_insert = agent_obs[insert_idx]
        self.amp_storage.insert(replay_insert)

        num_envs = self.storage.num_envs if self.storage is not None else 1
        batch_size = max(1, int(math.ceil(self.disc_batch_size * num_envs)))
        num_batches = max(1, int(math.ceil(agent_obs.shape[0] / batch_size)))
        num_steps = num_batches * max(1, self.disc_epochs)

        sum_amp_loss = zero.clone()
        sum_grad_pen = zero.clone()
        sum_policy_pred = zero.clone()
        sum_expert_pred = zero.clone()
        sum_acc_policy = zero.clone()
        sum_acc_expert = zero.clone()
        acc_policy_den = acc_expert_den = 0

        for _ in range(num_steps):
            # MimicKit samples a fresh batch from this rollout and an equally
            # sized replay batch, then concatenates them. `disc_replay_samples`
            # controls replay INSERTION once full; it is not the train-time
            # replay batch size. The old implementation pooled 1000 replay rows
            # with a 98k-row rollout, making replay only ~1% of the distribution.
            fresh_idx = torch.randint(
                0, agent_obs.shape[0], (batch_size,), device=dev
            )
            fresh_batch = agent_obs[fresh_idx]
            replay_batch = self.amp_storage.sample(batch_size)
            policy_batch = torch.cat([fresh_batch, replay_batch], dim=0)
            expert_batch = self.demo_fn(batch_size).to(dev)

            # One trunk forward for both the logits and the R1 penalty; the
            # unfused forward()+compute_loss() pair ran it three times per batch
            # (logits, then once inside each compute_grad_pen call — two of them
            # for ADD's two-sided penalty). Numerically identical, see
            # Discriminator.compute_loss_fused.
            amp_loss, grad_pen_loss, policy_d, expert_d = self.discriminator.compute_loss_fused(
                combined_obs=torch.cat([policy_batch, expert_batch], dim=0),
                num_policy=policy_batch.shape[0],
                lambda_=self.grad_penalty_coeff,
                two_sided=(self.mode == "add"),
            )

            disc_loss = amp_loss + grad_pen_loss
            if self.disc_logit_reg > 0.0:
                # Weights only (no bias), matching MimicKit's
                # `get_disc_logit_weights` and SHAC's `p.ndim > 1` filter.
                disc_loss = disc_loss + self.disc_logit_reg * self.discriminator.linear.weight.pow(2).sum()

            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

            if self.mode != "add":
                self.discriminator.update_normalization(
                    policy_batch.detach(), expert_batch.detach()
                )

            with torch.no_grad():
                policy_prob = torch.sigmoid(policy_d)
                expert_prob = torch.sigmoid(expert_d)
                sum_amp_loss += amp_loss.detach()
                sum_grad_pen += grad_pen_loss.detach()
                sum_policy_pred += policy_prob.mean()
                sum_expert_pred += expert_prob.mean()
                sum_acc_policy += (torch.round(policy_prob) == 0).sum()
                sum_acc_expert += (torch.round(expert_prob) == 1).sum()
                acc_policy_den += policy_prob.numel()
                acc_expert_den += expert_prob.numel()

        if self.mode == "add":
            # MimicKit freezes DiffNorm for the whole policy/discriminator
            # update, then records the complete rollout once. Updating from each
            # sampled minibatch made the classifier's coordinate system drift
            # during the optimization loop.
            self.discriminator.update_normalization(agent_obs.detach())

        return {
            "amp_loss": sum_amp_loss.item() / num_steps,
            "grad_pen_loss": sum_grad_pen.item() / num_steps,
            "policy_pred": sum_policy_pred.item() / num_steps,
            "expert_pred": sum_expert_pred.item() / num_steps,
            "acc_policy": sum_acc_policy.item() / max(1, acc_policy_den),
            "acc_expert": sum_acc_expert.item() / max(1, acc_expert_den),
            "num_steps": num_steps,
            "batch_size": batch_size,
        }
