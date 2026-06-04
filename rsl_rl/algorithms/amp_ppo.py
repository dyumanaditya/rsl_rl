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

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer


class AMP_PPO:
    """PPO with a jointly-trained AMP / ADD discriminator.

    The discriminator loss is combined with the PPO loss in a single
    backward pass, exactly as in amp-rsl-rl, rather than being trained
    in a separate optimisation step.

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
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        # Discriminator hyper-parameters
        disc_lr: Optional[float] = None,
        amp_replay_buffer_size: int = 100_000,
        grad_penalty_coeff: float = 10.0,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.mode = mode

        # Discriminator
        self.discriminator = discriminator.to(self.device)
        disc_obs_size = discriminator.input_dim
        self.amp_storage = ReplayBuffer(disc_obs_size, amp_replay_buffer_size, device)
        self.demo_fn = demo_fn
        self.grad_penalty_coeff = grad_penalty_coeff

        # Actor-critic
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage: Optional[RolloutStorage] = None

        # Single combined optimizer (actor-critic + discriminator).
        # Discriminator trunk and head get separate weight decay, mirroring
        # the amp-rsl-rl optimizer setup.
        _disc_lr = disc_lr if disc_lr is not None else learning_rate
        params = [
            {"params": self.actor_critic.parameters(), "lr": learning_rate},
            {
                "params": self.discriminator.trunk.parameters(),
                "lr": _disc_lr,
                "weight_decay": 10e-4,
            },
            {
                "params": self.discriminator.linear.parameters(),
                "lr": _disc_lr,
                "weight_decay": 10e-2,
            },
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        # Only the actor-critic group follows the adaptive-KL LR schedule. The
        # discriminator groups must keep their fixed ``_disc_lr`` (mimickit trains
        # the disc with an independent optimizer); letting the policy KL drive the
        # disc LR over-trains the discriminator and starves the style reward.
        self._policy_param_group = self.optimizer.param_groups[0]

        self.transition = RolloutStorage.Transition()

        # PPO hyper-parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
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
    ) -> None:
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
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
        """
        if self.mode == "add" and demo_disc_obs is not None:
            entry = (demo_disc_obs - disc_obs).detach()
        else:
            entry = disc_obs.detach()
        self.amp_storage.insert(entry)

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> Tuple[float, ...]:
        """Joint PPO + discriminator update.

        Mirrors amp-rsl-rl AMP_PPO.update() but adapted for flat obs and
        single-obs replay buffer.

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
        sum_amp_loss = torch.zeros((), device=dev)
        sum_grad_pen_loss = torch.zeros((), device=dev)
        sum_policy_pred = torch.zeros((), device=dev)
        sum_expert_pred = torch.zeros((), device=dev)
        sum_acc_policy_num = torch.zeros((), device=dev)
        sum_acc_expert_num = torch.zeros((), device=dev)
        mean_kl_divergence = 0.0
        acc_policy_den = acc_expert_den = 0

        total_updates = self.num_learning_epochs * self.num_mini_batches
        disc_batch_size = (
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        )

        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        disc_gen = self.amp_storage.feed_forward_generator(
            num_mini_batch=total_updates,
            mini_batch_size=disc_batch_size,
            allow_replacement=True,
        )

        for sample, disc_policy_batch in zip(generator, disc_gen):
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
            if self.desired_kl is not None and self.schedule == "adaptive":
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

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update only the policy group's LR; leave the disc groups
                    # pinned at their fixed _disc_lr (see __init__).
                    self._policy_param_group["lr"] = self.learning_rate

            # ----------------------------------------------------------
            # PPO surrogate loss
            # ----------------------------------------------------------
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ----------------------------------------------------------
            # Value loss
            # ----------------------------------------------------------
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            ppo_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # ----------------------------------------------------------
            # Discriminator loss
            # ----------------------------------------------------------
            disc_policy_batch = disc_policy_batch.to(self.device)
            expert_batch = self.demo_fn(disc_policy_batch.shape[0]).to(self.device)

            B_pol = disc_policy_batch.shape[0]
            combined = torch.cat([disc_policy_batch, expert_batch], dim=0)
            combined_d = self.discriminator(combined)
            policy_d, expert_d = combined_d[:B_pol], combined_d[B_pol:]

            amp_loss, grad_pen_loss = self.discriminator.compute_loss(
                policy_d=policy_d,
                expert_d=expert_d,
                expert_obs=expert_batch,
                lambda_=self.grad_penalty_coeff,
                policy_obs=disc_policy_batch if self.mode == "add" else None,
            )

            # ----------------------------------------------------------
            # Combined backward pass
            # ----------------------------------------------------------
            loss = ppo_loss + amp_loss + grad_pen_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Update normaliser: for ADD, only use policy diffs (not zeros) so
            # DiffNorm tracks the true diff scale, not an average with zero.
            if self.mode == "add":
                self.discriminator.update_normalization(disc_policy_batch.detach())
            else:
                self.discriminator.update_normalization(
                    disc_policy_batch.detach(), expert_batch.detach()
                )

            # ----------------------------------------------------------
            # Statistics
            # ----------------------------------------------------------
            with torch.no_grad():
                policy_prob = torch.sigmoid(policy_d)
                expert_prob = torch.sigmoid(expert_d)

                sum_value_loss += value_loss.detach()
                sum_surrogate_loss += surrogate_loss.detach()
                sum_amp_loss += amp_loss.detach()
                sum_grad_pen_loss += grad_pen_loss.detach()
                sum_policy_pred += policy_prob.mean()
                sum_expert_pred += expert_prob.mean()
                sum_acc_policy_num += (torch.round(policy_prob) == 0).sum()
                sum_acc_expert_num += (torch.round(expert_prob) == 1).sum()
                acc_policy_den += policy_prob.numel()
                acc_expert_den += expert_prob.numel()

        # Read the accumulated device-side statistics back to host in one shot.
        mean_value_loss = sum_value_loss.item() / total_updates
        mean_surrogate_loss = sum_surrogate_loss.item() / total_updates
        mean_amp_loss = sum_amp_loss.item() / total_updates
        mean_grad_pen_loss = sum_grad_pen_loss.item() / total_updates
        mean_policy_pred = sum_policy_pred.item() / total_updates
        mean_expert_pred = sum_expert_pred.item() / total_updates
        mean_kl_divergence /= total_updates
        acc_policy = sum_acc_policy_num.item() / max(1, acc_policy_den)
        acc_expert = sum_acc_expert_num.item() / max(1, acc_expert_den)

        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_amp_loss,
            mean_grad_pen_loss,
            mean_policy_pred,
            mean_expert_pred,
            acc_policy,
            acc_expert,
            mean_kl_divergence,
        )
