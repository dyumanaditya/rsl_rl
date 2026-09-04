# Copyright (c) 2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from amp-rsl-rl (ami-iit/amp-rsl-rl) AMPOnPolicyRunner.
#
# Key adaptations for the G1 env / current rsl_rl stack:
#  - obs interface: (obs, extras) flat tensors, not TensorDicts.
#  - disc_obs sourced from extras["disc_obs"] / extras["disc_obs_demo"].
#  - No AMPLoader: demo obs come from env.base_env.fetch_disc_obs_demo()
#    (AMP) or are zeros (ADD).
#  - Discriminator rewards computed DURING the rollout (inside
#    inference_mode) – the reward mixing happens before process_env_step,
#    exactly as in amp-rsl-rl.
#  - AMP/ADD use a combined PPO + discriminator backward pass via AMP_PPO.update().
#  - DeepMimic uses plain PPO on the env-provided tracking reward.

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.algorithms.amp_ppo import AMP_PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.utils import store_code_state
from utils.contact_force_logging import pop_contact_force_stats
from utils.metric_logging import episode_tags, log_scalar_aliases


class AMPOnPolicyRunner:
    """On-policy runner for imitation training.

    Mirrors the training loop of amp-rsl-rl's ``AMPOnPolicyRunner`` but
    works with the existing G1ImitationEnv interface:

    * ``env.get_observations()`` returns ``(obs, extras)``
    * ``extras["disc_obs"]`` is the agent discriminator observation
    * ``extras["disc_obs_demo"]`` is the reference discriminator observation
      (ADD mode only; for AMP mode demo is sampled from the motion library)

    Configuration keys expected in ``train_cfg``
    -----------------------------------------
    Standard rsl_rl keys (policy, algorithm, num_steps_per_env, …) plus:

    ``train_cfg["imitation"]``
        Should expose the ``ImitationCfg`` attributes used here:
        ``mode``, ``task_reward_weight``, ``disc_reward_weight``,
        ``disc_reward_scale``, ``disc_lr``, and optionally
        ``hidden_dims``, ``empirical_normalization``.
        This dict is obtained by the runner from the ``base_env._im_cfg``
        object — the caller does not need to put it in ``train_cfg``.

        ``mode="deepmimic"`` has no discriminator.  The env computes the
        closed-form DeepMimic tracking reward and this runner trains regular
        PPO on that reward while preserving the imitation logging/video hooks.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = dict(train_cfg["algorithm"])
        self.policy_cfg = dict(train_cfg["policy"])
        self.device = device
        self.env = env

        # ------------------------------------------------------------------
        # Resolve env-level imitation config
        # ------------------------------------------------------------------
        base_env = getattr(env, "base_env", env)
        im_cfg = base_env._im_cfg
        self.im_cfg = im_cfg
        self.imitation_mode = getattr(im_cfg, "mode", "none")
        if self.imitation_mode not in ("amp", "add", "deepmimic"):
            raise ValueError(f"Unknown imitation mode: {self.imitation_mode}")
        self.has_discriminator = self.imitation_mode in ("amp", "add")
        # Set below when the ADD discriminator has external root tracking on.
        self.aux_root = False
        disc_obs_size: int = int(getattr(base_env, "disc_obs_size", 0))

        # ------------------------------------------------------------------
        # Observations
        # ------------------------------------------------------------------
        obs, extras = self.env.get_observations()
        num_obs: int = obs.shape[1]
        num_critic_obs: int = extras["observations"].get("critic", obs).shape[1]

        # ------------------------------------------------------------------
        # Policy network
        # ------------------------------------------------------------------
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Optionally freeze the action-noise std at its init value (MimicKit uses
        # a FIXED std for AMP/ADD). Leaving it learnable with entropy_coef=0 and a
        # low init collapses exploration and stalls imitation learning.
        if getattr(im_cfg, "fixed_action_std", False):
            if hasattr(actor_critic, "std") and isinstance(actor_critic.std, torch.nn.Parameter):
                actor_critic.std.requires_grad_(False)
                print(
                    f"[AMPOnPolicyRunner] action std frozen at "
                    f"{actor_critic.std.detach().mean().item():.4f} (fixed_action_std=True)."
                )

        if self.has_discriminator:
            if disc_obs_size <= 0:
                raise RuntimeError(
                    f"Imitation mode {self.imitation_mode!r} requires discriminator observations, "
                    "but base_env.disc_obs_size is zero."
                )

            # ------------------------------------------------------------------
            # Discriminator
            # ------------------------------------------------------------------
            hidden_dims = getattr(im_cfg, "hidden_dims", [1024, 512])
            empirical_norm = getattr(im_cfg, "empirical_normalization", False)
            # ADD uses DiffNorm (scale by mean absolute diff); AMP uses standard norm.
            disc_norm_kwargs = (
                {"diff_normalization": True} if im_cfg.mode == "add"
                else {"empirical_normalization": empirical_norm}
            )
            # External root tracking (imitation.use_aux_root_tracking): resolve the
            # same ADD config FoRL-SHAC uses and hand the discriminator the
            # per-feature-group slices so it can split the global root out of the
            # adversarial differential and reward it separately. AMP has no aux root
            # term, so it is only wired for ADD.
            add_cfg = None
            disc_feature_groups = None
            if im_cfg.mode == "add":
                from imitation.wadd import resolve_add_cfg
                add_cfg = resolve_add_cfg(im_cfg)
                disc_feature_groups = getattr(base_env, "disc_feature_groups", None)
                if getattr(add_cfg, "use_aux_root_tracking", False) and not disc_feature_groups:
                    raise RuntimeError(
                        "imitation.use_aux_root_tracking is on but the env exposes no "
                        "disc_feature_groups; cannot locate the global root slots."
                    )
            discriminator = Discriminator(
                input_dim=disc_obs_size,
                hidden_layer_sizes=hidden_dims,
                reward_scale=im_cfg.disc_reward_scale,
                device=self.device,
                feature_groups=disc_feature_groups,
                add_cfg=add_cfg,
                **disc_norm_kwargs,
            ).to(self.device)
            self.aux_root = bool(getattr(discriminator, "aux_root", False))
            if self.aux_root:
                print(
                    f"[AMPOnPolicyRunner] external root tracking ON: disc differential "
                    f"{disc_obs_size} -> {discriminator.input_dim} dims (global root removed); "
                    f"aux root reward pos_w={add_cfg.aux_root_weight} ori_w={add_cfg.aux_root_ori_weight} "
                    f"kind={add_cfg.aux_root_reward_kind}."
                )

            # ------------------------------------------------------------------
            # Demo function
            # ------------------------------------------------------------------
            # ADD's positive class is the zero vector at the width the classifier
            # actually sees, i.e. the REDUCED width under external root tracking.
            disc_input_dim = discriminator.input_dim
            if im_cfg.mode == "amp":
                demo_fn = lambda n: base_env.fetch_disc_obs_demo(n).to(self.device)  # noqa: E731
            else:
                demo_fn = lambda n: torch.zeros(n, disc_input_dim, device=self.device)  # noqa: E731

            # ------------------------------------------------------------------
            # AMP_PPO algorithm
            # ------------------------------------------------------------------
            # Strip keys that PPO supports but AMP_PPO does not.
            _amp_ppo_params = set(AMP_PPO.__init__.__code__.co_varnames)
            alg_kwargs = {k: v for k, v in self.alg_cfg.items() if k in _amp_ppo_params}
            alg_kwargs.pop("class_name", None)

            self.alg: AMP_PPO | PPO = AMP_PPO(
                actor_critic=actor_critic,
                discriminator=discriminator,
                demo_fn=demo_fn,
                mode=im_cfg.mode,
                disc_lr=getattr(im_cfg, "disc_lr", None),
                amp_replay_buffer_size=getattr(im_cfg, "disc_buffer_size", 100_000),
                grad_penalty_coeff=getattr(im_cfg, "disc_grad_penalty", 10.0),
                # MimicKit disc schedule — the same three keys FoRL-SHAC reads.
                disc_epochs=getattr(im_cfg, "disc_epochs", 2),
                disc_batch_size=getattr(im_cfg, "disc_batch_size", 2),
                disc_replay_samples=getattr(im_cfg, "disc_replay_samples", 1000),
                disc_logit_reg=getattr(im_cfg, "disc_logit_reg", 0.0),
                device=self.device,
                **alg_kwargs,
            )
            # self.num_steps_per_env is assigned further down in __init__, so read
            # the rollout length straight from the config here.
            _T = int(self.cfg["num_steps_per_env"])
            _dbatch = max(1, int(self.alg.disc_batch_size * self.env.num_envs))
            _dsteps = max(1, -(-_T * self.env.num_envs // _dbatch)) * max(
                1, self.alg.disc_epochs
            )
            print(
                f"[AMPOnPolicyRunner] disc schedule (MimicKit): {_dsteps} steps x "
                f"batch {_dbatch} (disc_epochs={self.alg.disc_epochs}, "
                f"disc_batch_size={self.alg.disc_batch_size}/env, "
                f"disc_replay_samples={self.alg.disc_replay_samples})."
            )
        else:
            # DeepMimic has no discriminator.  The env replaces the task reward
            # with its closed-form tracking reward, so train normal PPO on it.
            if self.alg_cfg.get("rnd_cfg") is not None:
                rnd_state = extras["observations"].get("rnd_state")
                if rnd_state is None:
                    raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
                self.alg_cfg["rnd_cfg"]["num_state"] = rnd_state.shape[1]
                dt = getattr(env, "dt", None)
                if dt is None and hasattr(base_env, "get_timestep"):
                    dt = base_env.get_timestep()
                if dt is None:
                    dt = getattr(getattr(getattr(env, "cfg", None), "sim", None), "dt", 1.0)
                self.alg_cfg["rnd_cfg"]["weight"] *= dt
            if self.alg_cfg.get("symmetry_cfg") is not None:
                self.alg_cfg["symmetry_cfg"]["_env"] = env

            alg_class = eval(self.alg_cfg.get("class_name", "PPO"))
            _ppo_params = set(alg_class.__init__.__code__.co_varnames)
            alg_kwargs = {k: v for k, v in self.alg_cfg.items() if k in _ppo_params}
            alg_kwargs.pop("class_name", None)
            self.alg: AMP_PPO | PPO = alg_class(actor_critic, device=self.device, **alg_kwargs)
            print("[AMPOnPolicyRunner] DeepMimic tracking reward enabled (no discriminator).")

        self.num_steps_per_env: int = self.cfg["num_steps_per_env"]
        self.save_interval: int = self.cfg["save_interval"]
        self.empirical_normalization: bool = self.cfg.get("empirical_normalization", False)

        # Action-rate regularisation (BeyondMimic action_rate_l2), byte-for-byte
        # the FoRL-SHAC term in forl/algorithms/shac.py: it is added to the total
        # reward DIRECTLY (not gated by task_reward_weight, so it stays live for
        # pure-imitation ADD) and scaled by the control period so its absolute
        # magnitude matches IsaacLab's step_dt-multiplied reward manager.
        # ``env.action_rate_reward_weight`` used to be read only by SHAC, so under
        # alg=ppo it was a silent no-op and the two algorithms optimised different
        # objectives from the same config.
        env_cfg = getattr(getattr(self.env, "cfg", None), "env", None)
        self.action_rate_weight = float(
            getattr(env_cfg, "action_rate_reward_weight", 0.0) if env_cfg else 0.0
        )
        self._action_rate_dt = float(
            getattr(getattr(getattr(self.env, "cfg", None), "sim", None), "dt", 0.02)
        )
        self._act_rate_prev: torch.Tensor | None = None
        self.mean_action_rate_reward = 0.0
        if self.action_rate_weight != 0.0:
            print(
                f"[AMPOnPolicyRunner] action-rate reward enabled: weight="
                f"{self.action_rate_weight} * dt({self._action_rate_dt}) * "
                f"sum_j (a_t - a_(t-1))^2 (BeyondMimic-scaled; added directly, "
                f"not gated by task_reward_weight)."
            )

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.logger_type = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        # Optional live renderer (enabled when env.cfg.viz.render is True).
        self.renderer = self._make_renderer()

        # Optional periodic single-robot policy video logged to wandb.
        self.video_recorder = None
        self.video_interval = 0
        self._maybe_init_video_recorder()

    def _maybe_init_video_recorder(self):
        viz_cfg = getattr(getattr(self.env, "cfg", None), "viz", None)
        if viz_cfg is None or not getattr(viz_cfg, "record_policy_video", False):
            return
        if self.cfg.get("logger", "tensorboard").lower() != "wandb":
            print("[AMPOnPolicyRunner] policy video requires the wandb logger; skipping.")
            return
        try:
            from utils.policy_video import PolicyVideoRecorder
            self.video_recorder = PolicyVideoRecorder(
                self.env.cfg,
                device=self.device,
                writer=self.writer,
                video_steps=getattr(viz_cfg, "video_steps", 600),
                width=getattr(viz_cfg, "video_width", 1280),
                height=getattr(viz_cfg, "video_height", 720),
            )
            self.video_interval = int(getattr(viz_cfg, "video_interval", 200))
        except Exception as exc:  # noqa: BLE001
            print(f"[AMPOnPolicyRunner] Warning: could not init policy video recorder: {exc}")
            self.video_recorder = None

    def _record_policy_video(self, it: int):
        if self.video_recorder is None or not self.video_recorder.enabled:
            return
        if self.video_interval <= 0 or it % self.video_interval != 0:
            return
        self.eval_mode()
        try:
            self.video_recorder.record(
                it,
                self.alg.actor_critic.act_inference,
                obs_normalizer=self.obs_normalizer,
            )
        finally:
            self.train_mode()

    def _apply_action_rate_reward(self, rewards, actions, dones, act_rate_sum):
        """Add the BeyondMimic action-rate penalty, mirroring FoRL-SHAC exactly.

        ``-|w| * dt * sum_j (a_t - a_(t-1))^2``, added straight onto the already
        blended reward so it survives ``task_reward_weight=0``. Done envs get
        their previous action reset to zero, so the first action after an RSI
        reset is penalised against zero rather than against a stale action from
        another clip -- the same rule ``forl/algorithms/shac.py`` applies.
        No-op when ``env.action_rate_reward_weight`` is 0 (the default).
        """
        if self.action_rate_weight == 0.0:
            return rewards
        if self._act_rate_prev is None or self._act_rate_prev.shape != actions.shape:
            self._act_rate_prev = torch.zeros_like(actions)
        act_rate = torch.sum((actions - self._act_rate_prev) ** 2, dim=-1)
        rewards = rewards + self.action_rate_weight * self._action_rate_dt * act_rate
        self._act_rate_prev = actions.clone()
        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._act_rate_prev[done_ids] = 0.0
        if act_rate_sum is not None:
            # In-place: the caller owns the accumulator (a 0-dim device tensor
            # read back with one .item() after the rollout, no per-step sync).
            act_rate_sum.add_(act_rate.mean())
        return rewards

    @staticmethod
    def _normalize_without_update(normalizer, observations):
        """Apply a running normalizer without recording terminal observations."""
        was_training = normalizer.training
        normalizer.eval()
        try:
            return normalizer(observations)
        finally:
            normalizer.train(was_training)

    def _timeout_values(self, infos):
        """Evaluate pre-reset terminal states used for timeout bootstrapping.

        Recurrent policies retain the legacy current-state fallback because an
        all-env terminal forward would advance live environments' hidden state.
        The G1 imitation policies are feed-forward.
        """
        if self.alg.actor_critic.is_recurrent:
            return None
        terminal_obs = infos.get("terminal_critic_obs")
        time_outs = infos.get("time_outs")
        if terminal_obs is None or time_outs is None:
            return None
        terminal_obs = terminal_obs.to(self.device)
        terminal_obs = self._normalize_without_update(
            self.critic_obs_normalizer, terminal_obs
        )
        return self.alg.actor_critic.evaluate(terminal_obs).detach().squeeze(-1)

    def _make_renderer(self):
        try:
            viz_cfg = getattr(getattr(self.env, "cfg", None), "viz", None)
            if viz_cfg is None or not getattr(viz_cfg, "render", False):
                return None
            base_env = getattr(self.env, "base_env", self.env)
            from utils.render import Render
            return Render(self.env.cfg, base_env.model)
        except Exception as exc:
            print(f"[AMPOnPolicyRunner] Warning: could not initialise renderer: {exc}")
            return None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # ---- logging setup ----
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            else:
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs = self.obs_normalizer(obs.to(self.device))
        critic_obs = self.critic_obs_normalizer(critic_obs.to(self.device))

        # Initial disc obs – sourced directly from the env state (before any step)
        base_env = getattr(self.env, "base_env", self.env)
        disc_obs: torch.Tensor | None = None
        demo_disc_obs: torch.Tensor | None = None
        if self.has_discriminator:
            disc_obs = base_env.disc_obs.clone().to(self.device)
            if self.im_cfg.mode == "add":
                demo_disc_obs = base_env._disc_obs_demo.clone().to(self.device)

        self.train_mode()

        ep_infos = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        disc_rewbuffer: deque = deque(maxlen=100)
        task_rewbuffer: deque = deque(maxlen=100)
        style_rewbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_disc_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_task_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_style_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()
            # plant curriculum: anneal the reflected rotor inertia (no-op unless
            # sim.armature_curriculum is set).
            if hasattr(self.env, "maybe_update_armature"):
                self.env.maybe_update_armature(it)

            mean_style_reward_log = 0.0
            mean_task_reward_log = 0.0
            mean_aux_root_log = 0.0
            mean_aux_root_pos_log = 0.0
            mean_aux_root_ori_log = 0.0

            # ----------------------------------------------------------
            # Rollout (mirrors amp-rsl-rl learn() inner loop)
            # ----------------------------------------------------------
            with torch.inference_mode():
                # External-root-tracking stats accumulate ON DEVICE and are read
                # back with a single .item() after the rollout: a per-step .item()
                # would add three GPU->CPU syncs per step to the hot loop.
                aux_sums = (
                    [torch.zeros((), device=self.device) for _ in range(3)]
                    if self.aux_root else None
                )
                act_rate_sum = (
                    torch.zeros((), device=self.device)
                    if self.action_rate_weight != 0.0 else None
                )
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)

                    obs_raw, rewards, dones, infos = self.env.step(actions.to(self.env.device))

                    # Render
                    if self.renderer is not None:
                        base_env = getattr(self.env, "base_env", self.env)
                        self.renderer.render(base_env.body_q, base_env.body_qd)

                    obs_raw = obs_raw.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    obs = self.obs_normalizer(obs_raw)
                    if "critic" in infos.get("observations", {}):
                        critic_obs = self.critic_obs_normalizer(
                            infos["observations"]["critic"].to(self.device)
                        )
                    else:
                        critic_obs = obs

                    if self.has_discriminator:
                        # Disc obs for this transition (captured pre-reset by env)
                        next_disc_obs = infos.get("disc_obs")
                        if next_disc_obs is not None:
                            next_disc_obs = next_disc_obs.to(self.device)
                        else:
                            next_disc_obs = disc_obs  # fallback (shouldn't happen in imitation env)

                        next_demo_disc_obs: torch.Tensor | None = None
                        if self.im_cfg.mode == "add":
                            next_demo_disc_obs = infos.get("disc_obs_demo")
                            if next_demo_disc_obs is not None:
                                next_demo_disc_obs = next_demo_disc_obs.to(self.device)
                            else:
                                next_demo_disc_obs = demo_disc_obs

                        # ---- Style reward (computed inside inference_mode) ----
                        # NOTE: with external root tracking this is style + aux_root
                        # (predict_reward returns the sum, matching SHAC's disc_r).
                        if self.im_cfg.mode == "add":
                            # ADD: discriminator takes diff = demo - agent
                            diff = next_demo_disc_obs - next_disc_obs
                            style_rewards = self.alg.discriminator.predict_reward(diff)
                        else:
                            # AMP: discriminator takes agent disc_obs directly
                            style_rewards = self.alg.discriminator.predict_reward(next_disc_obs)

                        if aux_sums is not None:
                            disc = self.alg.discriminator
                            for i, part in enumerate(
                                (disc.last_aux_total, disc.last_aux_pos, disc.last_aux_ori)
                            ):
                                if part is not None:
                                    aux_sums[i] += part.mean()

                        mean_task_reward_log += rewards.mean().item()
                        mean_style_reward_log += style_rewards.mean().item()

                        # Mix task + style rewards (copy amp-rsl-rl 0.5/0.5 default
                        # but respect im_cfg weights if provided)
                        task_w = getattr(self.im_cfg, "task_reward_weight", 0.5)
                        disc_w = getattr(self.im_cfg, "disc_reward_weight", 0.5)
                        blended_rewards = task_w * rewards + disc_w * style_rewards
                        blended_rewards = self._apply_action_rate_reward(
                            blended_rewards, actions, dones, act_rate_sum
                        )

                        self.alg.process_env_step(
                            blended_rewards,
                            dones,
                            infos,
                            timeout_values=self._timeout_values(infos),
                        )

                        # Insert into disc replay buffer
                        self.alg.process_disc_step(next_disc_obs, next_demo_disc_obs)

                        # Advance disc obs
                        disc_obs = next_disc_obs
                        if self.im_cfg.mode == "add":
                            demo_disc_obs = next_demo_disc_obs
                    else:
                        # DeepMimic: env.step() already returns the tracking reward.
                        style_rewards = torch.zeros_like(rewards)
                        mean_task_reward_log += rewards.mean().item()
                        blended_rewards = self._apply_action_rate_reward(
                            rewards, actions, dones, act_rate_sum
                        )
                        self.alg.process_env_step(blended_rewards, dones, infos)

                    # ---- Logging ----
                    if self.log_dir is not None:
                        # Merge episode-end reward terms ("episode") and per-step
                        # metrics ("log", e.g. imitation tracking error) so neither
                        # channel is dropped when both are present on the same step.
                        log_entry = {}
                        if "episode" in infos:
                            log_entry.update(infos["episode"])
                        if "log" in infos:
                            log_entry.update(infos["log"])
                        if log_entry:
                            ep_infos.append(log_entry)

                        cur_reward_sum += blended_rewards
                        cur_disc_reward_sum += style_rewards
                        cur_task_reward_sum += rewards
                        if self.has_discriminator:
                            cur_style_reward_sum += style_rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        if self.has_discriminator:
                            disc_rewbuffer.extend(cur_disc_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        task_rewbuffer.extend(cur_task_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        if self.has_discriminator:
                            style_rewbuffer.extend(cur_style_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        cur_disc_reward_sum[new_ids] = 0
                        cur_task_reward_sum[new_ids] = 0
                        cur_style_reward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                start = stop
                self.alg.compute_returns(critic_obs)

            mean_style_reward_log /= self.num_steps_per_env
            mean_task_reward_log /= self.num_steps_per_env
            if act_rate_sum is not None:
                # Same key and same definition as FoRL-SHAC's Reward/action_rate_reward.
                self.mean_action_rate_reward = (
                    self.action_rate_weight
                    * self._action_rate_dt
                    * act_rate_sum.item()
                    / self.num_steps_per_env
                )
            if aux_sums is not None:
                mean_aux_root_log = aux_sums[0].item() / self.num_steps_per_env
                mean_aux_root_pos_log = aux_sums[1].item() / self.num_steps_per_env
                mean_aux_root_ori_log = aux_sums[2].item() / self.num_steps_per_env

            # ----------------------------------------------------------
            # Combined PPO + discriminator update
            # ----------------------------------------------------------
            if self.has_discriminator:
                (
                    mean_value_loss,
                    mean_surrogate_loss,
                    mean_amp_loss,
                    mean_grad_pen_loss,
                    mean_policy_pred,
                    mean_expert_pred,
                    mean_accuracy_policy,
                    mean_accuracy_expert,
                    mean_kl_divergence,
                ) = self.alg.update()
                mean_entropy = None
                mean_rnd_loss = None
                mean_symmetry_loss = None
            else:
                (
                    mean_value_loss,
                    mean_surrogate_loss,
                    mean_entropy,
                    mean_rnd_loss,
                    mean_symmetry_loss,
                ) = self.alg.update()
                mean_amp_loss = 0.0
                mean_grad_pen_loss = 0.0
                mean_policy_pred = 0.0
                mean_expert_pred = 0.0
                mean_accuracy_policy = 0.0
                mean_accuracy_expert = 0.0
                mean_kl_divergence = 0.0

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                self._record_policy_video(it)

            ep_infos.clear()

            if it == start_iter and self.log_dir is not None:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            # Union of keys across all collected dicts: reward terms only appear on
            # episode-end steps while tracking metrics appear every step, so keying
            # off ep_infos[0] alone would drop one or the other.
            keys = list({k: None for d in locs["ep_infos"] for k in d})
            for key in keys:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    val = ep_info[key]
                    if not isinstance(val, torch.Tensor):
                        val = torch.Tensor([val])
                    if len(val.shape) == 0:
                        val = val.unsqueeze(0)
                    infotensor = torch.cat((infotensor, val.to(self.device)))
                value = torch.mean(infotensor)
                log_scalar_aliases(self.writer, episode_tags(key), value, locs["it"])
                if "/" in key:
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/policy", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        # Actor/critic gradient norms + PPO clip fraction. Same keys FoRL-SHAC
        # logs, so the two algorithms are comparable in wandb. These are the
        # scalars that expose a starved actor: a clip fraction pinned near zero
        # with the learning rate parked on its adaptive-schedule floor means the
        # policy is not moving, however healthy the losses look.
        if hasattr(self.alg, "last_actor_grad_norm"):
            self.writer.add_scalar("Loss/Actor Grad Norm", self.alg.last_actor_grad_norm, locs["it"])
            self.writer.add_scalar("Loss/Critic Grad Norm", self.alg.last_critic_grad_norm, locs["it"])
            self.writer.add_scalar("Loss/clip_frac", self.alg.last_clip_frac, locs["it"])
        if hasattr(self.alg, "critic_learning_rate"):
            self.writer.add_scalar("Train/critic_lr", self.alg.critic_learning_rate, locs["it"])
        if self.action_rate_weight != 0.0:
            self.writer.add_scalar(
                "Reward/action_rate_reward", self.mean_action_rate_reward, locs["it"]
            )
        if self.has_discriminator:
            self.writer.add_scalar("Loss/amp_loss", locs["mean_amp_loss"], locs["it"])
            self.writer.add_scalar("Loss/grad_pen_loss", locs["mean_grad_pen_loss"], locs["it"])
            self.writer.add_scalar("Loss/policy_pred", locs["mean_policy_pred"], locs["it"])
            self.writer.add_scalar("Loss/expert_pred", locs["mean_expert_pred"], locs["it"])
            self.writer.add_scalar("Loss/accuracy_policy", locs["mean_accuracy_policy"], locs["it"])
            self.writer.add_scalar("Loss/accuracy_expert", locs["mean_accuracy_expert"], locs["it"])
            self.writer.add_scalar("Loss/kl_divergence", locs["mean_kl_divergence"], locs["it"])
            self.writer.add_scalar("Disc/loss", locs["mean_amp_loss"], locs["it"])
            self.writer.add_scalar("Disc/grad_penalty_loss", locs["mean_grad_pen_loss"], locs["it"])
            self.writer.add_scalar("Disc/agent_acc", locs["mean_accuracy_policy"], locs["it"])
            self.writer.add_scalar("Disc/demo_acc", locs["mean_accuracy_expert"], locs["it"])
            self.writer.add_scalar("Disc/policy_pred", locs["mean_policy_pred"], locs["it"])
            self.writer.add_scalar("Disc/expert_pred", locs["mean_expert_pred"], locs["it"])
            self.writer.add_scalar("Disc/mean_reward", locs["mean_style_reward_log"], locs["it"])
            if self.aux_root:
                # Split the (otherwise combined) disc reward into style vs the
                # external global-root aux so their balance is visible. Same keys
                # and same definition as forl/algorithms/shac.py, so a SHAC run and
                # a PPO run are directly comparable in wandb.
                aux = locs["mean_aux_root_log"]
                total = locs["mean_style_reward_log"]
                self.writer.add_scalar("Disc/aux_root_reward", aux, locs["it"])
                self.writer.add_scalar("Disc/aux_root_pos_reward", locs["mean_aux_root_pos_log"], locs["it"])
                self.writer.add_scalar("Disc/aux_root_ori_reward", locs["mean_aux_root_ori_log"], locs["it"])
                self.writer.add_scalar("Disc/style_reward", total - aux, locs["it"])
                self.writer.add_scalar("Disc/aux_root_fraction", aux / max(abs(total), 1e-6), locs["it"])
        else:
            if locs["mean_entropy"] is not None:
                self.writer.add_scalar("Loss/entropy", locs["mean_entropy"], locs["it"])
            if locs["mean_rnd_loss"] is not None:
                self.writer.add_scalar("Loss/rnd", locs["mean_rnd_loss"], locs["it"])
            if locs["mean_symmetry_loss"] is not None:
                self.writer.add_scalar("Loss/symmetry", locs["mean_symmetry_loss"], locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Per-leg contact forces over this iteration's rollout. Flushes the
        # substep records the sim accumulated; see utils/contact_force_logging.py
        # for how to read them when tuning cfg.sim.bundle_contact_force_thresh.
        for tag, value in pop_contact_force_stats(self.env).items():
            self.writer.add_scalar(tag, value, locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
        if len(locs["task_rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_task_reward", statistics.mean(locs["task_rewbuffer"]), locs["it"])
        if len(locs["style_rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_disc_reward", statistics.mean(locs["style_rewbuffer"]), locs["it"])
        str_ = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        log_string = (
            f"""{'#' * width}\n"""
            f"""{str_.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s"""
            f""" (collection: {locs['collection_time']:.3f}s,"""
            f""" learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
            f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
        )
        if self.has_discriminator:
            log_string += (
                f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                f"""{'Grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                f"""{'Policy / expert pred:':>{pad}} {locs['mean_policy_pred']:.3f} / {locs['mean_expert_pred']:.3f}\n"""
                f"""{'Disc acc (pol / exp):':>{pad}} {locs['mean_accuracy_policy']:.3f} / {locs['mean_accuracy_expert']:.3f}\n"""
            )
        elif locs["mean_entropy"] is not None:
            log_string += f"""{'Entropy:':>{pad}} {locs['mean_entropy']:.4f}\n"""
        if hasattr(self.alg, "last_actor_grad_norm"):
            log_string += (
                f"""{'Grad norm (actor/critic):':>{pad}} """
                f"""{self.alg.last_actor_grad_norm:.3f} / {self.alg.last_critic_grad_norm:.3f}\n"""
                f"""{'PPO clip frac / lr:':>{pad}} """
                f"""{self.alg.last_clip_frac:.3f} / {self.alg.learning_rate:.2e}\n"""
            )
        log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
        if len(locs["rewbuffer"]) > 0:
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            if self.has_discriminator:
                log_string += f"""{'Mean style reward:':>{pad}} {locs['mean_style_reward_log']:.4f}\n"""
                if self.aux_root:
                    log_string += (
                        f"""{'  of which aux-root (pos/ori):':>{pad}} {locs['mean_aux_root_log']:.4f}"""
                        f""" ({locs['mean_aux_root_pos_log']:.4f} / {locs['mean_aux_root_ori_log']:.4f})\n"""
                    )
            log_string += (
                f"""{'Mean task reward:':>{pad}} {locs['mean_task_reward_log']:.4f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )

        log_string += ep_string
        eta_seconds = (
            self.tot_time / (locs["it"] + 1) * (locs["num_learning_iterations"] - locs["it"])
        )
        eta_h, rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(rem, 60)
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s\n"""
        )
        print(log_string)

    # ------------------------------------------------------------------
    # Save / load / inference
    # ------------------------------------------------------------------

    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "optimizer_class_names": {
                "actor": type(self.alg.optimizer).__name__,
                "critic": type(self.alg.critic_optimizer).__name__
                if hasattr(self.alg, "critic_optimizer") else None,
                "disc": type(self.alg.disc_optimizer).__name__
                if self.has_discriminator else None,
            },
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # The critic has its own optimizer since the actor/critic split; older
        # checkpoints carry only the joint one, which load() tolerates.
        if hasattr(self.alg, "critic_optimizer"):
            saved_dict["critic_optimizer_state_dict"] = self.alg.critic_optimizer.state_dict()
        if self.has_discriminator:
            saved_dict["discriminator_state_dict"] = self.alg.discriminator.state_dict()
            # The discriminator has its own optimizer (MimicKit-style separate
            # disc update); it used to share `optimizer`, so checkpoints written
            # before that split carry no such entry — load() tolerates its absence.
            saved_dict["disc_optimizer_state_dict"] = self.alg.disc_optimizer.state_dict()
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.has_discriminator and "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(
                loaded_dict["discriminator_state_dict"], strict=False
            )
        if self.empirical_normalization and "obs_norm_state_dict" in loaded_dict:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            # Checkpoints written before the policy/discriminator optimizer split
            # hold one Adam with three param groups (actor-critic + disc trunk +
            # disc head); the current `optimizer` has only the actor group, so
            # loading them raises. Same for checkpoints predating the actor/critic
            # split, whose single group holds actor AND critic tensors. The
            # weights above are already restored, so fall back to fresh optimizer
            # state rather than failing the resume.
            saved_names = loaded_dict.get("optimizer_class_names", {})

            def load_optimizer_state(optimizer, state_key, name_key):
                state = loaded_dict.get(state_key)
                if state is None:
                    return
                current_name = type(optimizer).__name__
                saved_name = saved_names.get(name_key)
                if saved_name is None and state.get("param_groups"):
                    # Older checkpoints did not record the class. Adam groups
                    # contain `betas`; SGD groups contain `momentum`.
                    group = state["param_groups"][0]
                    saved_name = "Adam" if "betas" in group else (
                        "SGD" if "momentum" in group else None
                    )
                if saved_name is not None and saved_name != current_name:
                    print(
                        f"[AMPOnPolicyRunner] ignoring {saved_name} {name_key} optimizer "
                        f"state in {path}; current config uses {current_name}."
                    )
                    return
                try:
                    optimizer.load_state_dict(state)
                except (ValueError, KeyError) as exc:
                    print(
                        f"[AMPOnPolicyRunner] ignoring incompatible {name_key} optimizer "
                        f"state in {path} ({exc}); resuming with fresh state."
                    )

            load_optimizer_state(
                self.alg.optimizer, "optimizer_state_dict", "actor"
            )
            if hasattr(self.alg, "critic_optimizer"):
                load_optimizer_state(
                    self.alg.critic_optimizer,
                    "critic_optimizer_state_dict",
                    "critic",
                )
            if self.has_discriminator:
                load_optimizer_state(
                    self.alg.disc_optimizer,
                    "disc_optimizer_state_dict",
                    "disc",
                )
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.empirical_normalization:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.has_discriminator:
            self.alg.discriminator.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.has_discriminator:
            self.alg.discriminator.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path: str):
        self.git_status_repos.append(repo_file_path)
