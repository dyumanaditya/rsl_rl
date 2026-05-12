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
#  - Combined PPO + discriminator backward pass via AMP_PPO.update().

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.algorithms.amp_ppo import AMP_PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.utils import store_code_state


class AMPOnPolicyRunner:
    """On-policy runner for AMP / ADD imitation training.

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
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = dict(train_cfg["algorithm"])
        self.policy_cfg = dict(train_cfg["policy"])
        self.device = device
        self.env = env

        # ------------------------------------------------------------------
        # Resolve env-level AMP config
        # ------------------------------------------------------------------
        base_env = getattr(env, "base_env", env)
        im_cfg = base_env._im_cfg
        self.im_cfg = im_cfg
        disc_obs_size: int = base_env.disc_obs_size

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
        discriminator = Discriminator(
            input_dim=disc_obs_size,
            hidden_layer_sizes=hidden_dims,
            reward_scale=im_cfg.disc_reward_scale,
            device=self.device,
            **disc_norm_kwargs,
        ).to(self.device)

        # ------------------------------------------------------------------
        # Demo function
        # ------------------------------------------------------------------
        if im_cfg.mode == "amp":
            demo_fn = lambda n: base_env.fetch_disc_obs_demo(n).to(self.device)  # noqa: E731
        elif im_cfg.mode == "add":
            demo_fn = lambda n: torch.zeros(n, disc_obs_size, device=self.device)  # noqa: E731
        else:
            raise ValueError(f"Unknown imitation mode: {im_cfg.mode}")

        # ------------------------------------------------------------------
        # AMP_PPO algorithm
        # ------------------------------------------------------------------
        # Strip keys that PPO supports but AMP_PPO does not.
        _amp_ppo_params = set(AMP_PPO.__init__.__code__.co_varnames)
        alg_kwargs = {k: v for k, v in self.alg_cfg.items() if k in _amp_ppo_params}
        alg_kwargs.pop("class_name", None)

        self.alg: AMP_PPO = AMP_PPO(
            actor_critic=actor_critic,
            discriminator=discriminator,
            demo_fn=demo_fn,
            mode=im_cfg.mode,
            disc_lr=getattr(im_cfg, "disc_lr", None),
            amp_replay_buffer_size=getattr(im_cfg, "disc_buffer_size", 100_000),
            grad_penalty_coeff=getattr(im_cfg, "disc_grad_penalty", 10.0),
            device=self.device,
            **alg_kwargs,
        )

        self.num_steps_per_env: int = self.cfg["num_steps_per_env"]
        self.save_interval: int = self.cfg["save_interval"]
        self.empirical_normalization: bool = self.cfg.get("empirical_normalization", False)

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
        disc_obs = base_env.disc_obs.clone().to(self.device)
        demo_disc_obs: torch.Tensor | None = None
        if self.im_cfg.mode == "add":
            demo_disc_obs = base_env._disc_obs_demo.clone().to(self.device)

        self.train_mode()

        ep_infos = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        disc_rewbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_disc_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            mean_style_reward_log = 0.0
            mean_task_reward_log = 0.0

            # ----------------------------------------------------------
            # Rollout (mirrors amp-rsl-rl learn() inner loop)
            # ----------------------------------------------------------
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)

                    obs_raw, rewards, dones, infos = self.env.step(actions.to(self.env.device))

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
                    if self.im_cfg.mode == "add":
                        # ADD: discriminator takes diff = demo - agent
                        diff = next_demo_disc_obs - next_disc_obs
                        style_rewards = self.alg.discriminator.predict_reward(diff)
                    else:
                        # AMP: discriminator takes agent disc_obs directly
                        style_rewards = self.alg.discriminator.predict_reward(next_disc_obs)

                    mean_task_reward_log += rewards.mean().item()
                    mean_style_reward_log += style_rewards.mean().item()

                    # Mix task + style rewards (copy amp-rsl-rl 0.5/0.5 default
                    # but respect im_cfg weights if provided)
                    task_w = getattr(self.im_cfg, "task_reward_weight", 0.5)
                    disc_w = getattr(self.im_cfg, "disc_reward_weight", 0.5)
                    blended_rewards = task_w * rewards + disc_w * style_rewards

                    self.alg.process_env_step(blended_rewards, dones, infos)

                    # Insert into disc replay buffer
                    self.alg.process_disc_step(next_disc_obs, next_demo_disc_obs)

                    # Advance disc obs
                    disc_obs = next_disc_obs
                    if self.im_cfg.mode == "add":
                        demo_disc_obs = next_demo_disc_obs

                    # ---- Logging ----
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])

                        cur_reward_sum += blended_rewards
                        cur_disc_reward_sum += style_rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        disc_rewbuffer.extend(cur_disc_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        cur_disc_reward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                start = stop
                self.alg.compute_returns(critic_obs)

            mean_style_reward_log /= self.num_steps_per_env
            mean_task_reward_log /= self.num_steps_per_env

            # ----------------------------------------------------------
            # Combined PPO + discriminator update
            # ----------------------------------------------------------
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

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

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
            for key in locs["ep_infos"][0]:
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
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/amp_loss", locs["mean_amp_loss"], locs["it"])
        self.writer.add_scalar("Loss/grad_pen_loss", locs["mean_grad_pen_loss"], locs["it"])
        self.writer.add_scalar("Loss/policy_pred", locs["mean_policy_pred"], locs["it"])
        self.writer.add_scalar("Loss/expert_pred", locs["mean_expert_pred"], locs["it"])
        self.writer.add_scalar("Loss/accuracy_policy", locs["mean_accuracy_policy"], locs["it"])
        self.writer.add_scalar("Loss/accuracy_expert", locs["mean_accuracy_expert"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Loss/kl_divergence", locs["mean_kl_divergence"], locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learn_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_style_reward", locs["mean_style_reward_log"], locs["it"])
            self.writer.add_scalar("Train/mean_task_reward", locs["mean_task_reward_log"], locs["it"])
        if len(locs["disc_rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_disc_ep_reward", statistics.mean(locs["disc_rewbuffer"]), locs["it"])

        str_ = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s"""
                f""" (collection: {locs['collection_time']:.3f}s,"""
                f""" learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                f"""{'Grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                f"""{'Policy / expert pred:':>{pad}} {locs['mean_policy_pred']:.3f} / {locs['mean_expert_pred']:.3f}\n"""
                f"""{'Disc acc (pol / exp):':>{pad}} {locs['mean_accuracy_policy']:.3f} / {locs['mean_accuracy_expert']:.3f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean style reward:':>{pad}} {locs['mean_style_reward_log']:.4f}\n"""
                f"""{'Mean task reward:':>{pad}} {locs['mean_task_reward_log']:.4f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s"""
                f""" (collection: {locs['collection_time']:.3f}s,"""
                f""" learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                f"""{'Grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
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
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(
                loaded_dict["discriminator_state_dict"], strict=False
            )
        if self.empirical_normalization and "obs_norm_state_dict" in loaded_dict:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
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
        self.alg.discriminator.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        self.alg.discriminator.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path: str):
        self.git_status_repos.append(repo_file_path)
