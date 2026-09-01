# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl.utils import store_code_state
from utils.contact_force_logging import pop_contact_force_stats
from utils.metric_logging import episode_tags, log_scalar_aliases


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # resolve dimension of rnd gated state
        if self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for they key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_state"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.dt

        # if using symmetry then pass the environment config object
        if self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # init algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        # Drop keys this algorithm does not take, as AMPOnPolicyRunner already
        # does. The shared `ppo.algorithm` config block carries AMP/ADD-only
        # settings (separate actor/critic optimizer rates and clips, MimicKit's
        # advantage clip, the return-scaled value clip); without this filter a
        # plain non-imitation `alg=ppo` run dies on an unexpected keyword.
        _alg_params = set(alg_class.__init__.__code__.co_varnames)
        alg_kwargs = {k: v for k, v in self.alg_cfg.items() if k in _alg_params}
        self.alg: PPO = alg_class(actor_critic, device=self.device, **alg_kwargs)

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        # Optional discriminator (AMPDiscriminator / ADDDiscriminator).
        # Set before learn() to interleave disc training with each PPO update.
        self.discriminator = None
        # Pre-allocated buffers for per-step disc_obs collection (set in learn()).
        self._disc_obs_buf = None
        self._disc_demo_obs_buf = None

        # Optional live renderer (enabled when env.cfg.viz.render is True).
        self.renderer = self._make_renderer()

    def _make_renderer(self):
        try:
            viz_cfg = getattr(getattr(self.env, "cfg", None), "viz", None)
            if viz_cfg is None or not getattr(viz_cfg, "render", False):
                return None
            base_env = getattr(self.env, "base_env", self.env)
            from utils.render import Render
            return Render(self.env.cfg, base_env.model)
        except Exception as exc:
            print(f"[OnPolicyRunner] Warning: could not initialise renderer: {exc}")
            return None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)
        obs = self.obs_normalizer(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # disc reward per-episode buffer (accumulated after rollout from stored disc_r)
        disc_rewbuffer = deque(maxlen=100)
        cur_disc_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Allocate per-step disc_obs collection buffers (pre-allocated to avoid inference-tensor issues)
        if self.discriminator is not None:
            base_env = getattr(self.env, "base_env", self.env)
            disc_obs_size = base_env.disc_obs_size
            im_cfg = base_env._im_cfg
            self._disc_obs_buf = torch.zeros(
                self.num_steps_per_env, self.env.num_envs, disc_obs_size, device=self.device
            )
            self._disc_demo_obs_buf = (
                torch.zeros_like(self._disc_obs_buf) if im_cfg.mode == "add" else None
            )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # plant curriculum: anneal the reflected rotor inertia (no-op unless
            # sim.armature_curriculum is set).
            if hasattr(self.env, "maybe_update_armature"):
                self.env.maybe_update_armature(it)
            # Rollout
            with torch.inference_mode():
                for step in range(self.num_steps_per_env):
                    # Sample actions from policy
                    actions = self.alg.act(obs, critic_obs)
                    # Step environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))

                    # Render
                    if self.renderer is not None:
                        base_env = getattr(self.env, "base_env", self.env)
                        self.renderer.render(base_env.body_q, base_env.body_qd)

                    # Move to the agent device
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # Normalize observations
                    obs = self.obs_normalizer(obs)
                    # Extract critic observations and normalize
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"].to(self.device))
                    else:
                        critic_obs = obs

                    # Collect disc_obs into pre-allocated buffers.
                    # copy_() onto a regular (non-inference) destination tensor preserves its type,
                    # so _disc_obs_buf remains usable outside inference_mode for disc training.
                    if self._disc_obs_buf is not None and "disc_obs" in infos:
                        self._disc_obs_buf[step].copy_(infos["disc_obs"])
                        if self._disc_demo_obs_buf is not None and "disc_obs_demo" in infos:
                            self._disc_demo_obs_buf[step].copy_(infos["disc_obs_demo"])

                    # Intrinsic rewards (extracted here only for logging)!
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # Process env step and store in buffer (pure task rewards — disc added below)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

            # Inject disc rewards into the rollout storage and compute value-function returns.
            # Done outside inference_mode so _disc_obs_buf tensors are safe for autograd in disc training.
            start = stop
            mean_disc_reward = 0.0
            with torch.no_grad():
                if self.discriminator is not None and self._disc_obs_buf is not None:
                    base_env = getattr(self.env, "base_env", self.env)
                    im_cfg = base_env._im_cfg
                    disc_obs_flat = self._disc_obs_buf.flatten(0, 1)   # (T*N, obs_size)

                    if im_cfg.mode == "add" and self._disc_demo_obs_buf is not None:
                        disc_r = self.discriminator.compute_rewards(
                            disc_obs_flat, self._disc_demo_obs_buf.flatten(0, 1)
                        )
                    else:
                        disc_r = self.discriminator.compute_rewards(disc_obs_flat)

                    mean_disc_reward = disc_r.mean().item()
                    disc_r_storage = disc_r.view(self.num_steps_per_env, self.env.num_envs, 1)
                    # storage.rewards holds pure task rewards; blend in disc signal now
                    self.alg.storage.rewards.mul_(im_cfg.task_reward_weight).add_(
                        im_cfg.disc_reward_weight * disc_r_storage
                    )

                    # Accumulate disc reward per-episode for logging
                    if self.log_dir is not None:
                        stored_dones = self.alg.storage.dones  # (T, N, 1)
                        for step in range(self.num_steps_per_env):
                            cur_disc_reward_sum += disc_r_storage[step].squeeze(-1)
                            new_ids = (stored_dones[step].squeeze(-1) > 0).nonzero(as_tuple=False)
                            disc_rewbuffer.extend(cur_disc_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_disc_reward_sum[new_ids] = 0

                # Learning step
                self.alg.compute_returns(critic_obs)

            # Update policy
            # Note: we keep arguments here since locals() loads them
            mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Train discriminator on full rollout disc_obs (all T*N samples)
            disc_info = self._update_discriminator() if self.discriminator is not None else {}

            # Logging info and save checkpoint
            if self.log_dir is not None:
                # Log information — disc_info is in locals() automatically
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()

            # Save code state
            if it == start_iter and self.log_dir is not None:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _update_discriminator(self) -> dict:
        """Train discriminator (AMP / ADD) on the full rollout's disc_obs.

        Uses the pre-allocated _disc_obs_buf (shape: num_steps × num_envs × obs_size)
        filled during the rollout, providing T*N samples per update instead of just N.

        disc_batch_size in im_cfg is per-env (like MimicKit). Scale by num_envs so
        the number of gradient steps matches MimicKit's ~ceil(T / disc_batch_size) * epochs.
        """
        base_env = getattr(self.env, "base_env", self.env)
        im_cfg = base_env._im_cfg

        # Flatten to (T*N, obs_size) — these are regular tensors, safe for autograd
        agent_obs = self._disc_obs_buf.flatten(0, 1)

        if im_cfg.mode == "add" and self._disc_demo_obs_buf is not None:
            demo_obs = self._disc_demo_obs_buf.flatten(0, 1)
        else:
            demo_obs = base_env.fetch_disc_obs_demo(agent_obs.shape[0])

        # Scale batch size by num_envs to match MimicKit's per-env convention.
        effective_batch = max(1, im_cfg.disc_batch_size * self.env.num_envs)
        return self.discriminator.update(agent_obs, demo_obs, batch_size=effective_batch)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                log_scalar_aliases(self.writer, episode_tags(key), value, locs["it"])
                if "/" in key:
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/policy", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/entropy", locs["mean_entropy"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        if self.alg.rnd:
            self.writer.add_scalar("Loss/rnd", locs["mean_rnd_loss"], locs["it"])
        if self.alg.symmetry:
            self.writer.add_scalar("Loss/symmetry", locs["mean_symmetry_loss"], locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Per-leg contact forces over this iteration's rollout. Flushes the
        # substep records the sim accumulated; see utils/contact_force_logging.py
        # for how to read them when tuning cfg.sim.bundle_contact_force_thresh.
        for tag, value in pop_contact_force_stats(self.env).items():
            self.writer.add_scalar(tag, value, locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            mean_reward = statistics.mean(locs["rewbuffer"])
            mean_length = statistics.mean(locs["lenbuffer"])
            # separate logging for intrinsic and extrinsic rewards
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # everything else
            self.writer.add_scalar("Train/mean_reward", mean_reward, locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", mean_length, locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", mean_reward, self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", mean_length, self.tot_time
                )
        # disc reward per-episode (primary training signal for AMP/ADD)
        if len(locs.get("disc_rewbuffer", [])) > 0:
            self.writer.add_scalar("Train/mean_disc_reward", statistics.mean(locs["disc_rewbuffer"]), locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            )

            # -- For symmetry
            if self.alg.symmetry:
                log_string += f"""{'Symmetry loss:':>{pad}} {locs['mean_symmetry_loss']:.4f}\n"""

            log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""

            # -- For RND
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )

            log_string += f"""{'Mean total reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            )
            # -- For symmetry
            if self.alg.symmetry:
                log_string += f"""{'Symmetry loss:':>{pad}} {locs['mean_symmetry_loss']:.4f}\n"""

            log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""

            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string

        # -- Discriminator (AMP / ADD)
        disc_info = locs.get("disc_info", {})
        mean_disc_reward = locs.get("mean_disc_reward", 0.0)
        if disc_info:
            self.writer.add_scalar("Disc/loss", disc_info["disc_loss"], locs["it"])
            self.writer.add_scalar("Disc/agent_acc", disc_info["disc_agent_acc"], locs["it"])
            self.writer.add_scalar("Disc/demo_acc", disc_info["disc_demo_acc"], locs["it"])
            self.writer.add_scalar("Disc/mean_reward", mean_disc_reward, locs["it"])
            log_string += (
                f"""{'Disc loss:':>{pad}} {disc_info['disc_loss']:.4f}\n"""
                f"""{'Disc agent/demo acc:':>{pad}} {disc_info['disc_agent_acc']:.2f} / {disc_info['disc_demo_acc']:.2f}\n"""
                f"""{'Disc mean reward:':>{pad}} {mean_disc_reward:.4f}\n"""
            )
        if len(locs.get("disc_rewbuffer", [])) > 0:
            log_string += f"""{'Mean disc ep reward:':>{pad}} {statistics.mean(locs['disc_rewbuffer']):.2f}\n"""

        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save PPO model
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

        # Save discriminator alongside the policy checkpoint
        if self.discriminator is not None:
            self.discriminator.save(path.replace(".pt", "_disc.pt"))

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load PPO model
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        # -- Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        # -- Load optimizer if used
        if load_optimizer:
            # -- PPO
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND optimizer if used
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- Load current learning iteration
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        # -- PPO
        self.alg.actor_critic.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.actor_critic.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
