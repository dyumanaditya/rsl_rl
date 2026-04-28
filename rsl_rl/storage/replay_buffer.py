# Copyright (c) 2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from amp-rsl-rl (ami-iit/amp-rsl-rl) – simplified to store single
# observations (not state/next-state pairs) so it works with the existing
# G1ImitationEnv disc_obs format.

import torch
from typing import Generator, Union


class ReplayBuffer:
    """Fixed-size circular buffer that stores single observation vectors.

    Used by AMP_PPO to hold policy-generated disc_obs samples that are
    replayed alongside the current rollout when training the discriminator.
    """

    def __init__(
        self,
        obs_dim: int,
        buffer_size: int,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.obs = torch.zeros(buffer_size, obs_dim, dtype=torch.float32, device=self.device)
        self.step = 0
        self.num_samples = 0

    def insert(self, obs: torch.Tensor) -> None:
        """Add a batch of observations to the buffer (circular write)."""
        obs = obs.detach().to(self.device)
        B = obs.shape[0]
        end = self.step + B

        if end <= self.buffer_size:
            self.obs[self.step:end] = obs
        else:
            first_part = self.buffer_size - self.step
            self.obs[self.step:] = obs[:first_part]
            remainder = B - first_part
            self.obs[:remainder] = obs[first_part:]

        self.step = end % self.buffer_size
        self.num_samples = min(self.buffer_size, self.num_samples + B)

    def feed_forward_generator(
        self,
        num_mini_batch: int,
        mini_batch_size: int,
        allow_replacement: bool = True,
    ) -> Generator[torch.Tensor, None, None]:
        """Yield ``num_mini_batch`` batches of size ``mini_batch_size``."""
        total = num_mini_batch * mini_batch_size

        if total > self.num_samples:
            if not allow_replacement:
                raise ValueError(
                    f"Not enough samples in buffer: requested {total}, have {self.num_samples}"
                )
            cycles = (total + self.num_samples - 1) // self.num_samples
            big_perm = torch.randperm(self.num_samples * cycles, device=self.device)
            indices = big_perm[:total] % self.num_samples
        else:
            indices = torch.randperm(self.num_samples, device=self.device)[:total]

        for i in range(num_mini_batch):
            batch_idx = indices[i * mini_batch_size:(i + 1) * mini_batch_size]
            yield self.obs[batch_idx]

    def __len__(self) -> int:
        return self.num_samples
