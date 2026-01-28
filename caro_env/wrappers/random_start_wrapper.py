import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from caro_env.envs.caro import Player
import copy

from utils.gomuko import from_boards_to_state


class RandomStartWrapper(gym.Wrapper):
    CURRENT_OP = None
    DEVICE = None

    def __init__(self, env):
        super().__init__(env)

    def _get_opp_action(self, obs, info):
        # Convert single numpy obs to tensor [1, C, H, W]
        # Invert board logic: My stones (-1) -> 1, Enemy stones (1) -> -1
        board = np.expand_dims(obs["board"] * -1, axis=0)
        state = from_boards_to_state(board)
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=RandomStartWrapper.DEVICE
        )

        with torch.no_grad():
            action_logits = RandomStartWrapper.CURRENT_OP.actor(state_tensor)

            # Apply Mask
            mask = torch.tensor(
                info["action_mask"], device=RandomStartWrapper.DEVICE
            ).unsqueeze(0)
            masked_logits = action_logits + (1 - mask) * -1e9

            action = torch.argmax(masked_logits, dim=-1).item()

        return action

    def reset(self, **kwargs):
        options = kwargs.get("options", {}) or {}
        kwargs["options"] = options

        obs, info = self.env.reset(**kwargs)

        if options.get("opp_starts", np.random.random() > 0.5):
            action = self._get_opp_action(obs, info)
            # Capture ALL 5 values
            obs, _, _, _, info = self.env.step(action)

        return obs, info

    def step(self, action):
        # 1. Agent's move
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            return obs, float(reward), terminated, truncated, info

        opp_action = self._get_opp_action(obs, info)
        obs, opp_reward, terminated, truncated, info = self.env.step(opp_action)

        return obs, float(-opp_reward), terminated, truncated, info
