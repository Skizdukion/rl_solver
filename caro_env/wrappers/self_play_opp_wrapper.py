import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from caro_env.envs.caro import Player
import copy

from utils.gomuko import from_boards_to_state


class SelfPlayOpponentWrapper(gym.Wrapper):
    DEVICE = None
    OPPS = []

    def __init__(self, env):
        super().__init__(env)
        self.current_op = None

    @staticmethod
    def add_opp(op_agent: nn.Module):
        # Create a frozen copy
        op = copy.deepcopy(op_agent)
        op.to(SelfPlayOpponentWrapper.DEVICE)
        op.eval()
        for param in op.parameters():
            param.requires_grad = False

        SelfPlayOpponentWrapper.OPPS.append(op)
        if len(SelfPlayOpponentWrapper.OPPS) > 5:
            del SelfPlayOpponentWrapper.OPPS[1]

    def _get_opp_action(self, obs, info):
        # Convert single numpy obs to tensor [1, C, H, W]
        board = np.expand_dims(obs["board"], axis=0)
        state = from_boards_to_state(board)
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=SelfPlayOpponentWrapper.DEVICE
        )

        with torch.no_grad():
            action_logits = self.current_op.actor(state_tensor)

            # Apply Mask
            mask = torch.tensor(
                info["action_mask"], device=SelfPlayOpponentWrapper.DEVICE
            ).unsqueeze(0)
            masked_logits = action_logits + (1 - mask) * -1e9

            action = torch.argmax(masked_logits, dim=-1).item()
        return action

    def select_random_op(self):
        if len(SelfPlayOpponentWrapper.OPPS) == 0:
            raise Exception("No opps add yet")

        if np.random.random() < 0.5:
            self.current_op = SelfPlayOpponentWrapper.OPPS[
                len(SelfPlayOpponentWrapper.OPPS) - 1
            ]
        else:
            self.current_op = np.random.choice(SelfPlayOpponentWrapper.OPPS)

    def reset(self, **kwargs):
        options = kwargs.get("options", {}) or {}
        kwargs["options"] = options

        obs, info = self.env.reset(**kwargs)
        self.select_random_op()

        if options.get("opp_starts", np.random.random() > 0.5):
            action = self._get_opp_action(obs, info)
            # Capture ALL 5 values
            obs, reward, terminated, truncated, info = self.env.step(action)

            # # CRITICAL: If the opponent won on the first move,
            # if terminated or truncated:
            #     return self.reset(**kwargs)
            
        return obs, info

    def step(self, action):
        # 1. Agent's move
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            return obs, float(reward), terminated, truncated, info

        opp_action = self._get_opp_action(obs, info)
        obs, opp_reward, terminated, truncated, info = self.env.step(opp_action)

        if obs is None or reward is None or terminated is None or info is None:
            print(
                f"CRITICAL: Found None! Obs: {type(obs)}, Rew: {type(reward)}, Term: {type(terminated)}, Info: {type(info)}"
            )

        return obs, float(-opp_reward), terminated, truncated, info
