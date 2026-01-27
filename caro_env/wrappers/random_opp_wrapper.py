import gymnasium as gym
import numpy as np

from caro_env.envs.caro import Player


class RandomOpponentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        options = kwargs.get("options", {})

        if options is None:
            options = {}

        if "opp_starts" not in options:
            options["opp_starts"] = np.random.random() > 0.5

        # 3. Put it back into kwargs
        kwargs["options"] = options

        obs, info = self.env.reset(**kwargs)

        valid_indices = np.where(info["action_mask"] == 1)[0]
        opp_action = np.random.choice(valid_indices)
        obs, _, _, _, info = self.env.step(opp_action)

        return obs, info

    def step(self, action):
        # 1. Execute Agent's move
        obs, reward, terminated, truncated, info = self.env.step(action)

        # If Agent won or game ended, return immediately
        if terminated or truncated:
            # Reward is already 1 if agent won, 0 if draw
            return obs, float(reward), terminated, truncated, info

        # 2. Execute Random Opponent's move
        # We use the action_mask provided by the environment for the opponent
        opp_mask = info["action_mask"]
        valid_indices = np.where(opp_mask == 1)[0]

        if len(valid_indices) > 0:
            opp_action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, info = self.env.step(opp_action)

        return obs, -reward, terminated, truncated, info
