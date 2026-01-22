import os
import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
import caro_env


def run_random_simluation(env: gym.Env):
    _, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # 1. Render the frame for human viewing
        env.render()
        valid_actions = np.flatnonzero(info["action_mask"])
        action = env.np_random.choice(valid_actions)
        _, _, terminated, truncated, info = env.step(action)


env = gym.make("caro_env/Gomoku-v0", render_mode="rgb_array")
env = RecordVideo(env, video_folder="recordings", episode_trigger=lambda x: x == 0)

run_random_simluation(env)
env.close()
