import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from caro_env.envs.caro import Player
import copy

from utils.gomuko import from_boards_to_state


class HandleReset(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        pass

    def step(self, action):
        pass
