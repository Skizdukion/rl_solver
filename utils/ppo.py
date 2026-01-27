import gymnasium as gym
import caro_env
import torch.nn.functional as f
import torch
import numpy as np
import torch.distributions as distributions
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch
from tqdm import trange
import os


def calculate_returns_with_bootstrapping(rewards, dones, last_value, discount_factor):
    returns = torch.zeros_like(rewards)
    cumulative_reward = last_value  # Start with the predicted future value

    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + (
            cumulative_reward * discount_factor * (1 - dones[i])
        )
        returns[i] = cumulative_reward

    # Normalization follows...
    return returns


def calculate_advantages(returns, values):
    # returns: (steps, num_envs), values: (steps, num_envs)
    advantages = returns - values

    # Standardize over the entire rollout buffer
    std = advantages.std(unbiased=False)
    advantages = (advantages - advantages.mean()) / (std + 1e-8)

    return advantages


def calculate_surrogate_loss(
    actions_log_prob_old, actions_log_prob_new, espilon, advantages
):  # Full sequence of action -> [B, T]
    advantages = advantages.detach()  # Detach from critic network

    policy_ratio = (actions_log_prob_new - actions_log_prob_old).exp()

    sug_loss_1 = policy_ratio * advantages
    sug_loss_2 = (
        torch.clamp(policy_ratio, min=1 - espilon, max=1 + espilon) * advantages
    )

    sug_logss = torch.min(sug_loss_1, sug_loss_2)

    return sug_logss


def calculate_loss(sug_loss, entropy, entropy_coefficient, returns, value_preds):
    entropy_bonus = entropy_coefficient * entropy
    policy_loss = -(sug_loss + entropy_bonus).mean()
    value_loss = f.smooth_l1_loss(returns, value_preds).mean()
    return policy_loss, value_loss


def calculate_total_rewards_with_discount_factor(rewards, discount_factor):
    total_reward = 0
    for i in range(len(rewards)):
        total_reward += discount_factor**i * rewards[i]

    return total_reward


def save_checkpoint(episode, agent, optimizer, train_rewards, path):
    if not os.path.exists(path):
        os.makedirs(path)

    checkpoint = {
        "episode": episode,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_rewards": train_rewards,
    }
    torch.save(checkpoint, f"{path}/ppo_checkpoint_{episode}.pt")
    # Also save a 'latest' version for easy resuming
    torch.save(checkpoint, f"{path}/ppo_latest.pt")


def load_checkpoint(agent, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        agent.load_state_dict(checkpoint["model_state_dict"])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint["episode"]
        print(f"Resuming from episode {start_episode}")
        return start_episode
    return 0
