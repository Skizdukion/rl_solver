import gymnasium as gym
import caro_env
import torch.nn as nn
import torch.nn.functional as f
import torch

env = gym.make("caro_env/GridWorld-v0")


class BackboneNetwork(nn.Module):
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = nn.Linear(hidden_dimensions, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred


def create_agent(hidden_dims, dropout, env: gym.Env):
    input_features = (
        env.observation_space["agent"].shape[0]
        + env.observation_space["target"].shape[0]
    )  # only position of agent and target is needed
    actor_output = env.action_space.n
    critic_output = 1
    actor = BackboneNetwork(input_features, hidden_dims, actor_output, dropout)
    critic = BackboneNetwork(input_features, hidden_dims, critic_output, dropout)

    agent = ActorCritic(actor, critic)

    return agent


def calculate_returns(rewards, discount_factor):
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)

    returns = torch.tensor(returns)

    returns = (returns - returns.mean()) / (
        returns.std() + 1e-8
    )  # Add a tiny 'epsilon' to prevent division by zero

    return returns


def calculate_advantages(returns, values):
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-8
    )  # Add a tiny 'epsilon' to prevent division by zero
    return advantages


def calculate_surrogate_loss(
    actions_log_prob_old, actions_log_prob_new, espilon, advantages
):  # Full sequence of action -> [B, T]
    advantages = advantages.detech()  # Detach from critic network

    policy_ratio = (actions_log_prob_new - actions_log_prob_old).exp()

    sug_loss_1 = policy_ratio * advantages
    sug_loss_2 = (
        torch.clamp(policy_ratio, min=1 - espilon, max=1 + espilon) * advantages
    )

    sug_logss = torch.min(sug_loss_1, sug_loss_2)

    return sug_logss


def calculate_loss(sug_loss, entropy, entropy_coefficient, returns, value_preds):
    entropy_bonus = entropy_coefficient * entropy
    policy_loss = -(sug_loss + entropy_bonus).sum()  # why not mean
    value_loss = f.smooth_l1_loss(returns, value_preds).sum()  # why not mean
    return policy_loss, value_loss


def init_training():
    states = []
    actions = []
    actions_log_probability = []
    values = []
    rewards = []
    done = False
    episode_reward = 0
    return (
        states,
        actions,
        actions_log_probability,
        values,
        rewards,
        done,
        episode_reward,
    )


def forward_pass(env, agent, optimizer, discount_factor):
    states, actions, actions_log_probability, values, rewards, done, episode_reward = (
        init_training()
    )

    state = env.reset()
    agent.train()

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        states.append(state)
