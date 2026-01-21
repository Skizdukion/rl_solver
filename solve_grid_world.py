import gymnasium as gym
import caro_env
import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np
import torch.distributions as distributions
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    std = returns.std(unbiased=False)

    returns = (returns - returns.mean()) / (
        std + 1e-8
    )  # Add a tiny 'epsilon' to prevent division by zero

    return returns


def calculate_total_rewards_with_discount_factor(rewards, discount_factor):
    total_reward = 0
    for i in range(len(rewards)):
        total_reward += discount_factor**i * rewards[i]

    return total_reward


def calculate_advantages(returns, values):
    advantages = returns - values
    std = advantages.std(unbiased=False)
    advantages = (advantages - advantages.mean()) / (
        std + 1e-8
    )  # Add a tiny 'epsilon' to prevent division by zero
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
    policy_loss = -(sug_loss + entropy_bonus).sum()  # why not mean
    value_loss = f.smooth_l1_loss(returns, value_preds).sum()  # why not mean
    return policy_loss, value_loss


def init_training():
    states = []
    actions = []
    actions_log_probability = []
    values = []
    rewards = []
    terminated = False
    return (states, actions, actions_log_probability, values, rewards, terminated)


def forward_pass(env, agent, discount_factor):
    (
        states,
        actions,
        actions_log_probability,
        values,
        rewards,
        terminated,
    ) = init_training()

    observation, _ = env.reset()
    state = np.concatenate([observation["agent"], observation["target"]])  # Shape (4, )
    agent.train()

    while not terminated:
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(
            0
        )  # torch.Size([1, 4])
        states.append(state)

        action_pred, value_pred = agent(state)
        action_prob = f.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(
            action_prob
        )  # In here we could boost exploration

        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        observation, reward, terminated, _, _ = env.step(action.item())
        state = np.concatenate([observation["agent"], observation["target"]])

        actions.append(action)
        actions_log_probability.append(log_prob_action)

        values.append(value_pred)
        rewards.append(reward)

    states = torch.cat(states).to(device)
    actions = torch.cat(actions).to(device)
    actions_log_probability = torch.cat(actions_log_probability).to(device)
    values = torch.cat(values).squeeze(-1).to(device)
    returns = calculate_returns(rewards, discount_factor).to(device)
    advantages = calculate_advantages(returns, values).to(device)
    return states, actions, actions_log_probability, advantages, returns, rewards


def update_policy(
    agent,
    states,
    actions,
    actions_log_probability_old,
    advantages,
    returns,
    optimizer,
    ppo_steps,
    epsilon,
    entropy_coefficient,
):
    batch_size = 128
    total_policy_loss = 0
    total_value_loss = 0
    actions_log_probability_old = actions_log_probability_old.detach()
    actions = actions.detach()
    training_results_dataset = TensorDataset(
        states, actions, actions_log_probability_old, advantages, returns
    )
    batch_dataset = DataLoader(
        training_results_dataset, batch_size=batch_size, shuffle=False
    )

    for _ in range(ppo_steps):
        for batch_idx, (
            states,
            actions,
            actions_log_probability_old,
            advantages,
            returns,
        ) in enumerate(batch_dataset):
            action_pred, value_pred = agent(states)
            value_pred = value_pred.squeeze(-1)
            action_prob = f.softmax(action_pred, dim=-1)

            probability_distribution_new = distributions.Categorical(
                action_prob
            )  # efficient learning
            entropy = probability_distribution_new.entropy()

            actions_log_probability_new = probability_distribution_new.log_prob(actions)

            surrogate_loss = calculate_surrogate_loss(
                actions_log_probability_old,
                actions_log_probability_new,
                epsilon,
                advantages,
            )

            policy_loss, value_loss = calculate_loss(
                surrogate_loss, entropy, entropy_coefficient, returns, value_pred
            )

            optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()

            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


def evaluate(env, agent, discount_factor, early_stop=500):
    agent.eval()
    terminated = False
    rewards = []

    # state = env.reset()
    observation, _ = env.reset()
    state = np.concatenate([observation["agent"], observation["target"]])
    timestep = 0
    while not terminated and timestep < early_stop:
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_pred, _ = agent(state)
            action_prob = f.softmax(action_pred, dim=-1)
        action = torch.argmax(action_prob, dim=-1)
        observation, reward, terminated, _, _ = env.step(action.item())
        state = np.concatenate([observation["agent"], observation["target"]])
        rewards.append(reward)
        timestep += 1

    return calculate_total_rewards_with_discount_factor(rewards, discount_factor)


def run_ppo():
    MAX_EPISODES = 2000
    DISCOUNT_FACTOR = 0.99
    REWARD_THRESHOLD = 475
    PRINT_INTERVAL = 10
    PPO_STEPS = 8
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFFICIENT = 0.01
    HIDDEN_DIMENSIONS = 64
    DROPOUT = 0.2
    LEARNING_RATE = 1e-4

    train_rewards = []
    test_rewards = []
    policy_losses = []
    value_losses = []

    env = gym.make("caro_env/GridWorld-v0")
    agent = create_agent(HIDDEN_DIMENSIONS, DROPOUT, env)
    agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    for episode in trange(1, MAX_EPISODES + 1):
        states, actions, actions_log_probability, advantages, returns, rewards = (
            forward_pass(env, agent, DISCOUNT_FACTOR)
        )

        policy_loss, value_loss = update_policy(
            agent,
            states,
            actions,
            actions_log_probability,
            advantages,
            returns,
            optimizer,
            PPO_STEPS,
            EPSILON,
            ENTROPY_COEFFICIENT,
        )

        test_reward = evaluate(env, agent, DISCOUNT_FACTOR)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        train_rewards.append(
            calculate_total_rewards_with_discount_factor(rewards, DISCOUNT_FACTOR)
        )
        test_rewards.append(test_reward)
        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        mean_abs_policy_loss = np.mean(np.abs(policy_losses[-N_TRIALS:]))
        mean_abs_value_loss = np.mean(np.abs(value_losses[-N_TRIALS:]))

        if episode % PRINT_INTERVAL == 0:
            print(
                f"Episode: {episode:3} | \
                  Mean Train Rewards: {mean_train_rewards:3.1f} \
                  | Mean Test Rewards: {mean_test_rewards:3.1f} \
                  | Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} \
                  | Mean Abs Value Loss: {mean_abs_value_loss:2.2f}"
            )

        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f"Reached reward threshold in {episode} episodes")
            break

run_ppo()
