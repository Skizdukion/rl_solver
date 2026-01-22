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

from network.grid_world.backbone import ActorCritic, BackboneNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_agent(hidden_dims, dropout, env: gym.vector.VectorEnv):
    input_features = (
        env.single_observation_space["agent"].shape[0]
        + env.single_observation_space["target"].shape[0]
    )  # only position of agent and target is needed

    actor_output = env.single_action_space.n
    critic_output = 1
    actor = BackboneNetwork(input_features, hidden_dims, actor_output, dropout)
    critic = BackboneNetwork(
        input_features, hidden_dims, critic_output, dropout)

    agent = ActorCritic(actor, critic)

    return agent


def calculate_returns_with_bootstrapping(rewards, dones, last_value, discount_factor):
    returns = torch.zeros_like(rewards)
    cumulative_reward = last_value  # Start with the predicted future value

    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + \
            (cumulative_reward * discount_factor * (1 - dones[i]))
        returns[i] = cumulative_reward

    # Normalization follows...
    return returns


def calculate_total_rewards_with_discount_factor(rewards, discount_factor):
    total_reward = 0
    for i in range(len(rewards)):
        total_reward += discount_factor**i * rewards[i]

    return total_reward


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
        torch.clamp(policy_ratio, min=1 - espilon,
                    max=1 + espilon) * advantages
    )

    sug_logss = torch.min(sug_loss_1, sug_loss_2)

    return sug_logss


def calculate_loss(sug_loss, entropy, entropy_coefficient, returns, value_preds):
    entropy_bonus = entropy_coefficient * entropy
    policy_loss = -(sug_loss + entropy_bonus).mean()  # why not mean
    value_loss = f.smooth_l1_loss(returns, value_preds).mean()  # why not mean
    return policy_loss, value_loss


def forward_pass(envs, agent, num_rollout_steps, discount_factor):
    states = []
    actions = []
    actions_log_probability = []
    values = []
    rewards = []
    dones = []

    observations, _ = envs.reset()
    state = np.concatenate(
        [observations["agent"], observations["target"]], axis=-1)  # Shape (4, )
    agent.train()

    for _ in range(num_rollout_steps):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        states.append(state)

        action_pred, value_pred = agent(state)
        dist = distributions.Categorical(logits=action_pred)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        observations, reward, terminated, truncated, _ = envs.step(
            action.cpu().numpy())

        done_mask = terminated | truncated

        state = np.concatenate(
            [observations["agent"], observations["target"]], axis=-1)

        actions.append(action)
        actions_log_probability.append(log_prob_action)

        values.append(value_pred.squeeze(-1))
        rewards.append(torch.tensor(
            reward, dtype=torch.float32, device=device))
        dones.append(torch.tensor(
            done_mask, dtype=torch.float32, device=device))

    with torch.no_grad():
        # Get the value of the 'next_state' to estimate future rewards
        last_state_tensor = torch.tensor(
            state, dtype=torch.float32, device=device)
        _, last_value = agent(last_state_tensor)
        last_value = last_value.squeeze(-1)  # (num_envs,)

    states = torch.stack(states)
    actions = torch.stack(actions)
    actions_log_probability = torch.stack(actions_log_probability)
    values = torch.stack(values)
    rewards_tensor = torch.stack(rewards)
    dones_tensor = torch.stack(dones)

    returns = calculate_returns_with_bootstrapping(
        rewards_tensor, dones_tensor, last_value, discount_factor)
    advantages = calculate_advantages(returns, values).to(device)

    return states, actions, actions_log_probability, advantages, returns, rewards_tensor


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

    states = states.view(-1, states.size(-1))
    actions = actions.view(-1)
    actions_log_probability_old = actions_log_probability_old.view(-1).detach()
    advantages = advantages.view(-1)
    returns = returns.view(-1)

    batch_size = 128
    total_policy_loss = 0
    total_value_loss = 0
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

            actions_log_probability_new = probability_distribution_new.log_prob(
                actions)

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


def evaluate(single_env, agent, discount_factor, early_stop=500):
    agent.eval()
    terminated = False
    rewards = []

    observation, _ = single_env.reset()
    state = np.concatenate(
        [observation["agent"], observation["target"]], axis=-1)
    timestep = 0
    while not terminated and timestep < early_stop:
        state = torch.tensor(state, dtype=torch.float32,
                             device=device).unsqueeze(0)
        with torch.no_grad():
            action_pred, _ = agent(state)
            action_prob = f.softmax(action_pred, dim=-1)
        action = torch.argmax(action_prob, dim=-1)
        observation, reward, terminated, _, _ = single_env.step(action.item())
        state = np.concatenate([observation["agent"], observation["target"]])
        rewards.append(reward)
        timestep += 1

    return calculate_total_rewards_with_discount_factor(rewards, discount_factor)


def save_checkpoint(episode, agent, optimizer, train_rewards, path="checkpoints_grid_world"):
    if not os.path.exists(path):
        os.makedirs(path)

    checkpoint = {
        'episode': episode,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_rewards': train_rewards,
    }
    torch.save(checkpoint, f"{path}/ppo_checkpoint_{episode}.pt")
    # Also save a 'latest' version for easy resuming
    torch.save(checkpoint, f"{path}/ppo_latest.pt")

def run_ppo():
    MAX_EPISODES = 500
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
    MAX_TIME_STEPS = 500
    SAVE_INTERVAL = 50

    train_rewards = []
    test_rewards = []
    policy_losses = []
    value_losses = []

    envs = gym.make_vec("caro_env/GridWorld-v0", num_envs=4,
                        vectorization_mode="async")

    single_env = gym.make("caro_env/GridWorld-v0")

    agent = create_agent(HIDDEN_DIMENSIONS, DROPOUT, envs)
    agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    for episode in trange(1, MAX_EPISODES + 1):
        states, actions, actions_log_probability, advantages, returns, rewards = (
            forward_pass(envs, agent, MAX_TIME_STEPS, DISCOUNT_FACTOR)
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

        test_reward = evaluate(single_env, agent, DISCOUNT_FACTOR)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        train_rewards.append(
            calculate_total_rewards_with_discount_factor(
                rewards, DISCOUNT_FACTOR)
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

        if episode % SAVE_INTERVAL == 0:
            save_checkpoint(episode, agent, optimizer, train_rewards)

        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f"Reached reward threshold in {episode} episodes")
            break


run_ppo()
