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
from caro_env.wrappers.random_opp_wrapper import RandomOpponentWrapper
from caro_env.wrappers.random_start_wrapper import RandomStartWrapper
from caro_env.wrappers.self_play_opp_wrapper import SelfPlayOpponentWrapper
from network.caro.backbone import ActorCritic, BackboneNetwork
from utils.gomuko import calculate_segmented_returns_time_major, from_boards_to_state
from utils.ppo import (
    calculate_advantages,
    calculate_loss,
    calculate_surrogate_loss,
    calculate_total_rewards_with_discount_factor,
    load_agent_weights,
    load_optimizer_state,
    save_checkpoint,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_agent(env: gym.vector.VectorEnv):
    actor_output = env.single_action_space.n
    actor = BackboneNetwork(
        in_channels=3,
        hidden_channels=64,
        out_features=actor_output,
        dropout_prob=0.1,
        board_size=env.single_observation_space["board"].shape[0],
    )
    critic = BackboneNetwork(
        in_channels=3,
        hidden_channels=64,
        out_features=1,
        dropout_prob=0.1,
        board_size=env.single_observation_space["board"].shape[0],
    )

    agent = ActorCritic(actor, critic)

    return agent


def forward_pass(
    envs, agent, num_rollout_steps, current_opp_model, temperature, num_envs
):
    states = []
    actions = []
    actions_log_probability = []
    values = []
    rewards = []
    dones = []

    observation, info = envs.reset()
    state = from_boards_to_state(observation["board"])
    agent.train()

    for i in range(num_rollout_steps):
        # AGENT turn
        state = torch.tensor(
            state, dtype=torch.float32, device=device
        )  # Batch, 3, board_size, board_size

        states.append(state)
        action_pred, value_pred = agent(state)

        # 2. Mask the logits
        mask = torch.tensor(info["action_mask"], device=device)
        masked_action_pred = (action_pred / temperature) + (1 - mask) * -1e9

        dist = distributions.Categorical(logits=masked_action_pred)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        observation, reward, terminations, truncations, info = envs.step(
            action.cpu().numpy()
        )

        active_envs = ~(terminations | truncations)

        opp_state = (
            from_boards_to_state(observation["board"]) * -1
        )

        with torch.no_grad():
            opp_logits = current_opp_model.actor(opp_state)

            opp_mask = torch.tensor(info["action_mask"][active_envs])
            masked_opp_logits = opp_logits + (1 - opp_mask) * -1e9
            opp_actions = torch.argmax(masked_opp_logits, dim=-1)

        full_opp_actions = np.zeros(num_envs, dtype=int)
        full_opp_actions[active_envs] = opp_actions.cpu().numpy()

        obs, opp_rewards, terminations, truncations, infos = envs.step(full_opp_actions)

        state = from_boards_to_state(obs["board"])

        rewards = rewards - opp_rewards

        actions.append(action)
        actions_log_probability.append(log_prob_action)
        values.append(value_pred.squeeze(-1))
        # dones.append(torch.tensor(done_mask, dtype=torch.float32, device=device))
        state = from_boards_to_state(observation["board"])
        rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))

    with torch.no_grad():
        # Get the value of the 'next_state' to estimate future rewards
        last_state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        _, last_value = agent(last_state_tensor)
        last_value = last_value.squeeze(-1)  # (num_envs,)

    states = torch.stack(states)
    actions = torch.stack(actions)
    actions_log_probability = torch.stack(actions_log_probability)
    values = torch.stack(values)
    rewards_tensor = torch.stack(rewards)
    # dones_tensor = torch.stack(dones)

    returns = calculate_segmented_returns_time_major(rewards_tensor, last_value)
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
    states = states.flatten(
        0, 1
    )  # From [Rollout, Env, 3, 16, 16] -> [Rollout * Env, 3, 16, 16]
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
        for _, (
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


def make_env():
    env = gym.make("caro_env/Gomoku-v0")
    env = RandomStartWrapper(env)
    env = gym.wrappers.Autoreset(env)
    return env


def select_random_op():
    if len(SelfPlayOpponentWrapper.OPPS) == 0:
        raise Exception("No opps add yet")

    if np.random.random() < 0.5:
        op = SelfPlayOpponentWrapper.OPPS[len(SelfPlayOpponentWrapper.OPPS) - 1]
    else:
        op = np.random.choice(SelfPlayOpponentWrapper.OPPS)

    RandomStartWrapper.CURRENT_OP = op

    return op


def run_ppo():
    MAX_EPISODES = 20000
    DISCOUNT_FACTOR = 1
    PRINT_INTERVAL = 10
    PPO_STEPS = 8
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFFICIENT = 0.01
    LEARNING_RATE = 1e-4
    SAVE_INTERVAL = 50
    TEMPERATURE = 1.2
    ADD_OPP_INTERVAL = 500

    train_rewards = []
    policy_losses = []
    value_losses = []

    num_envs = 4

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])

    agent = create_agent(envs)
    SelfPlayOpponentWrapper.add_opp(agent)  # Random action opponent always as base line

    start_episode = load_agent_weights(agent, "checkpoints_gomuko/ppo_latest.pt")
    agent.to(device)

    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    load_optimizer_state(optimizer, "checkpoints_gomuko/ppo_latest.pt", device)

    SelfPlayOpponentWrapper.DEVICE = device

    MAX_TIME_STEPS = 512

    for episode in trange(start_episode, MAX_EPISODES + 1):
        current_opp_model = select_random_op()

        states, actions, actions_log_probability, advantages, returns, rewards = (
            forward_pass(
                envs, agent, MAX_TIME_STEPS, current_opp_model, TEMPERATURE, num_envs
            )
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

        # test_reward = evaluate(single_env, agent, DISCOUNT_FACTOR)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        train_rewards.append(
            calculate_total_rewards_with_discount_factor(rewards, DISCOUNT_FACTOR).cpu()
        )
        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_abs_policy_loss = np.mean(np.abs(policy_losses[-N_TRIALS:]))
        mean_abs_value_loss = np.mean(np.abs(value_losses[-N_TRIALS:]))

        if episode % PRINT_INTERVAL == 0:
            print(
                f"Episode: {episode:3} | \
                  Mean Train Rewards: {mean_train_rewards:3.1f} \
                  | Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} \
                  | Mean Abs Value Loss: {mean_abs_value_loss:2.2f}"
            )

        if episode % SAVE_INTERVAL == 0:
            save_checkpoint(
                episode, agent, optimizer, train_rewards, "checkpoints_gomuko"
            )

        if episode % ADD_OPP_INTERVAL == 0:
            SelfPlayOpponentWrapper.add_opp(agent)


run_ppo()
