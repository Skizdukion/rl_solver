import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
import torch
import caro_env
from caro_env.wrappers.self_play_opp_wrapper import SelfPlayOpponentWrapper
from network.caro.backbone import ActorCritic, BackboneNetwork
from utils.gomuko import from_boards_to_state
from utils.ppo import load_agent_weights


def create_agent(env: gym.Env):
    actor_output = env.action_space.n
    actor = BackboneNetwork(
        in_channels=3,
        hidden_channels=64,
        out_features=actor_output,
        dropout_prob=0.1,
        board_size=env.observation_space["board"].shape[0],
    )
    critic = BackboneNetwork(
        in_channels=3,
        hidden_channels=64,
        out_features=1,
        dropout_prob=0.1,
        board_size=env.observation_space["board"].shape[0],
    )

    agent = ActorCritic(actor, critic)

    return agent


def run_simluation(env, agent):
    observation, info = env.reset()
    terminated = False
    truncated = False

    agent.eval()  # Set to evaluation mode (disables dropout)

    with torch.no_grad():
        while not (terminated or truncated):
            env.render()
            board = np.expand_dims(observation["board"], axis=0)
            state = from_boards_to_state(board)
            state_tensor = torch.tensor(state, dtype=torch.float32)

            action_pred, _ = agent(state_tensor)

            mask = torch.tensor(info["action_mask"])
            masked_action_pred = action_pred + (1 - mask) * -1e9
            probs = torch.softmax(masked_action_pred, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

            observation, reward, terminated, truncated, info = env.step(action)

    # if reward == 1:
    #     print("AGENT WIN =======")
    # else:
    #     print("AGENT LOSE =======")

    return reward


env = gym.make("caro_env/Gomoku-v0", size=5, win_size=3, render_mode="rgb_array")
env = RecordVideo(env, video_folder="recordings", episode_trigger=lambda x: x == 0)
env = SelfPlayOpponentWrapper(env)

agent = create_agent(env)
# load_agent_weights(agent, "checkpoints_gomuko/150.pt")
load_agent_weights(agent, "checkpoints_gomuko_basic_opp/250.pt")

# basic_op = create_agent(env)
# load_agent_weights(basic_op, "checkpoints_gomuko/50.pt")

SelfPlayOpponentWrapper.DEVICE = "cpu"
SelfPlayOpponentWrapper.add_opp(agent)

total_reward = 0
for i in range(1000):
    total_reward += run_simluation(env, agent)

print("TOTAL AGENT REWARD")
print(total_reward)

# run_simluation(env, agent)

env.close()
