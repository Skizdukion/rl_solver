import os
import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from network.grid_world.backbone import ActorCritic, BackboneNetwork
import caro_env

def create_agent(env: gym.Env, hidden_dims=64, dropout=0.2):
    input_features = (
        env.observation_space["agent"].shape[0]
        + env.observation_space["target"].shape[0]
    )  # only position of agent and target is needed

    actor_output = env.action_space.n
    critic_output = 1
    actor = BackboneNetwork(input_features, hidden_dims, actor_output, dropout)
    critic = BackboneNetwork(
        input_features, hidden_dims, critic_output, dropout)

    agent = ActorCritic(actor, critic)

    return agent


def load_checkpoint(agent, path="checkpoints_grid_world/ppo_latest.pt"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        agent.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        print(f"Resuming from episode {start_episode}")
        return start_episode
    return 0


def run_simluation(env, agent):
    observation, _ = env.reset()
    terminated = False
    truncated = False

    agent.eval()  # Set to evaluation mode (disables dropout)

    with torch.no_grad():
        while not (terminated or truncated):
            # 1. Render the frame for human viewing
            env.render()

            # 2. Prepare state
            state = np.concatenate(
                [observation["agent"], observation["target"]])
            state_tensor = torch.tensor(
                state, dtype=torch.float32).unsqueeze(0)

            # 3. Select Best Action (Deterministically)
            action_pred, _ = agent(state_tensor)
            action = torch.argmax(action_pred, dim=-1).item()

            # 4. Step
            observation, _, terminated, truncated, _ = env.step(action)

    return


env = gym.make("caro_env/GridWorld-v0", render_mode="rgb_array")
env = RecordVideo(env, video_folder="recordings",
                  episode_trigger=lambda x: x % 50 == 0)

agent = create_agent(env)
load_checkpoint(agent)

run_simluation(env, agent)
env.close()