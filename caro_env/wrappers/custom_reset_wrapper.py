import gymnasium as gym

class CustomResetWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
    ):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            obs, info = self.env.reset()

        return obs, reward, terminated, truncated, info
