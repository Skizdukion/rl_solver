from gymnasium.envs.registration import register

register(
    id="caro_env/GridWorld-v0",
    entry_point="caro_env.envs:GridWorldEnv",
)

register(
    id="caro_env/Gomoku-v0",
    entry_point="caro_env.envs:GomokuEnv",
)
