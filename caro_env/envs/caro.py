from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Player(Enum):
    X = 1
    O = -1


class GomokuEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=16):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=-1, high=1, shape=(size, size), dtype=int),
                "current_player": spaces.Discrete(2),
            }
        )

        self.action_space = spaces.Discrete(size * size)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self._direction_check = [
            np.array([[1, 0], [-1, 0]]),
            np.array([[0, 1], [0, -1]]),
            np.array([[1, 1], [-1, -1]]),
            np.array([[1, -1], [-1, 1]]),
        ]
        self.board = np.zeros((size, size), dtype=np.int8)

    def _get_obs(self):
        if self._last_player is None:
            cur_player = Player.X.value
        else:
            cur_player = self._last_player * -1

        return {
            "board": self.board.copy(),
            "current_player": cur_player,
        }

    def _get_info(self):
        return {
            "last_play": self._last_play,
            "action_mask": (self.board.ravel() == 0).astype(np.int8),
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._last_play = None
        self._last_player = None
        self.board = np.zeros_like(self.board)

        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), info

    def step(self, action):
        assert isinstance(action, (int, np.integer)), "action must be an integer"

        x = action // self.size
        y = action % self.size

        if self._last_player is None:
            cur_player = Player.X.value
        else:
            cur_player = self._last_player * -1

        if self.board[x][y] != 0:
            raise Exception("Invalid move")

        self.board[x][y] = cur_player

        self._last_play = np.array([x, y], dtype=int)
        self._last_player = cur_player

        terminated = self._check_finish()

        reward = 1 if terminated else 0  # rewards base on current player

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _check_finish(self):
        # Four direction: left, right, up, down
        for direction in self._direction_check:
            point_1 = self._check_point_in_dir(direction[0])
            point_2 = self._check_point_in_dir(direction[1])

            if point_1 + point_2 - 1 >= 5:
                return True

        return False

    def _check_point_in_dir(self, direction):
        x = self._last_play[0]
        y = self._last_play[1]

        point = 1
        cur_x = x + direction[0]
        cur_y = y + direction[1]

        while True:
            if not (0 <= cur_x < self.size and 0 <= cur_y < self.size):
                return point

            if self.board[cur_x][cur_y] != self._last_player:
                return point

            point += 1
            cur_x += direction[0]
            cur_y += direction[1]

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        for r in range(self.size):
            for c in range(self.size):
                value = self.board[r, c]

                center = (
                    int((c + 0.5) * pix_square_size),
                    int((r + 0.5) * pix_square_size),
                )

                if value == 1:  # X
                    offset = pix_square_size * 0.3
                    pygame.draw.line(
                        canvas,
                        (0, 0, 0),
                        (center[0] - offset, center[1] - offset),
                        (center[0] + offset, center[1] + offset),
                        3,
                    )
                    pygame.draw.line(
                        canvas,
                        (0, 0, 0),
                        (center[0] - offset, center[1] + offset),
                        (center[0] + offset, center[1] - offset),
                        3,
                    )

                elif value == -1:  # O
                    pygame.draw.circle(
                        canvas,
                        (0, 0, 0),
                        center,
                        int(pix_square_size * 0.3),
                        3,
                    )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
