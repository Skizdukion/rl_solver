import gymnasium as gym
import numpy as np
import caro_env
from utils.gomuko import from_boards_to_state
from caro_env.wrappers.self_play_opp_wrapper import SelfPlayOpponentWrapper


def verify():
    env = gym.make("caro_env/Gomoku-v0")
    # We want Opponent to start (P1). Agent will be P2.
    # But wait, raw env doesn't have Opponent wrapper.
    # We can manually step.

    obs, info = env.reset(options={"opp_starts": True})
    # opp_starts=True -> Last player = Agent. Next = Opponent.
    # Opponent is P1. Agent is P2.

    # Manually make Opponent move (P1)
    # P1 is -1 in terms of logic inside step?
    # reset: _last_player = Agent (1).
    # step: cur = _last * -1 = -1.
    # So P1 places -1.
    # Let's verify this first.

    # Opponent (P1) places at 0
    obs, reward, term, trunc, info = env.step(0)
    board = obs["board"]
    print(f"After P1 move at 0. Board at 0: {board.flatten()[0]}")
    # If P1 is -1, then board[0] should be -1.

    # Now it is Agent's turn (P2).
    # Agent is always 1?
    # step: cur = _last(-1) * -1 = 1.
    # So Agent places 1.

    # Agent (P2) places at 1
    obs, reward, term, trunc, info = env.step(1)
    board = obs["board"]
    print(f"After P2 (Agent) move at 1. Board at 1: {board.flatten()[1]}")

    # Now let's check from_boards_to_state
    # Agent is considering its NEXT move.
    # Board has: 0->-1 (Opp), 1->1 (Agent).

    state = from_boards_to_state(np.expand_dims(board, 0))
    # state shape: (1, 3, N, N)
    # Channel 0: My stones (boards == 1)
    # Channel 1: Opp stones (boards == -1)

    # Agent is P2 (1).
    # My stones (1) are at index 1.
    # my_mask (boards==1) should show index 1.
    print(f"Agent View (P1=1, P2=1? Wait).")
    print(
        f"Channel 0 (My Stones) at index 1: {state[0, 0].flatten()[1]}"
    )  # Should be 1
    print(
        f"Channel 0 (My Stones) at index 0: {state[0, 0].flatten()[0]}"
    )  # Should be 0
    print(
        f"Channel 1 (Opp Stones) at index 0: {state[0, 1].flatten()[0]}"
    )  # Should be 1

    print("-" * 20)

    # Now consider the Opponent (SelfPlayOpponentWrapper).
    # Opponent is P1.
    # P1 stones are -1.
    # But Opponent uses the SAME from_boards_to_state logic.
    # Opponent thinks "My stones are 1".

    # If Opponent looks at this board:
    # Board has 1 (Agent), -1 (Opp/P1).
    # Opponent (P1) converts:
    # Channel 0 (My): matches 1 (Agent).
    # Channel 1 (Opp): matches -1 (Self).

    # So Opponent thinks Agent's stones are its own.
    print("Opponent Perspective check:")
    print(f"Board value at Agent's spot (1): {board.flatten()[1]}")  # 1
    print(f"Board value at Opp's spot (0): {board.flatten()[0]}")  # -1

    # If logic is: my_mask = (board == 1)
    is_my_stone = board.flatten()[1] == 1
    print(f"Does Opponent think Agent's stone is MINE? {is_my_stone}")

    is_opp_stone = board.flatten()[0] == -1
    print(f"Does Opponent think Its OWN stone is ENEMY? {is_opp_stone}")

    print("-" * 20)
    print("Testing FIX: Inverting Board (* -1)")

    # Apply Fix
    fixed_board = board * -1
    fixed_state = from_boards_to_state(np.expand_dims(fixed_board, 0))

    # Now Opponent should see:
    # Its stone (originally -1) -> becomes 1.
    # State Channel 0 (My Stones) should catch this.

    # Opponent stone is at index 0.
    # fixed_state[0, 0] is Channel 0 (My Stones).
    # fixed_state[0, 0][0] should be 1.

    print(f"Index 0 (Opp Stone). Original Board val: {board.flatten()[0]}")
    print(f"Index 0 (Opp Stone). Inverted Board val: {fixed_board.flatten()[0]}")
    print(
        f"Index 0 (Opp Stone). State Channel 0 (My): {fixed_state[0, 0].flatten()[0]}"
    )

    # Agent stone (originally 1) -> becomes -1.
    # State Channel 1 (Enemy Stones) should catch this.
    # Agent stone is at index 1.

    print(f"Index 1 (Agent Stone). Original Board val: {board.flatten()[1]}")
    print(f"Index 1 (Agent Stone). Inverted Board val: {fixed_board.flatten()[1]}")
    print(
        f"Index 1 (Agent Stone). State Channel 1 (Enemy): {fixed_state[0, 1].flatten()[1]}"
    )

    if fixed_state[0, 0].flatten()[0] == 1.0 and fixed_state[0, 1].flatten()[1] == 1.0:
        print(
            "\nSUCCESS: Opponent now correctly identifies its own stones and enemy stones."
        )
    else:
        print("\nFAILURE: Still incorrect.")


if __name__ == "__main__":
    verify()
