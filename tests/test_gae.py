import torch
import numpy as np
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gomuko import calculate_adversarial_gae


def test_immediate_win():
    print("Test Case 1: Immediate Win")
    # T=1, B=1
    # Player moves and wins immediately.
    rewards = torch.tensor([[1.0]])
    values = torch.tensor([[0.5]])  # Critic thought it was 0.5
    last_value = torch.tensor([0.0])  # Next state value (irrelevant due to termination)
    terminations = torch.tensor([[True]])

    gamma = 1.0
    lambd = 1.0

    advantages, returns = calculate_adversarial_gae(
        rewards, values, last_value, terminations, gamma, lambd
    )

    # Delta = r + gamma * (-next_val * mask) - val
    # Delta = 1.0 + 1.0 * (-0.0 * 0) - 0.5 = 0.5
    # GAE = Delta + gamma * lambd * mask * (-prev_gae)
    # GAE = 0.5 + 0 = 0.5
    # Return = Adv + Val = 0.5 + 0.5 = 1.0

    print(f"Advantages: {advantages}")
    print(f"Returns: {returns}")

    assert torch.isclose(returns[0, 0], torch.tensor(1.0)), "Return should be 1.0"
    assert torch.isclose(advantages[0, 0], torch.tensor(0.5)), "Advantage should be 0.5"
    print("Passed!\n")


def test_delayed_win():
    print("Test Case 2: Delayed Win (2 steps)")
    # T=2, B=1
    # t=0: Agent moves.
    # t=1: Opponent moves and loses.

    rewards = torch.tensor([[0.0], [-1.0]])
    values = torch.tensor([[0.2], [-0.5]])  # V(S0)=0.2, V(S1)=-0.5
    terminations = torch.tensor([[False], [True]])
    last_value = torch.tensor([0.0])

    gamma = 1.0
    lambd = 1.0

    # Backwards pass:

    # t=1 (Opponent):
    # R=-1, Term=True.
    # next_val = 0. next_non_term = 0.
    # delta = -1 + 0 - (-0.5) = -0.5
    # gae = -0.5 + 0 = -0.5
    # Return = -0.5 + (-0.5) = -1.0. Correct. Opponent lost.
    # last_gae = -0.5

    # t=0 (Agent):
    # R=0. Term=False.
    # next_val = values[1] = -0.5
    # next_non_term = 1.
    # delta = 0 + (1 * -(-0.5) * 1) - 0.2
    #       = 0 + 0.5 - 0.2 = 0.3
    # gae = delta + (1 * 1 * 1 * -last_gae)
    #     = 0.3 + (-(-0.5)) = 0.3 + 0.5 = 0.8
    # Return = 0.8 + 0.2 = 1.0.

    advantages, returns = calculate_adversarial_gae(
        rewards, values, last_value, terminations, gamma, lambd
    )

    print(f"Advantages: {advantages}")
    print(f"Returns: {returns}")

    assert torch.isclose(returns[1, 0], torch.tensor(-1.0)), "t=1 Return should be -1.0"
    assert torch.isclose(returns[0, 0], torch.tensor(1.0)), "t=0 Return should be 1.0"
    print("Passed!\n")


if __name__ == "__main__":
    test_immediate_win()
    test_delayed_win()
