import numpy as np
import torch


def from_boards_to_state(boards):
    """
    Converts multiple square boards into a 4-channel (B, 3, N, N) state.
    Assumes: boards shape is (Batch, N, N)
    """
    # 1. Validation
    assert isinstance(boards, np.ndarray), "Input must be a numpy array"
    assert boards.ndim == 3, f"Expected 3D array (B, N, N), got {boards.ndim}D"
    assert (
        boards.shape[1] == boards.shape[2]
    ), f"Each board must be square, got {boards.shape[1:]}"

    my_mask = boards == 1
    opp_mask = boards == -1
    empty_mask = boards == 0

    # 3. Stack along axis 1 (the channel dimension)
    # Resulting shape: (Batch, 3, N, N)
    state = np.stack(
        [
            my_mask.astype(np.float32),
            opp_mask.astype(np.float32),
            empty_mask.astype(np.float32),
        ],
        axis=1,
    )

    return state


def calculate_segmented_returns_time_major(rewards, last_value):
    """
    Args:
        rewards: Tensor of shape [T, B]
        last_value: Tensor of shape [B]
    Returns:
        returns: Tensor of shape [T, B]
    """
    T, B = rewards.shape

    # 1. Append last_value to the end of the time dimension
    # Shape becomes [T + 1, B]
    extended_rewards = torch.cat([rewards, last_value.unsqueeze(0)], dim=0)

    # 2. Flip the time dimension (Dim 0)
    rev = torch.flip(extended_rewards, dims=[0])

    # 3. Mask non-zero rewards
    mask = rev != 0

    # 4. Generate indices for the time dimension [T+1, B]
    # We want indices going from 0 to T on the first dimension
    indices = torch.arange(T + 1, device=rewards.device).unsqueeze(1).expand(-1, B)

    # Keep indices only where there is a reward or last_value
    masked_indices = indices * mask.long()

    # 5. Propagate the highest index (the "latest" reward) through time
    prop_indices = torch.cummax(masked_indices, dim=0)[0]

    # 6. Gather the values from the reversed tensor using the propagated indices
    # filled_rev[t, b] = rev[prop_indices[t, b], b]
    filled_rev = torch.gather(rev, 0, prop_indices)

    # 7. Flip back to original order and remove the extra time step
    returns = torch.flip(filled_rev, dims=[0])[:-1, :]

    return returns


def calculate_adversarial_gae(
    rewards, values, last_value, terminations, gamma=0.99, lambd=0.95
):
    """
    Args:
        rewards: [T, B] - 1.0 for win, -1.0 for loss, 0.0 otherwise.
        values: [T, B] - Critic's value predictions (Current Player's perspective).
        last_value: [B] - Critic's value for the state after the rollout.
        terminations: [T, B] - Boolean mask (True if game ended).
        gamma: Discount factor (usually 0.99 for Gomoku).
        lambd: GAE smoothing factor (usually 0.95).
    """
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)

    # We use GAE to accumulate the 'advantage' moving backwards through time
    last_gae_lam = 0

    # Step through time backwards
    for t in reversed(range(T)):
        # 1. Identify the 'next' value
        if t == T - 1:
            next_value = last_value
            # If the rollout ended exactly when the game ended, next_value is 0
            # (Standard RL logic: no future value after termination)
            next_non_terminal = 1.0 - terminations[t].float()
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - terminations[t].float()

        # 2. ADVERSARIAL TD-ERROR (Delta)
        # delta = reward + (gamma * -next_value * next_non_terminal) - values[t]
        # We negate next_value because it's the opponent's win probability!
        delta = rewards[t] + (gamma * -next_value * next_non_terminal) - values[t]

        # 3. ACCUMULATE GAE
        # We negate the previous GAE (last_gae_lam) because turns alternate.
        # If I am in a good spot now, the previous state was bad for my opponent.
        gae = delta + (gamma * lambd * next_non_terminal * -last_gae_lam)

        advantages[t] = gae
        last_gae_lam = gae

    # 4. CALCULATE RETURNS
    # Return = Advantage + Value
    # This provides the 'ground truth' target for your Critic training.
    returns = advantages + values

    return advantages, returns
