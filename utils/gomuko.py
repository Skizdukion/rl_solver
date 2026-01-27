import numpy as np


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
