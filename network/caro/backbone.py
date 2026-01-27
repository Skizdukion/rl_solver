import torch.nn as nn
import torch.nn.functional as f


class BackboneNetwork(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_features, dropout_prob, board_size=15
    ):
        super().__init__()
        self.board_size = board_size

        # Layer 1: Conv layer (B, in_channels, H, W) -> (B, hidden_channels, H, W)
        # kernel_size=3 with padding=1 keeps the spatial dimensions (H, W) the same.
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # Layer 2: Conv layer (B, hidden_channels, H, W) -> (B, hidden_channels, H, W)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )

        # Flatten: (B, hidden_channels, H, W) -> (B, hidden_channels * H * W)
        self.flatten = nn.Flatten()

        # Layer 3: Fully Connected (B, flattened_size) -> (B, out_features)
        # For a 15x15 board, this is: hidden_channels * 15 * 15
        self.fc = nn.Linear(hidden_channels * board_size * board_size, out_features)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Input x shape: (Batch, in_channels, board_size, board_size)

        # Pass through Conv 1
        x = f.relu(self.conv1(x))
        # Shape: (Batch, hidden_channels, board_size, board_size)

        x = self.dropout(x)

        # Pass through Conv 2
        x = f.relu(self.conv2(x))
        # Shape: (Batch, hidden_channels, board_size, board_size)

        x = self.dropout(x)

        # Flatten the 2D grid into a 1D vector per batch item
        x = self.flatten(x)
        # Shape: (Batch, hidden_channels * board_size * board_size)

        # Final output layer
        x = self.fc(x)
        # Shape: (Batch, out_features)

        return x


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred
