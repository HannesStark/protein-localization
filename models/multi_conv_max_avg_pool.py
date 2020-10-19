import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiConvMaxAvgPool(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 10, dropout=0.25, conv_dropout: int = 0.25,
                 kernel_size=7):
        super(MultiConvMaxAvgPool, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2)

        self.conv3 = nn.Conv1d(embeddings_dim * 2, embeddings_dim, kernel_size=kernel_size, stride=2,
                               padding=kernel_size // 2)
        self.conv4 = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size=kernel_size, stride=2,
                               padding=kernel_size // 2)

        self.dropout1 = nn.Dropout(conv_dropout)
        self.dropout2 = nn.Dropout(conv_dropout)
        self.dropout3 = nn.Dropout(conv_dropout)
        self.dropout4 = nn.Dropout(conv_dropout)
        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = F.relu(self.conv1(x))
        o = self.dropout1(o)
        o = F.relu(self.conv2(o))
        o = self.dropout2(o)

        o = torch.cat([o, x], dim=1)

        o = F.relu(self.conv3(o))
        o = self.dropout3(o)
        o = F.relu(self.conv4(o))
        o = self.dropout4(o)

        o1 = torch.mean(o, dim=-1)
        o2, _ = torch.max(o, dim=-1)
        o = torch.cat([o1, o2], dim=-1)
        o = self.linear(o)
        return self.output(o)
