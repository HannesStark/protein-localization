import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMaxAvgPoolSmall(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 10, dropout: float = 0.25, kernel_size: int = 7,
                 conv_dropout: float = 0.25):
        super(ConvMaxAvgPoolSmall, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, 256, kernel_size=1, stride=1,
                               padding=0)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2)
        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(512, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.


        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """

        mask = mask[:, None, :]  # add singleton dimension for broadcasting
        o = F.relu(self.conv1(x))  # [batch_size, embeddings_dim, sequence_length - kernel_size//2]
        o = F.relu(self.conv2(o))  # [batch_size, embeddings_dim, sequence_length - kernel_size//2]
        o = self.dropout(o)
        o1 = torch.sum(o * mask, dim=-1) / mask.sum(dim=-1)
        o2, _ = torch.max(o, dim=-1)
        o = torch.cat([o1, o2], dim=-1)
        o = self.linear(o)
        return self.output(o)

