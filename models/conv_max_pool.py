import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMaxPool(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 12, conv_dropout=0.25, dropout=0.25,
                 kernel_size=7):
        super(ConvMaxPool, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size=kernel_size, stride=1,
                               padding=0)
        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = F.relu(self.conv1(x))
        o = self.dropout(o)
        o = torch.mean(o, dim=-1)
        o = self.linear(o)
        return self.output(o)
