import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstAttention(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size = 7):
        super(FirstAttention, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size//2)
        self.attend = nn.Conv1d(embeddings_dim, embeddings_dim, 1, stride=1)

        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, seq_len) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.conv1(x)
        attention = self.attend(o)
        o = torch.sum(o * F.softmax(attention, dim=-1), dim=-1)  # [batchsize, embeddingsdim]
        o = self.linear(o)
        return self.output(o)
