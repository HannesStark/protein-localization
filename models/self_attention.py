import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multi_head_attention import MultiHeadAttention


class SelfAttention(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, dropout=0.25, attention_dropout=0.25, n_heads=8):
        super(SelfAttention, self).__init__()

        self.multi_head_attention = MultiHeadAttention(embeddings_dim, attention_dropout, n_heads, skip_last_linear=True)

        self.flat_conv1 = nn.Conv1d(1, 1, 3, stride=2, padding=1)
        self.flat_conv2 = nn.Conv1d(1, 1, 3, stride=2, padding=1)

        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim // 4, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.output = nn.Linear(32, 11)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        query = x.mean(dim=-1)  # [batch_size, embeddings_dim]
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, embeddings_dim]

        o, _ = self.multi_head_attention(query, x, x)  # [batch_size, 1, embeddings_dim]
        o = o.squeeze()  # [batch_size, embeddings_dim]
        o = o[:, None, :]
        o = F.relu(self.flat_conv1(o))
        o = F.relu(self.flat_conv2(o))
        o = o.view(x.shape[0], -1)  # [batchsize, embeddingsdim//?]
        o = self.linear(o)
        return self.output(o)
