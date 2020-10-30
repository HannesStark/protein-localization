import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multi_head_attention import MultiHeadAttention


class ConvMaxSelfAttention(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 12, dropout=0.25, kernel_size=7,
                 attention_dropout: float = 0.25, n_heads=8):
        super(ConvMaxSelfAttention, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2)
        self.multi_head_attention = MultiHeadAttention(embeddings_dim, attention_dropout, n_heads,
                                                       skip_last_linear=True)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU()
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

        # take mean as query
        query = torch.sum(o * mask[:, None, :], dim=-1) / mask[:, None, :].sum(dim=-1)  # [batch_size, embeddings_dim]
        o = o.permute(0, 2, 1)  # [batch_size, sequence_length, embeddings_dim]

        o, _ = self.multi_head_attention(query, o, o, mask)  # [batch_size, 1, embeddings_dim]
        o1 = o.squeeze()  # [batch_size, embeddings_dim]

        o2, _ = torch.max(o, dim=-1)
        o = torch.cat([o1, o2], dim=-1)
        o = self.linear(o)
        return self.output(o)
