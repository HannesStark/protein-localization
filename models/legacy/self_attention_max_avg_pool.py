import torch
import torch.nn as nn

from models.legacy.multi_head_attention import MultiHeadAttention


class SelfAttentionMaxAvgPool(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 12 , dropout=0.25, attention_dropout=0.25, n_heads=8):
        super(SelfAttentionMaxAvgPool, self).__init__()

        self.multi_head_attention1 = MultiHeadAttention(embeddings_dim, attention_dropout, n_heads)
        self.multi_head_attention2 = MultiHeadAttention(embeddings_dim, attention_dropout, n_heads,
                                                        skip_last_linear=True)

        self.linear = nn.Sequential(
            nn.Linear(2*embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask, sequence_lengths, frequencies) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, embeddings_dim]

        o, _ = self.multi_head_attention1(x, x, x)

        o1 = torch.mean(o, dim=-2)
        o2, _ = torch.max(o, dim=-2)
        o = torch.cat([o1, o2], dim=-1)
        o = self.linear(o)
        return self.output(o)
