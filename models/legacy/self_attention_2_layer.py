import torch
import torch.nn as nn

from models.legacy.multi_head_attention import MultiHeadAttention


class SelfAttention2Layer(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 12, dropout=0.25, attention_dropout=0.25,
                 n_heads=8):
        super(SelfAttention2Layer, self).__init__()

        self.multi_head_attention1 = MultiHeadAttention(embeddings_dim, attention_dropout, n_heads)
        self.multi_head_attention2 = MultiHeadAttention(embeddings_dim, attention_dropout, n_heads,
                                                        skip_last_linear=True)

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
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, embeddings_dim]

        o, _ = self.multi_head_attention1(x, x, x, mask)
        query = torch.sum(o * mask[..., None], dim=-2) / mask[..., None].sum(dim=-2)  # [batch_size, embeddings_dim]

        o, _ = self.multi_head_attention2(query, o, o, mask)  # [batch_size, 1, embeddings_dim]
        o = o.squeeze()  # [batch_size, embeddings_dim]
        o = self.linear(o)
        return self.output(o)
