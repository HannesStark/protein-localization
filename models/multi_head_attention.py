import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, dropout: float = 0.25, n_heads: int = 8, skip_last_linear=False):
        """
        CAREFUL: this does not do the multihead attention from the paper. The last linear layer is missing.
        Args:
            embeddings_dim: dimension of the passed embeddings
            dropout: dropout values
            n_heads: number of attention heads needs to be a divider of embeddings dim (see paper "Attention is all you need!")
        """
        super().__init__()

        assert embeddings_dim % n_heads == 0

        self.skip_last_linear = False
        self.embeddings_dim = embeddings_dim  # in paper, 512
        self.n_heads = n_heads  # in paper, 8
        self.head_dim = embeddings_dim // n_heads  # in paper, 512 // 8 = 64

        self.query_weights = nn.Linear(embeddings_dim, embeddings_dim)
        self.key_weights = nn.Linear(embeddings_dim, embeddings_dim)
        self.value_weigths = nn.Linear(embeddings_dim, embeddings_dim)

        self.last_linear = nn.Linear(embeddings_dim, embeddings_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        # query = [batch size, query len, embeddigns_dim]
        # key = [batch size, key len, embeddigns_dim]
        # value = [batch size, value len, embeddigns_dim]

        Q = self.query_weights(query)
        K = self.key_weights(key)
        V = self.value_weigths(value)

        # Q = [batch size, query len, embeddigns_dim]
        # K = [batch size, key len, embeddigns_dim]
        # V = [batch size, value len, embeddigns_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        attention_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        attention = torch.softmax(attention_scores, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)  # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.embeddings_dim)  # x = [batch size, query len, embeddigns_dim]

        if not self.skip_last_linear:
            x = self.last_linear(x)  # x = [batch size, query len, embeddigns_dim]

        return x, attention
