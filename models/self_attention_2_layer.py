import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention2Layer(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, dropout=0.25, n_heads=8):
        super(SelfAttention2Layer, self).__init__()

        self.embeddings_dim = embeddings_dim  # in paper, 512
        self.n_heads = n_heads  # in paper, 8
        self.head_dim = embeddings_dim // n_heads  # in paper, 512 // 8 = 64

        self.fc_q = nn.Linear(embeddings_dim, embeddings_dim)
        self.fc_k = nn.Linear(embeddings_dim, embeddings_dim)
        self.fc_v = nn.Linear(embeddings_dim, embeddings_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to('cuda')

        self.attend = nn.Sequential(nn.Conv1d(embeddings_dim, 512, 1, stride=1), nn.Dropout(dropout))

        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim, 16),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )
        self.output = nn.Linear(16, 11)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        x = x.permute(0, 2, 1)
        query = x
        key = x
        value = x
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)  # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, self.embeddings_dim)  # x = [batch size*query len, embeddings_dim]

        x = self.linear(x)
        return self.output(x)
