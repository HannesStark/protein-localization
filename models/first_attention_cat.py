import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstAttention(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=7, conv_dropout: float = 0.25):
        super(FirstAttention, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2)
        self.attend = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2*embeddings_dim, 32),
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
        embeddings_dim = x.shape[1]
        mask = mask[:, None, :]  # add singleton dimension for broadcasting
        o = self.conv1(x)
        o = self.dropout(o)
        attention = self.attend(x)
        attention = attention.masked_fill(mask == False, -1e9)
        o1 = torch.sum(o * F.softmax(attention, dim=-1), dim=-1)  # [batchsize, embeddingsdim]
        o2, _ = torch.max(o, dim=-1)
        o = torch.cat([o1, o2], dim=-1)
        o = self.linear(o)
        return self.output(o)

        mask = mask[:, None, :]  # add singleton dimension for broadcasting
        o1 = F.relu(self.conv1(x[:, :embeddings_dim // 2, :]))
        o1 = self.dropout1(o1)
        o1_avg = torch.sum(o1 * mask, dim=-1) / mask.sum(dim=-1)
        o1_max, _ = torch.max(o1, dim=-1)
        o1 = torch.cat([o1_avg, o1_max], dim=-1)
        o1 = self.linear1(o1)

        o2 = F.relu(self.conv2(x[:, embeddings_dim // 2:, :]))
        o2 = self.dropout2(o2)
        o2_avg = torch.sum(o2 * mask, dim=-1) / mask.sum(dim=-1)
        o2_max, _ = torch.max(o2, dim=-1)
        o2 = torch.cat([o2_avg, o2_max], dim=-1)
        o2 = self.linear2(o2)

        o = torch.cat([o1, o2], dim=-1)
        return self.output(o)