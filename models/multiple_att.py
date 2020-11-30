import torch
import torch.nn as nn
import torch.nn.functional as F


class MultipleAtt(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=7, conv_dropout: float = 0.25):
        super(MultipleAtt, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2)
        self.attend1 = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2)
        self.attend2 = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout1 = nn.Dropout(conv_dropout)
        self.dropout2 = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(3 * embeddings_dim, 32),
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
        o = self.conv1(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout1(o)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attend1(x)
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)
        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddingsdim]

        o = self.conv2(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout2(o)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attend2(x)
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)
        o2 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddingsdim]

        o3, _ = torch.max(o, dim=-1)
        o = torch.cat([o1, o2, o3], dim=-1)
        o = self.linear(o)
        return self.output(o)
