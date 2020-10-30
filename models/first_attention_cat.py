import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstAttentionCat(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=7, conv_dropout: float = 0.25):
        super(FirstAttentionCat, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim // 2, embeddings_dim // 2, kernel_size, stride=1,
                               padding=kernel_size // 2)
        self.attend1 = nn.Conv1d(embeddings_dim // 2, embeddings_dim // 2, kernel_size, stride=1,
                                 padding=kernel_size // 2)

        self.conv2 = nn.Conv1d(embeddings_dim // 2, embeddings_dim // 2, kernel_size, stride=1,
                               padding=kernel_size // 2)
        self.attend2 = nn.Conv1d(embeddings_dim // 2, embeddings_dim // 2, kernel_size, stride=1,
                                 padding=kernel_size // 2)

        self.dropout1 = nn.Dropout(conv_dropout)
        self.dropout2 = nn.Dropout(conv_dropout)

        self.linear1 = nn.Sequential(
            nn.Linear(embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(64, output_dim)

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

        o1 = self.conv1(x[:, :embeddings_dim // 2, :])
        o1 = self.dropout1(o1)
        attention1 = self.attend1(x[:, :embeddings_dim // 2, :])
        attention1 = attention1.masked_fill(mask == False, -1e9)
        o1_att = torch.sum(o1 * F.softmax(attention1, dim=-1), dim=-1)  # [batchsize, embeddingsdim]
        o1_max, _ = torch.max(o1, dim=-1)
        o1 = torch.cat([o1_att, o1_max], dim=-1)
        o1 = self.linear1(o1)

        o2 = self.conv2(x[:, :embeddings_dim // 2, :])
        o2 = self.dropout2(o2)
        attention2 = self.attend2(x[:, :embeddings_dim // 2, :])
        attention2 = attention2.masked_fill(mask == False, -1e9)
        o2_att = torch.sum(o2 * F.softmax(attention2, dim=-1), dim=-1)  # [batchsize, embeddingsdim]
        o2_max, _ = torch.max(o2, dim=-1)
        o2 = torch.cat([o2_att, o2_max], dim=-1)
        o2 = self.linear2(o2)

        o = torch.cat([o1, o2], dim=-1)
        return self.output(o)
