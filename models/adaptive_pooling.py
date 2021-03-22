import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePooling(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(AdaptivePooling, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, embeddings_dim, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(embeddings_dim, embeddings_dim, 15, stride=1, padding=7)
        self.conv3 = nn.Conv1d(embeddings_dim, embeddings_dim, 9, stride=1, padding=4)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(5)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(5 * 3 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, output_dim)

        self.id0 = nn.Identity()
        self.id1 = nn.Identity()
        self.id2 = nn.Identity()

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """

        o1 = F.relu(self.conv1(x))  # [batchsize, embeddingsdim, seq_len]
        o2 = F.relu(self.conv2(x))
        o3 = F.relu(self.conv3(x))

        o = torch.cat([o1, o2, o3], dim=1)
        o = self.adaptive_pool(o)
        o = o.view(o.shape[0], -1)
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]
