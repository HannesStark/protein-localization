import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvsStrangeAttendConcat(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, dropout=0.25):
        super(ConvsStrangeAttendConcat, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, 512, 21, stride=1)
        self.conv2 = nn.Conv1d(embeddings_dim, 512, 15, stride=1)
        self.conv3 = nn.Conv1d(embeddings_dim, 512, 9, stride=1)
        self.conv4 = nn.Conv1d(embeddings_dim, 512, 5, stride=1)
        self.conv5 = nn.Conv1d(embeddings_dim, 512, 3, stride=1)
        self.conv6 = nn.Conv1d(embeddings_dim, 512, 1, stride=1)

        self.attend1 = nn.Sequential(nn.Conv1d(512, 512, 1, stride=1), nn.Dropout(dropout))
        self.attend2 = nn.Sequential(nn.Conv1d(512, 512, 1, stride=1), nn.Dropout(dropout))
        self.attend3 = nn.Sequential(nn.Conv1d(512, 512, 1, stride=1), nn.Dropout(dropout))
        self.attend4 = nn.Sequential(nn.Conv1d(512, 512, 1, stride=1), nn.Dropout(dropout))
        self.attend5 = nn.Sequential(nn.Conv1d(512, 512, 1, stride=1), nn.Dropout(dropout))
        self.attend6 = nn.Sequential(nn.Conv1d(512, 512, 1, stride=1), nn.Dropout(dropout))

        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim * 3, 15),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(15)
        )
        self.output = nn.Linear(15, 11)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """

        o1 = F.relu(self.conv1(x))  # [batchsize, embeddingsdim/2, seq_len]
        o2 = F.relu(self.conv2(x))
        o3 = F.relu(self.conv3(x))
        o4 = F.relu(self.conv4(x))
        o5 = F.relu(self.conv5(x))
        o6 = F.relu(self.conv6(x))

        attention1 = self.attend1(o1)
        attention2 = self.attend2(o2)
        attention3 = self.attend3(o3)
        attention4 = self.attend4(o4)
        attention5 = self.attend5(o5)
        attention6 = self.attend5(o6)

        o1 = torch.sum(o1 * F.softmax(attention1, dim=-1), dim=-1)  # [batchsize, embeddingsdim/2]
        o2 = torch.sum(o2 * F.softmax(attention2, dim=-1), dim=-1)
        o3 = torch.sum(o3 * F.softmax(attention3, dim=-1), dim=-1)
        o4 = torch.sum(o4 * F.softmax(attention4, dim=-1), dim=-1)
        o5 = torch.sum(o5 * F.softmax(attention5, dim=-1), dim=-1)
        o6 = torch.sum(o6 * F.softmax(attention6, dim=-1), dim=-1)

        o = torch.cat([o1, o2, o3, o4, o5, o6], dim=-1)  # [batchsize, embeddingsdim*3]
        return self.output(o)
