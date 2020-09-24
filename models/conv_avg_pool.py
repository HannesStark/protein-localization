import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAvgPool(nn.Module):
    def __init__(self, n_conv_layers: int = 6):
        super(ConvAvgPool, self).__init__()


        self.conv1 = nn.Conv1d(1024, 512, 3, stride=1)
        self.conv2 = nn.Conv1d(512, 256, 3, stride=1)
        self.conv_layers = nn.ModuleList()
        for i in range(n_conv_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(256, 256, 3, stride=1),
                nn.Sigmoid(),
                nn.BatchNorm1d(256)
            ))

        self.linear = nn.Sequential(
            nn.Linear(256, 32),
            nn.Dropout(0.25),
            nn.Sigmoid(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, sequence_length, embeddings_dim] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = F.sigmoid(self.conv1(x))
        o = F.sigmoid(self.conv2(o))
        for conv_layer in self.conv_layers:
            o = conv_layer(o)
        o = torch.mean(o, dim=-1)
        o = self.linear(o)
        return self.output(o)
