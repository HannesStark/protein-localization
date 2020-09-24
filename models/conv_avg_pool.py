import torch
import torch.nn as nn


class ConvAvgPool(nn.Module):
    def __init__(self, n_conv_layers: int = 8):
        super(ConvAvgPool, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(n_conv_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(1024, 1024, 3, stride=1),
                nn.Sigmoid()
            ))

        self.linear = nn.Sequential(
            nn.Linear(1024, 32),
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
        o = x
        for conv_layer in self.conv_layers:
            o = conv_layer(o)
        x = torch.mean(x, dim=-1)
        x = self.linear(x)
        return self.output(x)
