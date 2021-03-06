import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(torch.nn.Module):

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

    def forward(self,
        query: torch.Tensor,  # [decoder_dim]
        values: torch.Tensor, # [seq_length, encoder_dim]
        ):
        weights = self._get_weights(query, values) # [seq_length]
        weights = torch.nn.functional.softmax(weights, dim=0)
        return weights @ values  # [encoder_dim]

class AdditiveAttention(Attention):

    def __init__(self, encoder_dim, decoder_dim):
        super().__init__(encoder_dim, decoder_dim)
        self.v = torch.nn.Parameter(
            torch.FloatTensor(self.decoder_dim).uniform_(-0.1, 0.1))
        self.W_1 = torch.nn.Linear(self.decoder_dim, self.decoder_dim)
        self.W_2 = torch.nn.Linear(self.encoder_dim, self.decoder_dim)

    def _get_weights(self,
        query: torch.Tensor,  # [decoder_dim]
        values: torch.Tensor,  # [seq_length, encoder_dim]
    ):
        query = query.repeat(values.size(0), 1)  # [seq_length, decoder_dim]
        weights = self.W_1(query) + self.W_2(values)  # [seq_length, decoder_dim]
        return torch.tanh(weights) @ self.v  # [seq_length]

class DeepLoc(nn.Module):
    def __init__(self, embeddings_dim: int, output_dim, lstm_hidden_dim, n_layers, dropout):
        super(DeepLoc, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, embeddings_dim, 21, stride=1, padding=21 // 2)
        self.conv2 = nn.Conv1d(embeddings_dim, embeddings_dim, 15, stride=1, padding=15 // 2)
        self.conv3 = nn.Conv1d(embeddings_dim, embeddings_dim, 9, stride=1, padding=9 // 2)
        self.conv4 = nn.Conv1d(embeddings_dim, embeddings_dim, 5, stride=1, padding=5 // 2)
        self.conv5 = nn.Conv1d(embeddings_dim, embeddings_dim, 3, stride=1, padding=3 // 2)
        self.conv6 = nn.Conv1d(embeddings_dim, embeddings_dim, 1, stride=1, padding=1 // 2)

        self.conv7 = nn.Conv1d(embeddings_dim, embeddings_dim, 3, stride=1, padding=3 // 2)

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(embeddings_dim, lstm_hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, x, mask):
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o1 = F.relu(self.conv1(x))  # [batchsize, embeddingsdim, seq_len]
        o2 = F.relu(self.conv2(x))
        o3 = F.relu(self.conv3(x))
        o4 = F.relu(self.conv4(x))
        o5 = F.relu(self.conv5(x))
        o6 = F.relu(self.conv6(x))

        o = torch.cat([o1, o2, o3, o4, o5, o6], dim=-1)  # [batchsize, embeddingsdim, ~sequence_len*6]

        o = F.relu(self.conv7(o))
        o = self.dropout(o)
        print(o.shape)
        o = o.permute(2, 0, 1)  # [seq_len, batch_size, embedding_dim]
        # num_directions is 2 for a bidirectional lstm
        # output: [seq_len, batch, num_directions * hidden_size] hidden state of t=seq_len
        # hidden_state: [num_layers * num_directions, batch, hidden_size] hidden state of t=seq_len
        # cell_state: [num_layers * num_directions, batch, hidden_size] cell state of t=seq_len
        output, (hidden_state, cell_state) = self.lstm(o)

        hidden = self.dropout(
            torch.cat((hidden_state[-2], hidden_state[-1]), dim=1))  # [batch size, lstm_hidden_dim * num directions]

        return self.output(hidden)  # [batch_size, output_dim]
