import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from ..utils.unpad import unpad_padded


class BiLSTMLayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=1280,
        num_layers=1,
        dropout=0.3,
        bidirectional=True,
    ):
        super(BiLSTMLayer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.norm = nn.LayerNorm(self.hidden_size * self.num_directions)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

    def forward(self, x, hidden=None):
        xl = list(map(len, x))
        x = pad_sequence(x, True)

        packed_emb = pack_padded_sequence(x, xl, True, False)
        packed_outputs, hidden = self.lstm(packed_emb, hidden)
        x, _ = pad_packed_sequence(packed_outputs, True)

        if self.bidirectional:
            hidden = self._cat_directions(hidden)

        if isinstance(hidden, tuple):
            hidden = torch.cat(hidden, 0)

        x = self.norm(x)
        x = unpad_padded(x, xl)

        return x

    def _cat_directions(self, hidden):
        def _cat(h):
            return torch.cat([h[0 : h.size(0) : 2], h[1 : h.size(0) : 2]], 2)

        if isinstance(hidden, tuple):
            hidden = tuple([_cat(h) for h in hidden])
        else:
            hidden = _cat(hidden)

        return hidden
