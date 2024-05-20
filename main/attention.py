import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentionMechanism(nn.Module):
    def forward(self, *inputs):
        raise NotImplementedError("Implement this.")


class BahdanauAttention(AttentionMechanism):

    def __init__(self, hidden_size=1, key_size=1, query_size=1):

        super(BahdanauAttention, self).__init__()

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        self.proj_keys = None
        self.proj_query = None

    def forward(self, query: Tensor = None, mask: Tensor = None, values: Tensor = None):
        self._check_input_shapes_forward(query=query, mask=mask, values=values)

        assert mask is not None, "mask is required"
        assert self.proj_keys is not None, "projection keys have to get pre-computed"

        self.compute_proj_query(query)

        scores = self.energy_layer(torch.tanh(self.proj_query + self.proj_keys))

        scores = scores.squeeze(2).unsqueeze(1)

        scores = torch.where(mask, scores, scores.new_full([1], float("-inf")))

        alphas = F.softmax(scores, dim=-1)

        context = alphas @ values

        return context, alphas

    def compute_proj_keys(self, keys: Tensor):
        self.proj_keys = self.key_layer(keys)

    def compute_proj_query(self, query: Tensor):
        self.proj_query = self.query_layer(query)

    def _check_input_shapes_forward(
        self, query: torch.Tensor, mask: torch.Tensor, values: torch.Tensor
    ):
        assert query.shape[0] == values.shape[0] == mask.shape[0]
        assert query.shape[1] == 1 == mask.shape[1]
        assert query.shape[2] == self.query_layer.in_features
        assert values.shape[2] == self.key_layer.in_features
        assert mask.shape[2] == values.shape[1]

    def __repr__(self):
        return "BahdanauAttention"


class LuongAttention(AttentionMechanism):
    def __init__(self, hidden_size: int = 1, key_size: int = 1):
        super(LuongAttention, self).__init__()
        self.key_layer = nn.Linear(
            in_features=key_size, out_features=hidden_size, bias=False
        )
        self.proj_keys = None

    def forward(
        self,
        query: torch.Tensor = None,
        mask: torch.Tensor = None,
        values: torch.Tensor = None,
    ):
        self._check_input_shapes_forward(query=query, mask=mask, values=values)

        assert self.proj_keys is not None, "projection keys have to get pre-computed"
        assert mask is not None, "mask is required"

        scores = query @ self.proj_keys.transpose(1, 2)

        scores = torch.where(mask, scores, scores.new_full([1], float("-inf")))

        alphas = F.softmax(scores, dim=-1)

        context = alphas @ values

        return context, alphas

    def compute_proj_keys(self, keys: Tensor):
        self.proj_keys = self.key_layer(keys)

    def _check_input_shapes_forward(
        self, query: torch.Tensor, mask: torch.Tensor, values: torch.Tensor
    ):
        assert query.shape[0] == values.shape[0] == mask.shape[0]
        assert query.shape[1] == 1 == mask.shape[1]
        assert query.shape[2] == self.key_layer.out_features
        assert values.shape[2] == self.key_layer.in_features
        assert mask.shape[2] == values.shape[1]

    def __repr__(self):
        return "LuongAttention"
