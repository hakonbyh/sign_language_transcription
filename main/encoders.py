import logging

import torch
import torch.nn as nn
from main.helpers import freeze_params
from main.transformer_layers import (
    BERTIdentity,
    PositionalEncoding,
    TransformerEncoderLayer,
)
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertConfig, BertModel

logger = logging.getLogger(__name__)


class Encoder(nn.Module):

    @property
    def output_size(self):
        return self._output_size


class RecurrentEncoder(Encoder):

    def __init__(
        self,
        rnn_type: str = "gru",
        hidden_size: int = 1,
        emb_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        bidirectional: bool = True,
        freeze: bool = False,
        **kwargs,
    ) -> None:

        super(RecurrentEncoder, self).__init__()

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.type = rnn_type
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.rnn = rnn(
            emb_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self._output_size = 2 * hidden_size if bidirectional else hidden_size

        if freeze:
            freeze_params(self)

    def _check_shapes_input_forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> None:
        assert embed_src.shape[0] == src_length.shape[0]
        assert embed_src.shape[2] == self.emb_size
        assert len(src_length.shape) == 1

    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> (Tensor, Tensor):
        self._check_shapes_input_forward(
            embed_src=embed_src, src_length=src_length, mask=mask
        )

        embed_src = self.emb_dropout(embed_src)

        packed = pack_padded_sequence(embed_src, src_length, batch_first=True)
        output, hidden = self.rnn(packed)

        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden

        output, _ = pad_packed_sequence(output, batch_first=True)
        batch_size = hidden.size()[1]
        hidden_layerwise = hidden.view(
            self.rnn.num_layers,
            2 if self.rnn.bidirectional else 1,
            batch_size,
            self.rnn.hidden_size,
        )
        fwd_hidden_last = hidden_layerwise[-1:, 0]
        bwd_hidden_last = hidden_layerwise[-1:, 1]

        hidden_concat = torch.cat([fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        return output, hidden_concat

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.rnn)


class TransformerEncoder(Encoder):
    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        freeze_like_fpt: bool = False,
        freeze_type: str = None,
        **kwargs,
    ):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        logger.info(f"Freezing transformer: {freeze_like_fpt}")
        if freeze_like_fpt:
            logger.info(f"Freezing type: {freeze_type}")
            for name, p in self.layers.named_parameters():
                name = name.lower()
                if "layer_norm" not in name:
                    p.requires_grad = False
            if freeze_type == "finetune_ff":
                for name, p in self.layers.named_parameters():
                    name = name.lower()
                    if "feed_forward" in name:
                        p.requires_grad = True

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = hidden_size

        if freeze:
            freeze_params(self)

    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> (Tensor, Tensor):
        x = self.pe(embed_src)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x), None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].src_src_att.num_heads,
        )


class BERTEncoder(Encoder):
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 3,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        freeze_pt: str = None,
        pretrained_name: str = "bert-base-uncased",
        input_layer_init: str = None,
        pretrain: bool = True,
        **kwargs,
    ):
        super(BERTEncoder, self).__init__()

        if pretrain:
            logger.info("Using pretrained model")
            self.bert_model = BertModel.from_pretrained(pretrained_name)
        else:
            logger.info("Using BERT from scratch")
            self.config = BertConfig.from_pretrained(pretrained_name)
            self.bert_model = BertModel(self.config)
        self.encoder = self.bert_model.encoder
        assert self.encoder.config.hidden_size == hidden_size

        orig_nr_layers = len(self.encoder.layer)
        for i in range(orig_nr_layers - 1, num_layers - 1, -1):
            self.encoder.layer[i] = BERTIdentity()
        self.num_layers = num_layers

        if pretrain:
            logger.info(f"Freezing pre-trained transformer: {freeze_pt}")
            logger.info(f"Freezing entire encoder: {freeze}")
            if freeze_pt != "rnd2rnd":
                for name, p in self.encoder.named_parameters():
                    name = name.lower()
                    if "layernorm" not in name:
                        p.requires_grad = False
                if freeze_pt == "freeze_ff" or freeze_pt == "finetune_ff":
                    for name, p in self.encoder.named_parameters():
                        name = name.lower()
                        if "output" in name and "attention" not in name:
                            p.requires_grad = True
                if freeze_pt == "finetune_ff":
                    for name, p in self.encoder.named_parameters():
                        name = name.lower()
                        if "intermediate" in name:
                            p.requires_grad = True

        self.input_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        if input_layer_init == "orthogonal":
            torch.nn.init.orthogonal_(self.input_layer.weight)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = hidden_size

    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> (Tensor, Tensor):
        x = self.pe(embed_src)
        x = self.emb_dropout(x)

        x = self.input_layer(x)

        x = self.layer_norm(x)
        mask = mask.squeeze(dim=1)
        mask = self.bert_model.get_extended_attention_mask(
            mask, mask.shape, mask.device
        )
        x = self.encoder(x, attention_mask=mask)

        return x.last_hidden_state, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            self.num_layers,
            self.encoder.config.num_attention_heads,
        )
