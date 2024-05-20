import logging
from typing import Optional

import torch
import torch.nn as nn
from main.attention import BahdanauAttention, LuongAttention
from main.encoders import Encoder
from main.helpers import freeze_params, subsequent_mask
from main.transformer_layers import (
    BERTIdentity,
    PositionalEncoding,
    TransformerDecoderLayer,
)
from torch import Tensor
from transformers import BertConfig, BertModel

logger = logging.getLogger(__name__)


class Decoder(nn.Module):

    @property
    def output_size(self):
        return self._output_size


class RecurrentDecoder(Decoder):

    def __init__(
        self,
        rnn_type: str = "gru",
        emb_size: int = 0,
        hidden_size: int = 0,
        encoder: Encoder = None,
        attention: str = "bahdanau",
        num_layers: int = 1,
        vocab_size: int = 0,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        init_hidden: str = "bridge",
        input_feeding: bool = True,
        freeze: bool = False,
        **kwargs,
    ) -> None:

        super(RecurrentDecoder, self).__init__()

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.type = rnn_type
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout, inplace=False)
        self.hidden_size = hidden_size
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.input_feeding = input_feeding
        if self.input_feeding:
            self.rnn_input_size = emb_size + hidden_size
        else:
            self.rnn_input_size = emb_size

        self.rnn = rnn(
            self.rnn_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.att_vector_layer = nn.Linear(
            hidden_size + encoder.output_size, hidden_size, bias=True
        )

        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
        self._output_size = vocab_size

        if attention == "bahdanau":
            self.attention = BahdanauAttention(
                hidden_size=hidden_size,
                key_size=encoder.output_size,
                query_size=hidden_size,
            )
        elif attention == "luong":
            self.attention = LuongAttention(
                hidden_size=hidden_size, key_size=encoder.output_size
            )
        else:
            raise ValueError(
                "Unknown attention mechanism: %s. "
                "Valid options: 'bahdanau', 'luong'." % attention
            )

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.init_hidden_option = init_hidden
        if self.init_hidden_option == "bridge":
            self.bridge_layer = nn.Linear(encoder.output_size, hidden_size, bias=True)
        elif self.init_hidden_option == "last":
            if encoder.output_size != self.hidden_size:
                if encoder.output_size != 2 * self.hidden_size:
                    raise ValueError(
                        "For initializing the decoder state with the "
                        "last encoder state, their sizes have to match "
                        "(encoder: {} vs. decoder:  {})".format(
                            encoder.output_size, self.hidden_size
                        )
                    )
        if freeze:
            freeze_params(self)

    def _check_shapes_input_forward_step(
        self,
        prev_embed: Tensor,
        prev_att_vector: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor,
        hidden: Tensor,
    ) -> None:
        assert prev_embed.shape[1:] == torch.Size([1, self.emb_size])
        assert prev_att_vector.shape[1:] == torch.Size([1, self.hidden_size])
        assert prev_att_vector.shape[0] == prev_embed.shape[0]
        assert encoder_output.shape[0] == prev_embed.shape[0]
        assert len(encoder_output.shape) == 3
        assert src_mask.shape[0] == prev_embed.shape[0]
        assert src_mask.shape[1] == 1
        assert src_mask.shape[2] == encoder_output.shape[1]
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        assert hidden.shape[0] == self.num_layers
        assert hidden.shape[1] == prev_embed.shape[0]
        assert hidden.shape[2] == self.hidden_size

    def _check_shapes_input_forward(
        self,
        trg_embed: Tensor,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        src_mask: Tensor,
        hidden: Tensor = None,
        prev_att_vector: Tensor = None,
    ) -> None:
        assert len(encoder_output.shape) == 3
        assert len(encoder_hidden.shape) == 2
        assert encoder_hidden.shape[-1] == encoder_output.shape[-1]
        assert src_mask.shape[1] == 1
        assert src_mask.shape[0] == encoder_output.shape[0]
        assert src_mask.shape[2] == encoder_output.shape[1]
        assert trg_embed.shape[0] == encoder_output.shape[0]
        assert trg_embed.shape[2] == self.emb_size
        if hidden is not None:
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            assert hidden.shape[1] == encoder_output.shape[0]
            assert hidden.shape[2] == self.hidden_size
        if prev_att_vector is not None:
            assert prev_att_vector.shape[0] == encoder_output.shape[0]
            assert prev_att_vector.shape[2] == self.hidden_size
            assert prev_att_vector.shape[1] == 1

    def _forward_step(
        self,
        prev_embed: Tensor,
        prev_att_vector: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor,
        hidden: Tensor,
    ) -> (Tensor, Tensor, Tensor):
        self._check_shapes_input_forward_step(
            prev_embed=prev_embed,
            prev_att_vector=prev_att_vector,
            encoder_output=encoder_output,
            src_mask=src_mask,
            hidden=hidden,
        )

        if self.input_feeding:
            rnn_input = torch.cat([prev_embed, prev_att_vector], dim=2)
        else:
            rnn_input = prev_embed

        rnn_input = self.emb_dropout(rnn_input)

        _, hidden = self.rnn(rnn_input, hidden)

        if isinstance(hidden, tuple):
            query = hidden[0][-1].unsqueeze(1)
        else:
            query = hidden[-1].unsqueeze(1)
        context, att_probs = self.attention(
            query=query, values=encoder_output, mask=src_mask
        )

        att_vector_input = torch.cat([query, context], dim=2)
        att_vector_input = self.hidden_dropout(att_vector_input)

        att_vector = torch.tanh(self.att_vector_layer(att_vector_input))

        return att_vector, hidden, att_probs

    def forward(
        self,
        trg_embed: Tensor,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        src_mask: Tensor,
        unroll_steps: int,
        hidden: Tensor = None,
        prev_att_vector: Tensor = None,
        **kwargs,
    ) -> (Tensor, Tensor, Tensor, Tensor):

        self._check_shapes_input_forward(
            trg_embed=trg_embed,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            hidden=hidden,
            prev_att_vector=prev_att_vector,
        )

        if hidden is None:
            hidden = self._init_hidden(encoder_hidden)

        if hasattr(self.attention, "compute_proj_keys"):
            self.attention.compute_proj_keys(keys=encoder_output)

        att_vectors = []
        att_probs = []

        batch_size = encoder_output.size(0)

        if prev_att_vector is None:
            with torch.no_grad():
                prev_att_vector = encoder_output.new_zeros(
                    [batch_size, 1, self.hidden_size]
                )

        for i in range(unroll_steps):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            prev_att_vector, hidden, att_prob = self._forward_step(
                prev_embed=prev_embed,
                prev_att_vector=prev_att_vector,
                encoder_output=encoder_output,
                src_mask=src_mask,
                hidden=hidden,
            )
            att_vectors.append(prev_att_vector)
            att_probs.append(att_prob)

        att_vectors = torch.cat(att_vectors, dim=1)
        att_probs = torch.cat(att_probs, dim=1)
        outputs = self.output_layer(att_vectors)
        return outputs, hidden, att_probs, att_vectors

    def _init_hidden(self, encoder_final: Tensor = None) -> (Tensor, Optional[Tensor]):
        batch_size = encoder_final.size(0)

        if self.init_hidden_option == "bridge" and encoder_final is not None:
            hidden = (
                torch.tanh(self.bridge_layer(encoder_final))
                .unsqueeze(0)
                .repeat(self.num_layers, 1, 1)
            )
        elif self.init_hidden_option == "last" and encoder_final is not None:
            if encoder_final.shape[1] == 2 * self.hidden_size:
                encoder_final = encoder_final[:, : self.hidden_size]
            hidden = encoder_final.unsqueeze(0).repeat(self.num_layers, 1, 1)
        else:
            with torch.no_grad():
                hidden = encoder_final.new_zeros(
                    self.num_layers, batch_size, self.hidden_size
                )

        return (hidden, hidden) if isinstance(self.rnn, nn.LSTM) else hidden

    def __repr__(self):
        return "RecurrentDecoder(rnn=%r, attention=%r)" % (self.rnn, self.attention)


class TransformerDecoder(Decoder):

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_size: int = 512,
        ff_size: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        vocab_size: int = 1,
        freeze: bool = False,
        **kwargs,
    ):
        super(TransformerDecoder, self).__init__()

        self._hidden_size = hidden_size
        self._output_size = vocab_size

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.pe = PositionalEncoding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)

        if freeze:
            freeze_params(self)

    def forward(
        self,
        trg_embed: Tensor = None,
        encoder_output: Tensor = None,
        encoder_hidden: Tensor = None,
        src_mask: Tensor = None,
        unroll_steps: int = None,
        hidden: Tensor = None,
        trg_mask: Tensor = None,
        **kwargs,
    ):
        assert trg_mask is not None, "trg_mask required for Transformer"

        x = self.pe(trg_embed)
        x = self.emb_dropout(x)

        trg_mask = trg_mask & subsequent_mask(trg_embed.size(1)).type_as(trg_mask)

        for layer in self.layers:
            x = layer(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask)

        x = self.layer_norm(x)
        output = self.output_layer(x)

        return output, x, None, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].trg_trg_att.num_heads,
        )


class BERTDecoder(Decoder):
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 3,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        freeze_pt: str = None,
        pretrained_name: str = "bert-base-uncased",
        vocab_size: int = 1,
        input_layer_init: str = None,
        pretrain: bool = True,
        **kwargs,
    ):
        super(BERTDecoder, self).__init__()

        self.config = BertConfig.from_pretrained(pretrained_name)
        self.config.is_decoder = True
        self.config.add_cross_attention = True
        if pretrain:
            logger.info("Using pretrained model")
            self.bert_model = BertModel.from_pretrained(
                pretrained_name, config=self.config
            )
        else:
            logger.info("Using BERT from scratch")
            self.bert_model = BertModel(self.config)

        self.decoder = self.bert_model.encoder
        assert self.decoder.config.hidden_size == hidden_size

        orig_nr_layers = len(self.decoder.layer)
        for i in range(orig_nr_layers - 1, num_layers - 1, -1):
            self.decoder.layer[i] = BERTIdentity()
        self.num_layers = num_layers

        if pretrain:
            logger.info(f"Freezing pre-trained transformer: {freeze_pt}")
            logger.info(f"Freezing entire decoder: {freeze}")
            for name, p in self.decoder.named_parameters():
                name = name.lower()
                if "layernorm" not in name:
                    p.requires_grad = False
            if freeze_pt == "freeze_ff" or freeze_pt == "finetune_ff":
                for name, p in self.decoder.named_parameters():
                    name = name.lower()
                    if "output" in name and "attention" not in name:
                        p.requires_grad = True
            if freeze_pt == "finetune_ff":
                for name, p in self.decoder.named_parameters():
                    name = name.lower()
                    if "intermediate" in name:
                        p.requires_grad = True
            for name, p in self.decoder.named_parameters():
                name = name.lower()
                if "crossattention" in name:
                    p.requires_grad = True

        self.input_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        if input_layer_init == "orthogonal":
            torch.nn.init.orthogonal_(self.input_layer.weight)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = vocab_size

        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        trg_embed: Tensor = None,
        encoder_output: Tensor = None,
        encoder_hidden: Tensor = None,
        src_mask: Tensor = None,
        unroll_steps: int = None,
        hidden: Tensor = None,
        trg_mask: Tensor = None,
        **kwargs,
    ):
        x = self.pe(trg_embed)
        x = self.emb_dropout(x)

        x = self.input_layer(x)

        x = self.layer_norm(x)

        src_mask = src_mask[:, None, :, :].to(torch.float)
        src_mask = (1.0 - src_mask) * -10000.0

        trg_mask = trg_mask.squeeze(dim=1)
        trg_mask = self.bert_model.get_extended_attention_mask(
            trg_mask, trg_mask.shape, trg_mask.device
        )

        x = self.decoder(
            x,
            attention_mask=trg_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=src_mask,
        )

        x = x.last_hidden_state

        output = self.output_layer(x)

        return output, x, None, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            self.num_layers,
            self.decoder.config.num_attention_heads,
        )
