import logging
import os
import random
import socket
import sys

import numpy as np
import torch
from main import print_cuda_memory
from main.dataset import SignTranslationDataset, load_dataset_file
from main.vocabulary import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    Vocabulary,
    build_vocab,
)
from torchtext import data
from torchtext.data import Dataset, Iterator
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGENET_MEAN = 0.485, 0.456, 0.406
IMAGENET_STD = 0.229, 0.224, 0.225


def load_data(
    data_cfg: dict, keep_only=None
) -> (Dataset, Dataset, Vocabulary, Vocabulary):

    if keep_only is None:
        keep_only = {"train": None, "dev": None, "test": None}

    data_path = data_cfg.get("data_path", "./data")

    level = data_cfg["level"]
    txt_lowercase = data_cfg["txt_lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    def tokenize_text(text):
        if level == "char":
            return list(text)
        else:
            return text.split()

    sequence_field = data.RawField()

    mBartVocab = False

    sgn_field = data.RawField()

    sgn_len_field = data.RawField()

    gls_field = data.Field(
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        batch_first=True,
        lower=False,
        include_lengths=True,
    )

    txt_field = data.Field(
        init_token=BOS_TOKEN if not mBartVocab else None,
        eos_token=(EOS_TOKEN if not mBartVocab else None),
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=txt_lowercase,
        include_lengths=True,
    )

    txt_field_pred = data.Field(
        init_token=(BOS_TOKEN if not mBartVocab else EOS_TOKEN),
        eos_token=(EOS_TOKEN if not mBartVocab else None),
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=txt_lowercase,
        include_lengths=True,
    )

    train_files, train_val_files, dev_files = load_dataset_file(data_path, 10)

    train_data = SignTranslationDataset(
        files=train_files,
        fields=(sequence_field, sgn_field, sgn_len_field, gls_field, txt_field),
        filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
        and len(vars(x)["txt"]) <= max_sent_length,
        keep_only=keep_only["train"],
    )

    train_val_data = SignTranslationDataset(
        files=train_val_files,
        fields=(sequence_field, sgn_field, sgn_len_field, gls_field, txt_field),
        filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
        and len(vars(x)["txt"]) <= max_sent_length,
        keep_only=keep_only["train"],
    )

    gls_max_size = data_cfg.get("gls_voc_limit", sys.maxsize)
    gls_min_freq = data_cfg.get("gls_voc_min_freq", 1)
    txt_max_size = data_cfg.get("txt_voc_limit", sys.maxsize)
    txt_min_freq = data_cfg.get("txt_voc_min_freq", 1)

    gls_vocab_file = data_cfg.get("gls_vocab", None)
    txt_vocab_file = data_cfg.get("txt_vocab", None)

    gls_vocab = build_vocab(
        field="gls",
        min_freq=gls_min_freq,
        max_size=gls_max_size,
        dataset=train_data,
        vocab_file=gls_vocab_file,
    )

    txt_vocab = build_vocab(
        field="txt",
        min_freq=txt_min_freq,
        max_size=txt_max_size,
        dataset=train_data,
        vocab_file=txt_vocab_file,
        mBartVocab=mBartVocab,
    )

    dev_data = SignTranslationDataset(
        files=dev_files,
        fields=(sequence_field, sgn_field, sgn_len_field, gls_field, txt_field_pred),
        keep_only=keep_only["dev"],
    )

    gls_field.vocab = gls_vocab
    txt_field.vocab = txt_vocab
    txt_field_pred.vocab = txt_vocab
    return train_data, train_val_data, dev_data, gls_vocab, txt_vocab


global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch


def token_batch_size_fn(new, count, sofar):
    global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
    if count == 1:
        max_sgn_in_batch = 0
        max_gls_in_batch = 0
        max_txt_in_batch = 0
    max_sgn_in_batch = max(max_sgn_in_batch, len(new.sgn))
    max_gls_in_batch = max(max_gls_in_batch, len(new.gls))
    max_txt_in_batch = max(max_txt_in_batch, len(new.txt) + 2)
    sgn_elements = count * max_sgn_in_batch
    gls_elements = count * max_gls_in_batch
    txt_elements = count * max_txt_in_batch
    return max(sgn_elements, gls_elements, txt_elements)


def make_data_iter(
    dataset: Dataset,
    batch_size: int,
    batch_type: str = "sentence",
    train: bool = False,
    shuffle: bool = False,
) -> Iterator:

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        data_iter = data.BucketIterator(
            repeat=False,
            sort=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=True,
            sort_within_batch=True,
            sort_key=lambda x: x.sgn_len,
            shuffle=shuffle,
        )
    else:
        data_iter = data.BucketIterator(
            repeat=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=False,
            sort=False,
        )

    return data_iter
