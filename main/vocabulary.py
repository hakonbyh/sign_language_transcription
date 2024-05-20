import logging
from collections import Counter, defaultdict
from typing import List

import numpy as np
from torchtext.data import Dataset

logger = logging.getLogger(__name__)

SIL_TOKEN = "<si>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


class Vocabulary:

    def __init__(self):
        self.specials = []
        self.itos = []
        self.stoi = None
        self.DEFAULT_UNK_ID = None

    def _from_list(self, tokens: List[str] = None, fast: bool = False):
        self.add_tokens(tokens=self.specials + tokens, fast=fast)
        assert len(self.stoi) == len(self.itos)

    def _from_file(self, file: str):
        logger.info(f"Loading vocabulary from {file}...")
        tokens = []
        with open(file, "r", encoding="utf-8") as open_file:
            lines = open_file.readlines()
            logger.info(f"Vocabulary contains {len(lines)} entries")
            for line in lines:
                tokens.append(line.strip("\n"))
        self._from_list(tokens, fast=True)
        logger.info("Vocabulary loaded")

    def __str__(self) -> str:
        return self.stoi.__str__()

    def to_file(self, file: str):
        with open(file, "w", encoding="utf-8") as open_file:
            for t in self.itos:
                open_file.write("{}\n".format(t))

    def add_tokens(self, tokens: List[str], fast: bool = False):
        for t in tokens:
            new_index = len(self.itos)
            if fast or t not in self.itos:
                self.itos.append(t)
                self.stoi[t] = new_index

    def is_unk(self, token: str) -> bool:
        return self.stoi[token] == self.DEFAULT_UNK_ID()

    def __len__(self) -> int:
        return len(self.itos)


class TextVocabulary(Vocabulary):
    def __init__(
        self, tokens: List[str] = None, file: str = None, mBartVocab: bool = False
    ):
        super().__init__()
        if not mBartVocab:
            self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
        else:
            self.specials = []
            assert file is not None
        self.DEFAULT_UNK_ID = lambda: 0 if not mBartVocab else 3
        self.stoi = defaultdict(self.DEFAULT_UNK_ID)

        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

    def array_to_sentence(self, array: np.array, cut_at_eos=True) -> List[str]:
        sentence = []
        for i in array:
            s = self.itos[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            sentence.append(s)
        return sentence

    def arrays_to_sentences(self, arrays: np.array, cut_at_eos=True) -> List[List[str]]:
        sentences = []
        for array in arrays:
            sentences.append(self.array_to_sentence(array=array, cut_at_eos=cut_at_eos))
        return sentences


class GlossVocabulary(Vocabulary):
    def __init__(self, tokens: List[str] = None, file: str = None):
        super().__init__()
        self.specials = [SIL_TOKEN, UNK_TOKEN, PAD_TOKEN]
        self.DEFAULT_UNK_ID = lambda: 1
        self.stoi = defaultdict(self.DEFAULT_UNK_ID)

        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

        assert self.stoi[SIL_TOKEN] == 0

    def arrays_to_sentences(self, arrays: np.array) -> List[List[str]]:
        gloss_sequences = []
        for array in arrays:
            sequence = []
            for i in array:
                sequence.append(self.itos[i])
            gloss_sequences.append(sequence)
        return gloss_sequences


def filter_min(counter: Counter, minimum_freq: int):
    filtered_counter = Counter({t: c for t, c in counter.items() if c >= minimum_freq})
    return filtered_counter


def sort_and_cut(counter: Counter, limit: int):
    tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
    return vocab_tokens


def build_vocab(
    field: str,
    max_size: int,
    min_freq: int,
    dataset: Dataset,
    vocab_file: str = None,
    mBartVocab: bool = False,
) -> Vocabulary:

    if vocab_file is not None:
        if field == "gls":
            vocab = GlossVocabulary(file=vocab_file)
        elif field == "txt":
            vocab = TextVocabulary(file=vocab_file, mBartVocab=mBartVocab)
        else:
            raise ValueError("Unknown vocabulary type")
    else:
        tokens = []
        for i in dataset.examples:
            if field == "gls":
                tokens.extend(i.gls)
            elif field == "txt":
                tokens.extend(i.txt)
            else:
                raise ValueError("Unknown field type")

        counter = Counter(tokens)
        if min_freq > -1:
            counter = filter_min(counter, min_freq)
        vocab_tokens = sort_and_cut(counter, max_size)
        assert len(vocab_tokens) <= max_size

        if field == "gls":
            vocab = GlossVocabulary(tokens=vocab_tokens)
        elif field == "txt":
            vocab = TextVocabulary(tokens=vocab_tokens, mBartVocab=mBartVocab)
        else:
            raise ValueError("Unknown vocabulary type")

        assert len(vocab) <= max_size + len(vocab.specials)
        assert vocab.itos[vocab.DEFAULT_UNK_ID()] == UNK_TOKEN

    for i, s in enumerate(vocab.specials):
        if i != vocab.DEFAULT_UNK_ID():
            assert not vocab.is_unk(s)

    return vocab
