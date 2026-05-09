"""
dataset.py — Multi30k Dataset Loading & Preprocessing
DA6401 Assignment 3

Source  : https://huggingface.co/datasets/bentrevett/multi30k
Tokenization : spaCy (de_core_news_sm for German, en_core_web_sm for English)

Special token indices (fixed):
    <unk> = 0   <pad> = 1   <sos> = 2   <eos> = 3

Usage:
    train_ds = Multi30kDataset('train')
    train_ds.build_vocab(min_freq=2)
    train_ds.process_data()

    val_ds = Multi30kDataset('validation')
    val_ds.src_vocab = train_ds.src_vocab
    val_ds.tgt_vocab = train_ds.tgt_vocab
    val_ds.process_data()

    loader = DataLoader(train_ds, batch_size=128, collate_fn=collate_fn)
"""

from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy


# ── Special token constants ────────────────────────────────────────────
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocab:
    """
    Simple vocabulary object.

    Attributes:
        stoi : dict  token → index
        itos : list  index → token
    """

    def __init__(self, stoi: dict, itos: list) -> None:
        self.stoi = stoi
        self.itos = itos

    def __len__(self) -> int:
        return len(self.itos)

    def lookup_token(self, idx: int) -> str:
        """Return the token at index idx (supports both access styles)."""
        return self.itos[idx]

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        """Map a list of tokens to their indices (unknown → UNK_IDX)."""
        unk = self.stoi.get("<unk>", UNK_IDX)
        return [self.stoi.get(t, unk) for t in tokens]


# ══════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    def __init__(self, split: str = "train"):
        """
        Loads the Multi30k dataset and prepares tokenizers.

        Args:
            split : 'train', 'validation', or 'test'
        """
        self.split = split

        # Load dataset from Hugging Face
        raw = load_dataset("bentrevett/multi30k")

        # Map split name variants
        split_key = "validation" if split in ("val", "validation") else split
        self.data = raw[split_key]

        # Load spaCy tokenizers
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            raise OSError(
                "German spaCy model not found. "
                "Run: python -m spacy download de_core_news_sm"
            )
        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "English spaCy model not found. "
                "Run: python -m spacy download en_core_web_sm"
            )

        # Vocab and processed data set by build_vocab / process_data
        self.src_vocab: Vocab = None
        self.tgt_vocab: Vocab = None
        self.processed_data: List[Tuple[torch.Tensor, torch.Tensor]] = []

    # ── Tokenisers ────────────────────────────────────────────────────

    def tokenize_de(self, text: str) -> List[str]:
        """Lowercase German tokenisation via spaCy."""
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str) -> List[str]:
        """Lowercase English tokenisation via spaCy."""
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    # ── Vocab building ────────────────────────────────────────────────

    def build_vocab(self, min_freq: int = 2) -> None:
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>

        Always built from the *training* split regardless of self.split,
        so val/test datasets get the correct vocab when they share it.

        Args:
            min_freq : Minimum token frequency to include in vocab (default 2).
        """
        raw_train = load_dataset("bentrevett/multi30k")["train"]

        counter_de: Counter = Counter()
        counter_en: Counter = Counter()

        for example in raw_train:
            counter_de.update(self.tokenize_de(example["de"]))
            counter_en.update(self.tokenize_en(example["en"]))

        # Build German (source) vocab: specials first, then by frequency
        de_tokens = SPECIALS + [
            tok for tok, freq in counter_de.most_common()
            if freq >= min_freq
        ]
        de_stoi = {tok: idx for idx, tok in enumerate(de_tokens)}
        self.src_vocab = Vocab(de_stoi, de_tokens)

        # Build English (target) vocab
        en_tokens = SPECIALS + [
            tok for tok, freq in counter_en.most_common()
            if freq >= min_freq
        ]
        en_stoi = {tok: idx for idx, tok in enumerate(en_tokens)}
        self.tgt_vocab = Vocab(en_stoi, en_tokens)

    # ── Data processing ───────────────────────────────────────────────

    def process_data(self) -> None:
        """
        Convert English and German sentences into integer token lists using
        spaCy and the defined vocabulary.

        Wraps each sequence with <sos> and <eos>.
        self.src_vocab and self.tgt_vocab must be set before calling this.
        """
        assert self.src_vocab is not None and self.tgt_vocab is not None, (
            "Call build_vocab() first (or assign src_vocab / tgt_vocab "
            "from a training dataset instance)."
        )

        self.processed_data = []
        unk_src = self.src_vocab.stoi.get("<unk>", UNK_IDX)
        unk_tgt = self.tgt_vocab.stoi.get("<unk>", UNK_IDX)

        for example in self.data:
            src_tokens = self.tokenize_de(example["de"])
            tgt_tokens = self.tokenize_en(example["en"])

            src_ids = (
                [SOS_IDX]
                + [self.src_vocab.stoi.get(t, unk_src) for t in src_tokens]
                + [EOS_IDX]
            )
            tgt_ids = (
                [SOS_IDX]
                + [self.tgt_vocab.stoi.get(t, unk_tgt) for t in tgt_tokens]
                + [EOS_IDX]
            )

            self.processed_data.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )
            )

    # ── Dataset protocol ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.processed_data[idx]


# ══════════════════════════════════════════════════════════════════════
#  COLLATE FUNCTION  (for DataLoader)
# ══════════════════════════════════════════════════════════════════════

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = PAD_IDX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length sequences in a batch to the same length.

    Args:
        batch   : list of (src_tensor, tgt_tensor) pairs
        pad_idx : padding value (default PAD_IDX = 1)

    Returns:
        src_padded : [batch_size, max_src_len]
        tgt_padded : [batch_size, max_tgt_len]
    """
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


# ══════════════════════════════════════════════════════════════════════
#  CONVENIENCE BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_datasets(min_freq: int = 2):
    """
    Build train / val / test Multi30kDataset objects sharing the same vocab.

    Returns:
        train_ds, val_ds, test_ds : Multi30kDataset instances
    """
    train_ds = Multi30kDataset("train")
    train_ds.build_vocab(min_freq=min_freq)
    train_ds.process_data()

    val_ds = Multi30kDataset("validation")
    val_ds.src_vocab = train_ds.src_vocab
    val_ds.tgt_vocab = train_ds.tgt_vocab
    val_ds.process_data()

    test_ds = Multi30kDataset("test")
    test_ds.src_vocab = train_ds.src_vocab
    test_ds.tgt_vocab = train_ds.tgt_vocab
    test_ds.process_data()

    return train_ds, val_ds, test_ds
