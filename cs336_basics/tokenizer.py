import os
import time
import heapq
import regex as re

from typing import BinaryIO
from collections import defaultdict
from functools import wraps
from multiprocessing import Pool, cpu_count

import pickle
from collections import defaultdict, Counter
from cs336_basics.tokenization_utils import find_chunk_boundaries, pretokenize_chunk, train_bpe


class Tokenizer:

    def __init__(self, vocab=None, merges=None, split_pattern="gpt2", special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.merge_rules = {pair: idx for idx, pair in enumerate(merges)}

        self._gpt2_split_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._gpt4_split_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            self.vocab = pickle.load(f)

        with open(merges_filepath, 'rb') as f:
            self.merges = pickle.load(f)

        self.special_tokens = special_tokens

        return cls(self.vocab, self.merges, self.special_tokens)

    def encode(self, text: str) -> list[int]:
        tokens = [match.group() for match in re.finditer(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", text)]
        ids = []
        for token in tokens:
            token_bytes = [bytes([b]) for b in token.encode('utf-8')]
            while len(token_bytes) > 1:
                pairs = [(token_bytes[i], token_bytes[i+1]) for i in range(len(token_bytes)-1)]
                pair_to_merge = min(
                    (pair for pair in pairs if pair in self.merges),
                    key=lambda pair: self.merges.index(pair),
                    default=None
                )
                if pair_to_merge is None:
                    break
                i = pairs.index(pair_to_merge)
                token_bytes[i:i+2] = [pair_to_merge[0] + pair_to_merge[1]]
            ids.extend(self.token_to_id[b] for b in token_bytes)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')

    @classmethod
    def train(cls, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.vocab, self.merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            num_processes=None,
            use_naive_merge=False
        )

        return cls(vocab, merges, special_tokens)
