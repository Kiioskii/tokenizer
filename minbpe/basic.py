"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_stats, merge
import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.inverse_byte_shuffle = None
        self.pattern = GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)

        self.byte_shuffle = None
        self.inv_byte_shuffle = None

    def _split(self, text):
        return self.compiled_pattern.findall(text)

    def load_gpt4(self, mergeable_ranks, recover_merges_fn):
        # merges
        self.merges = recover_merges_fn(mergeable_ranks)
        # byte shuffle
        self.byte_shuffle = {
            i: mergeable_ranks[bytes([i])] for i in range(256)
        }
        self.inv_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

        # vocab
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        chunks = self._split(text)
        ids_list = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            stats = {}
            for ids in ids_list:
                get_stats(ids, stats)

            if not stats:
                break

            pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]
            idx = 256 + i

            ids_list = [merge(ids, pair, idx) for ids in ids_list]
            self.merges[pair] = idx

        self.vocab = self._build_vocab()


    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        unshuffled = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        return unshuffled.decode("utf-8", errors="replace")

    def encode(self, text):
        chunks = self._split(text)
        result = []

        for chunk in chunks:
            raw_bytes = chunk.encode("utf-8")
            if self.byte_shuffle:
                ids = [self.byte_shuffle[b] for b in raw_bytes]
            else:
                ids = list(raw_bytes)

            while True:
                stats = get_stats(ids)
                if not stats:
                    break

                valid_pairs = [p for p in stats if p in self.merges]
                if not valid_pairs:
                    break

                pair = min(valid_pairs, key=lambda p: self.merges[p])
                idx = self.merges[pair]
                ids = merge(ids, pair, idx)

            result.extend(ids)
        return result
