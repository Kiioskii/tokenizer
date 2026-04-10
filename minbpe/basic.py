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
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.inverse_byte_shuffle = None
        self.pattern = GPT4_SPLIT_PATTERN
        self.special_tokens = GPT4_SPECIAL_TOKENS
        self.inverse_special_tokens = {
            v: k for k, v in self.special_tokens.items()
        }
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
        byte_stream = bytearray()
        text_parts = []

        for idx in ids:
            if idx in self.inverse_special_tokens:
                if byte_stream:
                    text_parts.append(
                        bytes(byte_stream).decode("utf-8", errors="replace")
                    )
                    byte_stream.clear()

                text_parts.append(self.inverse_special_tokens[idx])
                continue

            token_bytes = self.vocab[idx]

            if self.inverse_byte_shuffle:
                token_bytes = bytes(self.inverse_byte_shuffle[b] for b in token_bytes)

            byte_stream.extend(token_bytes)

        if byte_stream:
            text_parts.append(
                bytes(byte_stream).decode("utf-8", errors="replace")
            )

        return "".join(text_parts)

    def encode(self, text, allowed_special=set()):
        if allowed_special == "all":
            allowed_special = set(self.special_tokens.keys())

        result = []
        i = 0

        while i < len(text):
            matched = False
            for token, token_id in self.special_tokens.items():
                if text.startswith(token, i):
                    if token not in allowed_special:
                        raise ValueError(f"Special token {token} not allowed")

                    result.append(token_id)
                    i += len(token)
                    matched = True
                    break

            if matched:
                continue

            j = i
            while j < len(text):
                if any(text.startswith(tok, j) for tok in self.special_tokens):
                    break
                j += 1

            chunk = text[i:j]
            chunks = self._split(chunk)

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

            i = j

        return result
