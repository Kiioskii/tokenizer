import tiktoken

from minbpe import BasicTokenizer
from minbpe.gpt4 import recover_merges


def build_tokenizer_from_gpt4(enc):
    tokenizer = BasicTokenizer()

    tokenizer.merges = recover_merges(enc._mergeable_ranks)

    tokenizer.vocab = tokenizer._build_vocab()

    tokenizer.byte_shuffle = {
        i: enc._mergeable_ranks[bytes([i])]
        for i in range(256)
    }

    tokenizer.inverse_byte_shuffle = {
        v: k for k, v in tokenizer.byte_shuffle.items()
    }

    return tokenizer


def run_test(text):
    enc = tiktoken.get_encoding("cl100k_base")

    # GPT-4
    gpt_ids = enc.encode(text)
    gpt_decoded = enc.decode(gpt_ids)

    # Twój tokenizer
    tokenizer = build_tokenizer_from_gpt4(enc)
    my_ids = tokenizer.encode(text)
    my_decoded = tokenizer.decode(my_ids)

    print("TEXT:", text)
    print("\nGPT-4 IDs:", gpt_ids)
    print("MY IDS:   ", my_ids)

    print("GPT DECODE:", gpt_decoded)
    print("MY DECODE:", my_decoded)

    print("\nENCODE MATCH:", gpt_ids == my_ids)
    print("DECODE MATCH:", gpt_decoded == my_decoded)

if __name__ == "__main__":
    run_test("hello world!!!? (안녕하세요!) lol123 😉")
