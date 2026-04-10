import tiktoken

from minbpe import recover_merges, BasicTokenizer


def main():
    text = "hello world!!!? (안녕하세요!) lol123 😉"

    enc = tiktoken.get_encoding("cl100k_base")
    gpt4_ids = enc.encode(text)
    gpt4_decoded = enc.decode(gpt4_ids)

    # odzyskanie merges i byte shuffle (jeśli potrzebne)
    merges = recover_merges(enc._mergeable_ranks)
    byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}

    print("GPT-4 encode:", gpt4_ids)
    print("GPT-4 decode:", gpt4_decoded)


    tokenizer = BasicTokenizer()
    # trenuj tokenizer na tym samym tekście (lub większym korpusie)
    tokenizer.train(text, vocab_size=512, verbose=False)
    my_ids = tokenizer.encode(text)
    my_decoded = tokenizer.decode(my_ids)

    print("\nBasicTokenizer encode:", my_ids)
    print("BasicTokenizer decode:", my_decoded)

    # =========================
    # porównanie wyników
    # =========================
    encode_match = gpt4_ids == my_ids
    decode_match = gpt4_decoded == my_decoded

    print("\nEncode match:", encode_match)
    print("Decode match:", decode_match)

if __name__ == "__main__":
    main()
