import pandas as pd
import torch


class Tokenizer:
    def __init__(
        self,
        use_gpt: bool,
        file_path="./data/dataset5/step2/dict_id2word.pkl",
    ):
        """Initializes the tokenizer."""
        self.dict_id2word = pd.read_pickle(file_path)
        self.use_gpt = use_gpt
        self.vocab_size = len(self.dict_id2word)
        self.start_token_id = self.vocab_size if not use_gpt else None
        self.end_token_id = self.vocab_size if use_gpt else self.vocab_size + 1
        self.pad_token_id = self.end_token_id + 1
        self.max_label_length = 151

    def encode(self, tokens: list):
        encoded_tokens = torch.tensor(tokens, dtype=torch.long)
        len_label = len(encoded_tokens)
        dif = self.max_label_length - len_label
        encoded_tokens = torch.cat(
            (
                encoded_tokens,
                torch.tensor([self.end_token_id], dtype=torch.long),
                torch.full((dif,), self.pad_token_id, dtype=torch.long),
            )
        )
        if not self.use_gpt:
            encoded_tokens = torch.cat(
                [torch.tensor([self.start_token_id], dtype=torch.long), encoded_tokens]
            )
        return encoded_tokens

    def decode(self, token_ids):
        """
        Decodes a list of token IDs back to a string.

        :param token_ids: List of integers representing token IDs.
        :return: Decoded string.
        """
        return "".join(self.decode_seq(token_ids))

    def decode_seq(self, token_ids):
        return [
            self.dict_id2word[id.item()]
            for id in token_ids
            if id.item() in self.dict_id2word
        ]

    def get_vocab_size(self):
        return self.vocab_size + 2 if self.use_gpt else self.vocab_size + 3

    def get_max_label_length(self):
        return self.max_label_length
