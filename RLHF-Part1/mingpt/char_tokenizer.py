import torch


class CharTokenizer:
    pad_char = "␣"
    eot_char = "⏎"

    def __init__(self, data):
        chars = sorted(list(set(data) | {self.pad_char, self.eot_char}))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.pad_token = self.stoi[self.pad_char]
        self.eot_token = self.stoi[self.eot_char]

    def __call__(self, text: str) -> torch.LongTensor:
        # encode every character to an integer
        return torch.tensor([self.stoi[s] for s in text if s in self.stoi], dtype=torch.long)

    def decode(self, tokens, ignore_padding=False) -> str:
        return "".join(self.itos[i] for i in tokens.tolist() if i in self.itos and not ignore_padding or i != self.pad_token)
