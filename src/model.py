import json

import torch
import torch.nn as nn

from src.paths import MODEL, VOCAB

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>", "<to-guc>", "<to-es>"]
TO_GUC = 4
TO_ES = 5
MAX_LEN = 48


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 64, hid: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.encoder = nn.GRU(emb, hid, batch_first=True)
        self.decoder = nn.GRU(emb, hid, batch_first=True)
        self.out = nn.Linear(hid, vocab_size)

    def encode(self, source: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(source)
        lengths = (source != PAD).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)
        return hidden

    def forward(self, source: torch.Tensor, target_in: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(source)
        output, _ = self.decoder(self.embedding(target_in), hidden)
        return self.out(output)

    def generate(self, source: torch.Tensor, limit: int = MAX_LEN) -> list[int]:
        hidden = self.encode(source)
        token = torch.tensor([[SOS]])
        produced: list[int] = []
        for _ in range(limit):
            output, hidden = self.decoder(self.embedding(token), hidden)
            token = self.out(output[:, -1]).argmax(dim=-1, keepdim=True)
            value = int(token.item())
            if value in (EOS, PAD, SOS):
                break
            if value not in (UNK, TO_GUC, TO_ES):
                produced.append(value)
        return produced


class Vocabulary:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {index: token for token, index in stoi.items()}

    @classmethod
    def build(cls, texts: list[str]) -> "Vocabulary":
        stoi = {token: index for index, token in enumerate(SPECIALS)}
        for text in texts:
            for char in text:
                if char not in stoi:
                    stoi[char] = len(stoi)
        return cls(stoi)

    def save(self) -> None:
        VOCAB.write_text(json.dumps(self.stoi, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls) -> "Vocabulary":
        return cls(json.loads(VOCAB.read_text(encoding="utf-8")))

    def encode_source(self, text: str, direction: str) -> list[int]:
        marker = TO_GUC if direction == "es>guc" else TO_ES
        ids = [marker]
        ids.extend(self.stoi.get(char, UNK) for char in text)
        ids.append(EOS)
        return ids[:MAX_LEN]

    def encode_target(self, text: str) -> list[int]:
        ids = [SOS]
        ids.extend(self.stoi.get(char, UNK) for char in text)
        ids.append(EOS)
        return ids[: MAX_LEN + 1]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[index] for index in ids if index in self.itos)


def save_model(network: Seq2Seq) -> None:
    torch.save(network.state_dict(), MODEL)


def load_model(vocab_size: int) -> Seq2Seq:
    network = Seq2Seq(vocab_size)
    network.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
    network.eval()
    return network
