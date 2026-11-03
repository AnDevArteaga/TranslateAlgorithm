import csv
import random

import torch
import torch.nn as nn

from src.clean import prepare
from src.model import PAD, Seq2Seq, Vocabulary, save_model
from src.paths import PAIRS


def read_pairs() -> list[tuple[str, str]]:
    with PAIRS.open(encoding="utf-8") as handle:
        return [(row["guc"], row["spa"]) for row in csv.DictReader(handle)]


def batches(pairs, vocab: Vocabulary, batch_size: int):
    random.shuffle(pairs)
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        sources = [vocab.encode_source(source, direction) for source, _, direction in chunk]
        targets = [vocab.encode_target(target) for _, target, _ in chunk]
        yield pad(sources), pad(targets)


def pad(rows: list[list[int]]) -> torch.Tensor:
    width = max(len(row) for row in rows)
    tensor = torch.full((len(rows), width), PAD, dtype=torch.long)
    for index, row in enumerate(rows):
        tensor[index, : len(row)] = torch.tensor(row)
    return tensor


def main() -> None:
    torch.manual_seed(7)
    random.seed(7)
    dictionary = prepare()
    directed: list[tuple[str, str, str]] = []
    for guc, spa in read_pairs():
        directed.append((spa, guc, "es>guc"))
        directed.append((guc, spa, "guc>es"))

    vocab = Vocabulary.build([source + target for source, target, _ in directed])
    vocab.save()
    network = Seq2Seq(len(vocab.stoi))
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    epochs = 16
    for epoch in range(1, epochs + 1):
        network.train()
        total = 0.0
        seen = 0
        for source, target in batches(directed, vocab, 128):
            optimizer.zero_grad()
            logits = network(source, target[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * source.size(0)
            seen += source.size(0)
        print(f"epoca {epoch}/{epochs}  perdida {total / seen:.4f}")

    network.eval()
    save_model(network)
    print(f"pares {dictionary['pairs']}  entradas es {len(dictionary['es_to_guc'])}")


if __name__ == "__main__":
    main()
