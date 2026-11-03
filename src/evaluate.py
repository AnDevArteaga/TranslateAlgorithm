import random

import torch
import torch.nn as nn

from src.clean import spanish_keys
from src.model import Seq2Seq, Vocabulary
from src.train import batches, pad, read_pairs


def index(pairs: list[tuple[str, str]]) -> tuple[dict, dict]:
    es_to_guc: dict[str, list[str]] = {}
    guc_to_es: dict[str, list[str]] = {}
    for guc, spa in pairs:
        guc_to_es.setdefault(guc, [])
        if spa not in guc_to_es[guc]:
            guc_to_es[guc].append(spa)
        for key in spanish_keys(spa):
            es_to_guc.setdefault(key, [])
            if guc not in es_to_guc[key]:
                es_to_guc[key].append(guc)
    return es_to_guc, guc_to_es


def distance(left: str, right: str) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char != other),
                )
            )
        previous = current
    return previous[-1]


def character_score(prediction: str, gold: str) -> float:
    longest = max(len(prediction), len(gold), 1)
    return 1 - distance(prediction, gold) / longest


def outcome(senses: list[str] | None, gold: str, prediction: str) -> dict:
    covered = senses is not None
    dictionary_hit = bool(senses and gold in senses)
    return {
        "covered": covered,
        "dictionary_hit": dictionary_hit,
        "model_exact": prediction == gold,
        "characters": character_score(prediction, gold),
        "correct": dictionary_hit or (not covered and prediction == gold),
    }


def train_network(pairs: list[tuple[str, str]]) -> tuple[Seq2Seq, Vocabulary]:
    directed = []
    for guc, spa in pairs:
        directed.append((spa, guc, "es>guc"))
        directed.append((guc, spa, "guc>es"))
    vocab = Vocabulary.build([source + target for source, target, _ in directed])
    network = Seq2Seq(len(vocab.stoi))
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(1, 17):
        network.train()
        for source, target in batches(directed, vocab, 128):
            optimizer.zero_grad()
            logits = network(source, target[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
        print(f"epoca {epoch}/16")
    network.eval()
    return network, vocab


def predict(network: Seq2Seq, vocab: Vocabulary, text: str, direction: str) -> str:
    ids = vocab.encode_source(text, direction)
    with torch.no_grad():
        produced = network.generate(torch.tensor([ids]))
    return vocab.decode(produced).strip()


def split_pairs(pairs: list[tuple[str, str]], holdout: float = 0.1):
    order = list(pairs)
    random.Random(7).shuffle(order)
    cut = max(1, round(len(order) * holdout))
    return order[cut:], order[:cut]


def measure(train: list[tuple[str, str]], test: list[tuple[str, str]], network, vocab) -> list[dict]:
    es_to_guc, guc_to_es = index(train)
    rows = []
    for guc, spa in test:
        for direction, source, gold, senses in (
            ("es>guc", spa, guc, es_to_guc.get(spa)),
            ("guc>es", guc, spa, guc_to_es.get(guc)),
        ):
            prediction = predict(network, vocab, source, direction)
            rows.append(
                {
                    "direction": direction,
                    "source": source,
                    "gold": gold,
                    "prediction": prediction,
                    "known_form": senses is not None,
                    "exact": prediction == gold,
                    "characters": character_score(prediction, gold),
                }
            )
    return rows


def report(rows: list[dict], label: str) -> None:
    if not rows:
        return
    exact = 100 * sum(row["exact"] for row in rows) / len(rows)
    characters = 100 * sum(row["characters"] for row in rows) / len(rows)
    known = 100 * sum(row["known_form"] for row in rows) / len(rows)
    print(f"{label}")
    print(f"  casos {len(rows)}")
    print(f"  acierto exacto del modelo {exact:.1f}%")
    print(f"  cercanía por caracteres {characters:.1f}%")
    print(f"  la forma ya tenía otro sentido guardado {known:.1f}%")
    for row in rows[:3]:
        print(f"  ejemplo {row['source']} -> {row['prediction']}  (esperado {row['gold']})")


def main() -> None:
    torch.manual_seed(7)
    random.seed(7)
    train, test = split_pairs(read_pairs())
    print(f"entrenamiento {len(train)}  prueba {len(test)}")
    es_to_guc, _ = index(train)
    stored = sum(guc in es_to_guc.get(spa, []) for guc, spa in train)
    print(f"diccionario sobre pares guardados {100 * stored / len(train):.1f}%")
    network, vocab = train_network(train)
    rows = measure(train, test, network, vocab)
    report(rows, "ambos sentidos")
    report([row for row in rows if row["direction"] == "es>guc"], "español a wayuunaiki")
    report([row for row in rows if row["direction"] == "guc>es"], "wayuunaiki a español")


if __name__ == "__main__":
    main()
