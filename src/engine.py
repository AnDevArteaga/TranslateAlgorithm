import json
import re

import torch

from src.clean import normalize
from src.model import Vocabulary, load_model
from src.paths import DICTIONARY

PUNCTUATION = ".,;:!?¿¡\"“”()[]"


class TranslateError(Exception):
    def __init__(self, message: str, status: int):
        self.message = message
        self.status = status


class Engine:
    def __init__(self):
        self.dictionary = json.loads(DICTIONARY.read_text(encoding="utf-8"))
        self.vocab = Vocabulary.load()
        self.model = load_model(len(self.vocab.stoi))

    def translate(self, text: str, source: str, target: str) -> dict:
        source = (source or "").strip().lower()
        target = (target or "").strip().lower()
        if source not in {"es", "guc", "embera"} or target not in {"es", "guc", "embera"}:
            raise TranslateError("Idioma no reconocido. Use es, guc o embera.", 400)
        if source == "embera" or target == "embera":
            raise TranslateError(
                "Emberá todavía no tiene un corpus. El servicio no inventa palabras.",
                422,
            )
        if source == target:
            raise TranslateError("El idioma de origen y el de destino son el mismo.", 400)
        cleaned = normalize(text)
        if not cleaned:
            raise TranslateError("Escriba una palabra o una frase corta.", 400)

        direction = "es>guc" if source == "es" else "guc>es"
        table = self.dictionary["es_to_guc" if source == "es" else "guc_to_es"]
        tokens = [piece for piece in (token.strip(PUNCTUATION) for token in cleaned.split()) if piece]
        if len(tokens) > 1 and cleaned in table:
            tokens = [cleaned]

        parts = [self._one(token, table, direction) for token in tokens]
        origins = {part["origin"] for part in parts}
        return {
            "source": source,
            "target": target,
            "origin": origins.pop() if len(origins) == 1 else "mixed",
            "approximate": any(part["approximate"] for part in parts),
            "parts": parts,
        }

    def _one(self, token: str, table: dict, direction: str) -> dict:
        matches = table.get(token)
        if matches:
            return {
                "input": token,
                "origin": "dictionary",
                "approximate": False,
                "matches": matches,
            }
        suggestion = self._generate(token, direction)
        return {
            "input": token,
            "origin": "model",
            "approximate": True,
            "matches": [suggestion] if suggestion else [],
        }

    def _generate(self, token: str, direction: str) -> str:
        ids = self.vocab.encode_source(token, direction)
        with torch.no_grad():
            produced = self.model.generate(torch.tensor([ids]))
        text = self.vocab.decode(produced).strip()
        return re.sub(r"\s+", " ", text)
