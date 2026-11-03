import csv
import json
import re
import unicodedata

import pandas as pd

from src.paths import ARTIFACTS, DICTIONARY, PAIRS, PARQUET

FUNCTION_WORDS = {
    "a",
    "al",
    "con",
    "de",
    "del",
    "el",
    "en",
    "estar",
    "la",
    "las",
    "lo",
    "los",
    "para",
    "por",
    "que",
    "se",
    "ser",
    "un",
    "una",
    "y",
}

SHORT_WORDS = {"a", "o", "y"}
POS = r"v_t|v_i|v_e|v_r|n_pos|adj|adv|conj|posp|pron|interj|part|pl|s|n|v"
POS_LINE = re.compile(rf"^(?:[—\-–*]+\s*)?(?:{POS})\b", re.IGNORECASE)
HEADWORD_NOTE = re.compile(r"\s*\(\d+\)\s*")
POS_SUFFIX = re.compile(r"\s+v\.[a-z.]+\s*,?\s*$", re.IGNORECASE)
PHONETIC = re.compile(r"^[a-záéíóúñ] no [a-záéíóúñ]\b", re.IGNORECASE)
PAREN = re.compile(r"\([^)]*\)")
SPACE = re.compile(r"\s+")
EDGE = "¡!¿?.,;:\"“”"


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("ʻ", "'")
        .replace("`", "'")
    )
    return SPACE.sub(" ", text).strip().lower()


def headwords(value: str) -> list[str]:
    text = HEADWORD_NOTE.sub(" ", value or "")
    text = POS_SUFFIX.sub("", text)
    forms = []
    for part in text.split(","):
        form = normalize(part).strip(EDGE)
        if not form or form.startswith("-"):
            continue
        if not re.search(r"[a-záéíóúñü']", form):
            continue
        forms.append(form)
    return list(dict.fromkeys(forms))


def _sense(chunk: str) -> str | None:
    chunk = re.sub(r"(?:^|\s)-[a-záéíóúñ]{1,8}(?=\s|$)", " ", chunk, flags=re.IGNORECASE)
    chunk = re.sub(rf"(?:^|\s)\*?(?:{POS})(?=\s|$)", " ", chunk, flags=re.IGNORECASE)
    text = SPACE.sub(" ", chunk).strip().strip(EDGE)
    if not text or POS_LINE.match(text) or PHONETIC.match(text):
        return None
    if "ü" in text or "'" in text:
        return None
    if len(text) == 1 and text not in SHORT_WORDS:
        return None
    if len(text.split()) > 12:
        return None
    if not re.search(r"[a-záéíóúñ]", text):
        return None
    return text


def spanish_senses(value: str) -> list[str]:
    text = normalize(value)
    if not text:
        return []
    senses = []
    for chunk in text.split(";"):
        sense = _sense(chunk)
        if sense and sense not in senses:
            senses.append(sense)
    return senses


def spanish_keys(gloss: str) -> list[str]:
    bare = SPACE.sub(" ", PAREN.sub(" ", gloss)).strip()
    keys = [gloss]
    if bare and bare != gloss:
        keys.append(bare)
    words = bare.split()
    if words and not any(word in FUNCTION_WORDS for word in words):
        keys.extend(words)
    return list(dict.fromkeys(keys))


def build(frame: pd.DataFrame) -> tuple[list[dict[str, str]], dict]:
    known: set[str] = set()
    raw_rows: list[tuple[str, str]] = []
    for raw in frame["translation"]:
        item = raw if isinstance(raw, dict) else json.loads(raw)
        forms = headwords(str(item.get("guc", "")))
        senses = spanish_senses(str(item.get("spa", "")))
        known.update(forms)
        for form in forms:
            raw_rows.append((form, "\n".join(senses)))

    pairs: list[dict[str, str]] = []
    es_to_guc: dict[str, list[str]] = {}
    guc_to_es: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for form, packed in raw_rows:
        for gloss in packed.split("\n"):
            if not gloss or gloss in known or all(word in known for word in gloss.split()):
                continue
            key = (form, gloss)
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"guc": form, "spa": gloss})
            guc_to_es.setdefault(form, [])
            if gloss not in guc_to_es[form]:
                guc_to_es[form].append(gloss)
            for spanish in spanish_keys(gloss):
                es_to_guc.setdefault(spanish, [])
                if form not in es_to_guc[spanish]:
                    es_to_guc[spanish].append(form)

    dictionary = {
        "source": "Gaxys/wayuu_spa_dict",
        "es_to_guc": es_to_guc,
        "guc_to_es": guc_to_es,
        "pairs": len(pairs),
    }
    return pairs, dictionary


def write_outputs(pairs: list[dict[str, str]], dictionary: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with PAIRS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["guc", "spa"])
        writer.writeheader()
        writer.writerows(pairs)
    DICTIONARY.write_text(
        json.dumps(dictionary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def prepare() -> dict:
    frame = pd.read_parquet(PARQUET)
    pairs, dictionary = build(frame)
    write_outputs(pairs, dictionary)
    return dictionary
