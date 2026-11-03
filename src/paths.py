from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
PARQUET = DATA / "wayuu.parquet"
PAIRS = DATA / "wayuu_pairs.csv"
DICTIONARY = ARTIFACTS / "dictionary.json"
VOCAB = ARTIFACTS / "vocab.json"
MODEL = ARTIFACTS / "model.pt"
