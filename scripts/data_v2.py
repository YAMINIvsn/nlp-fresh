"""One unified loader for the v2 benchmark. Every script should use this."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "v2"
LABELS = ["Compliant", "Non-Compliant", "Ambiguous"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_split(name: str) -> list:
    """name in {'train','val','test','all'} -> list of record dicts."""
    with open(DATA_DIR / f"{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)