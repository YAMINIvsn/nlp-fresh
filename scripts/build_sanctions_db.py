"""
Build a unified sanctions entity-matching table from OFAC's SDN.CSV,
ADD.CSV, and ALT.CSV files.

OFAC's raw CSVs ship WITHOUT header rows, so column names must be
assigned manually based on OFAC's documented schema. All three files
link together via the entity number (ent_num).

Output: a single JSON file where each entity has its primary name,
all known aliases, and all known addresses in one record — this is
what your rule engine's sanctions-matching logic (KYC-008, KYC-009,
FATF-009) should check incoming customer names against.
"""

import pandas as pd
import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "sanctions"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sanctions_entities.json"

# OFAC's documented column layout for each file (no headers in the raw CSVs)
SDN_COLUMNS = [
    "ent_num", "sdn_name", "sdn_type", "program",
    "title", "call_sign", "vess_type", "tonnage",
    "grt", "vess_flag", "vess_owner", "remarks",
]

ADD_COLUMNS = [
    "ent_num", "add_num", "address",
    "city_state_province_postal_code", "country", "add_remarks",
]

ALT_COLUMNS = [
    "ent_num", "alt_num", "alt_type", "alt_name", "alt_remarks",
]


def load_ofac_csv(filename: str, columns: list) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {path} — download {filename} from "
            f"https://sanctionslistservice.ofac.treas.gov/ and place it in {RAW_DIR}"
        )
    df = pd.read_csv(
        path,
        header=None,
        names=columns,
        encoding="latin-1",   # OFAC files sometimes contain non-UTF8 characters
        na_values=["-0-"],    # OFAC uses "-0-" as its null placeholder
        quotechar='"',
        skipinitialspace=True,
    )
    return df


def build_ofac_entities() -> list:
    sdn = load_ofac_csv("SDN.CSV", SDN_COLUMNS)
    add = load_ofac_csv("ADD.CSV", ADD_COLUMNS)
    alt = load_ofac_csv("ALT.CSV", ALT_COLUMNS)

    print(f"[OFAC] Loaded {len(sdn)} primary SDN entries, {len(add)} addresses, {len(alt)} aliases.")

    # OFAC's real files sometimes end with a stray DOS EOF marker (0x1A /
    # Ctrl-Z) or other malformed trailing row, which pandas reads as an
    # extra row whose ent_num isn't a valid number. Coerce to numeric and
    # drop anything that isn't, so a handful of junk rows don't crash the
    # whole pipeline.
    sdn["ent_num"] = pd.to_numeric(sdn["ent_num"], errors="coerce")
    add["ent_num"] = pd.to_numeric(add["ent_num"], errors="coerce")
    alt["ent_num"] = pd.to_numeric(alt["ent_num"], errors="coerce")

    n_bad_sdn = sdn["ent_num"].isna().sum()
    if n_bad_sdn:
        print(f"[OFAC] Dropping {n_bad_sdn} malformed SDN row(s) with an invalid entity number.")
    sdn = sdn.dropna(subset=["ent_num"])
    add = add.dropna(subset=["ent_num"])
    alt = alt.dropna(subset=["ent_num"])

    sdn["ent_num"] = sdn["ent_num"].astype(int)
    add["ent_num"] = add["ent_num"].astype(int)
    alt["ent_num"] = alt["ent_num"].astype(int)

    # Group aliases and addresses by entity number for fast lookup
    aliases_by_entity = alt.groupby("ent_num")["alt_name"].apply(list).to_dict()
    addresses_by_entity = add.groupby("ent_num")["address"].apply(list).to_dict()

    entities = []
    for _, row in sdn.iterrows():
        ent_num = row["ent_num"]
        entities.append({
            "source": "OFAC",
            "id": f"ofac-{int(ent_num)}",
            "primary_name": str(row["sdn_name"]).strip(),
            "schema": str(row["sdn_type"]) if pd.notna(row["sdn_type"]) else None,
            "program": str(row["program"]) if pd.notna(row["program"]) else None,
            "aliases": [str(a).strip() for a in aliases_by_entity.get(ent_num, [])],
            "addresses": [str(a).strip() for a in addresses_by_entity.get(ent_num, [])],
            "countries": [],
        })

    return entities


def build_opensanctions_entities(filename: str = "opensanctions_targets.csv") -> list:
    """
    Parse OpenSanctions' 'Simplified CSV' export (targets.simple.csv).
    Official column layout (per opensanctions.org/docs/bulk/csv):
        id, schema, name, aliases, birth_date, countries, addresses,
        identifiers, sanctions, phones, emails, program_ids, dataset,
        first_seen, last_seen
    Multi-value fields (aliases, countries, addresses, program_ids) are
    semicolon-separated in the raw file.
    """
    path = RAW_DIR / filename
    if not path.exists():
        print(f"[OpenSanctions] {path} not found — skipping OpenSanctions merge. "
              f"Download targets.simple.csv from opensanctions.org and place it there to include it.")
        return []

    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    print(f"[OpenSanctions] Loaded {len(df)} target entities.")

    def split_multi(value):
        if pd.isna(value) or not str(value).strip():
            return []
        return [v.strip() for v in str(value).split(";") if v.strip()]

    entities = []
    for _, row in df.iterrows():
        entities.append({
            "source": "OpenSanctions",
            "id": str(row.get("id", "")),
            "primary_name": str(row.get("name", "")).strip(),
            "schema": str(row.get("schema", "")) if pd.notna(row.get("schema")) else None,
            "program": ";".join(split_multi(row.get("program_ids"))) or None,
            "aliases": split_multi(row.get("aliases")),
            "addresses": split_multi(row.get("addresses")),
            "countries": split_multi(row.get("countries")),
        })

    return entities


def build_entity_table() -> list:
    """Merge OFAC and OpenSanctions entities into one unified list."""
    ofac_entities = build_ofac_entities()
    opensanctions_entities = build_opensanctions_entities()

    combined = ofac_entities + opensanctions_entities
    print(f"\nCombined total: {len(combined)} entities "
          f"({len(ofac_entities)} OFAC + {len(opensanctions_entities)} OpenSanctions)")

    return combined


if __name__ == "__main__":
    entities = build_entity_table()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(entities)} unified entity records to {OUT_PATH}")
    if entities:
        print("\nSample OFAC record:")
        ofac_sample = next((e for e in entities if e["source"] == "OFAC"), None)
        if ofac_sample:
            print(json.dumps(ofac_sample, indent=2, ensure_ascii=False))
        print("\nSample OpenSanctions record:")
        os_sample = next((e for e in entities if e["source"] == "OpenSanctions"), None)
        if os_sample:
            print(json.dumps(os_sample, indent=2, ensure_ascii=False))
