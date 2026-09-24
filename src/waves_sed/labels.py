"""The 447 classifier outputs are NOT the CSV's MID ordering or all 527 classes."""

import json
from importlib.resources import files


def load_labels() -> tuple[tuple[str, ...], tuple[str, ...]]:
    entries = json.loads(
        files("waves_sed.resources").joinpath("audioset_strong.json").read_text(encoding="utf-8")
    )
    if len(entries) != 447 or [row["index"] for row in entries] != list(range(447)):
        raise ValueError("ATST-F Strong requires the ordered 447-class vocabulary")
    ids, names = tuple(row["id"] for row in entries), tuple(row["name"] for row in entries)
    if len(set(ids)) != 447 or len(set(names)) != 447:
        raise ValueError("Duplicate entries in model vocabulary")
    return ids, names
