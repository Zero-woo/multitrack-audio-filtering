# WAVES adapter fixtures

These compact JSON excerpts come from `C:/WAVES/data/frozen_pass2/` at WAVES
revision `07af161`. Final stems, decisions, planned descriptions/roles/intervals,
and attempt numbers/audio IDs are retained verbatim for four examples:

- `fb5k_1229__01`: relabelled weapon fire.
- `fb5k_1354__02`: unchanged wizard laugh.
- `fb5k_3156__03`: repeated chopping, with a merged child.
- `legacy_16__02`: span changed to onset, relabelled, with a merged child.

Unselected final stems and unrelated candidates are omitted. Report pairwise
metrics and individual attempt scores are omitted; they are not used by the
adapter. Merged child plans remain present to catch accidental interval unions.
No audio is included. Tests construct materialized metadata using the actual
`scripts/pipeline/materialize_stems.py` schema and make explicit mutations for
missing, invalid, and conflicting metadata cases.
