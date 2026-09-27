"""Verify graph semantics and the version gap with ATST-F's strong labels."""

import hashlib
import json
from importlib.resources import files

import pytest

from waves_sed.labels import load_labels
from waves_sed.ontology import AudioSetOntology


def node(class_id, name=None, children=(), restrictions=()):
    return {
        "id": class_id,
        "name": name or class_id,
        "description": "",
        "child_ids": list(children),
        "restrictions": list(restrictions),
    }


def load_custom(tmp_path, entries):
    path = tmp_path / "ontology.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return AudioSetOntology.load(path)


def test_bundled_official_ontology_is_pinned_and_unchanged():
    ontology = AudioSetOntology.load()
    payload = files("waves_sed.resources").joinpath("audioset_ontology.json").read_bytes()

    assert len(ontology.nodes) == 632
    assert ontology.provenance["kind"] == "bundled"
    assert ontology.provenance["source"] == "https://github.com/audioset/ontology"
    assert ontology.provenance["revision"] == "d417d32bf59c711abb5910fd2f76a0eb44697991"
    assert ontology.provenance["sha256"] == hashlib.sha256(payload).hexdigest()
    assert ontology.provenance["license"] == "CC-BY-SA-4.0"
    assert ontology.get("/m/0bt9lr").name == "Dog"
    assert ontology.resolve_name("Walk, footsteps") == "/m/07pbtc8"
    assert "/m/05tny_" in ontology.descendants("/m/0bt9lr")
    assert "/m/0bt9lr" in ontology.ancestors("/m/05tny_")


def test_bundled_ontology_reports_actual_strong_label_coverage():
    # The official archive predates additions/renames in the strong vocabulary.
    # Never manufacture parents for these classifier outputs to make counts fit.
    ontology = AudioSetOntology.load()
    class_ids, class_names = load_labels()
    coverage = ontology.vocabulary_coverage(class_ids, class_names)

    assert coverage["class_count"] == 447
    assert coverage["ontology_present_count"] == 416
    assert set(coverage["ontology_missing_class_ids"]) == {
        "/t/dd00144",
        "/t/dd00142",
        "/m/01j2bj",
        "/t/dd00138",
        "/m/08dckq",
        "/m/018p4k",
        "/m/0269r2s",
        "/m/0hgq8df",
        "/t/dd00147",
        "/m/0c1tlg",
        "/m/098_xr",
        "/m/056r_1",
        "/m/0d4wf",
        "/m/04ctx",
        "/m/02ll1_",
        "/m/06cyt0",
        "/t/dd00141",
        "/m/0641k",
        "/m/0md09",
        "/m/040b_t",
        "/m/0bcdqg",
        "/m/02f9f_",
        "/m/07pqmly",
        "/m/01lynh",
        "/m/07sk0jz",
        "/m/07s13rg",
        "/m/0fw86",
        "/t/dd00143",
        "/m/0bzvm2",
        "/m/0174k2",
        "/m/02417f",
    }
    assert len(coverage["name_mismatches"]) == 11
    assert {
        "id": "/m/0199g",
        "model_name": "Bicycle, tricycle",
        "ontology_name": "Bicycle",
    } in coverage["name_mismatches"]
    assert class_names[class_ids.index("/m/07pbtc8")] == "Walk, footsteps"
    assert ontology.get("/m/0199g").name == "Bicycle"


def test_dag_traversal_handles_multiple_parents_and_disconnected_nodes(tmp_path):
    ontology = load_custom(
        tmp_path,
        [
            node("root", children=("right", "left")),
            node("left", children=("leaf",)),
            node("right", children=("leaf",)),
            node("leaf"),
            node("other"),
        ],
    )

    assert ontology.descendants("root") == ("leaf", "left", "right", "root")
    assert ontology.descendants("root", include_self=False) == ("leaf", "left", "right")
    assert ontology.ancestors("leaf") == ("leaf", "left", "right", "root")
    assert ontology.ancestors("leaf", include_self=False) == ("left", "right", "root")
    assert ontology.descendants("leaf", include_self=False) == ()
    assert ontology.ancestors("root", include_self=False) == ()
    assert ontology.descendants("other") == ("other",)


def test_custom_file_provenance_has_its_own_hash_and_no_official_claims(tmp_path):
    ontology = load_custom(tmp_path, [node("root", name="Custom")])
    path = tmp_path / "ontology.json"

    assert ontology.provenance == {
        "kind": "custom",
        "source": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "node_count": 1,
    }
    assert ontology.resolve_name("Custom") == "root"
    with pytest.raises(ValueError, match="Unknown ontology class name"):
        ontology.resolve_name("custom")


@pytest.mark.parametrize("method", ["get", "descendants", "ancestors"])
def test_unknown_id_is_explicit(tmp_path, method):
    ontology = load_custom(tmp_path, [node("known")])
    with pytest.raises(ValueError, match="Unknown ontology class ID"):
        getattr(ontology, method)("unknown")


@pytest.mark.parametrize(
    "entries, message",
    [
        ([node("same"), node("same")], "Duplicate ontology ID"),
        ([node("a", "Same"), node("b", "Same")], "Duplicate ontology name"),
        ([node("root", children=("missing",))], "missing child missing"),
        ([node("root", children=("root",))], "cycle"),
        ([node("a", children=("b",)), node("b", children=("a",))], "cycle"),
        ([], "nonempty list"),
        ({"id": "object"}, "nonempty list"),
        (["string"], "must be an object"),
    ],
)
def test_invalid_graphs_fail_at_load(tmp_path, entries, message):
    with pytest.raises(ValueError, match=message):
        load_custom(tmp_path, entries)


@pytest.mark.parametrize(
    "field, value",
    [
        ("id", ""),
        ("id", 1),
        ("name", "   "),
        ("description", None),
        ("child_ids", "child"),
        ("child_ids", [1]),
        ("child_ids", ["child", "child"]),
        ("restrictions", None),
        ("restrictions", [False]),
        ("citation_uri", []),
        ("positive_examples", "url"),
    ],
)
def test_malformed_fields_are_rejected(tmp_path, field, value):
    entry = node("root")
    entry[field] = value
    with pytest.raises(ValueError, match=field):
        load_custom(tmp_path, [entry])


@pytest.mark.parametrize("field", ["id", "name", "description", "child_ids", "restrictions"])
def test_required_fields_cannot_be_omitted(tmp_path, field):
    entry = node("root")
    del entry[field]
    with pytest.raises(ValueError, match=field):
        load_custom(tmp_path, [entry])


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"[", "valid UTF-8 JSON"),
        (b"\xff", "valid UTF-8 JSON"),
        (b'[{"id": "first", "id": "second"}]', "Duplicate ontology JSON field"),
    ],
)
def test_invalid_json_and_duplicate_fields_fail(tmp_path, payload, message):
    path = tmp_path / "ontology.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        AudioSetOntology.load(str(path))


def test_deep_custom_ontology_does_not_rely_on_python_recursion(tmp_path):
    count = 1200
    ontology = load_custom(
        tmp_path,
        [
            node(str(index), children=(str(index + 1),) if index + 1 < count else ())
            for index in range(count)
        ],
    )
    assert len(ontology.descendants("0")) == count
    assert len(ontology.ancestors(str(count - 1))) == count


def test_coverage_preserves_classifier_order_and_reports_display_name_changes(tmp_path):
    ontology = load_custom(tmp_path, [node("id", name="Original")])
    assert ontology.vocabulary_coverage(["missing-b", "id", "missing-a"], ["B", "New", "A"]) == {
        "class_count": 3,
        "ontology_present_count": 1,
        "ontology_missing_class_ids": ["missing-b", "missing-a"],
        "name_mismatches": [{"id": "id", "model_name": "New", "ontology_name": "Original"}],
    }


@pytest.mark.parametrize(
    "ids, names, message",
    [
        (["id"], [], "same length"),
        (["id", "id"], ["One", "Two"], "unique"),
        ([""], ["Name"], "IDs must be nonempty"),
        (["id"], [None], "names must be nonempty"),
    ],
)
def test_invalid_coverage_vocabulary_fails(tmp_path, ids, names, message):
    ontology = load_custom(tmp_path, [node("id")])
    with pytest.raises(ValueError, match=message):
        ontology.vocabulary_coverage(ids, names)
