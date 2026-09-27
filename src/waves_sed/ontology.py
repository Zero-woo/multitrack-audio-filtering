"""Load the pinned official AudioSet DAG without inference dependencies.

Ontology IDs and names are separate from the model's ordered 447 output labels.
The official archived ontology does not contain every later strong-label ID.
"""

import hashlib
import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class AudioSetNode:
    id: str
    name: str
    description: str
    child_ids: tuple[str, ...]
    restrictions: tuple[str, ...]


def _text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(
            f"Ontology {field} must be a {'nonempty ' if not allow_empty else ''}string"
        )
    return value


def _strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"Ontology {field} must be a list of strings")
    result = tuple(_text(item, field) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"Ontology {field} contains duplicate entries")
    return result


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate ontology JSON field: {key}")
        result[key] = value
    return result


class AudioSetOntology:
    """A validated directed acyclic graph with deterministic ID-based traversal.

    ``descendants`` and ``ancestors`` return IDs in lexical order, including
    the requested ID by default. Multiple parents are supported. ``resolve_name``
    accepts only an exact display name; it does not guess synonyms or targets.
    """

    def __init__(self, nodes: dict[str, AudioSetNode], provenance: dict):
        self.nodes = nodes
        self.provenance = provenance
        self._names = {node.name: node.id for node in nodes.values()}
        parents: dict[str, list[str]] = {class_id: [] for class_id in nodes}
        for node in nodes.values():
            for child_id in node.child_ids:
                if child_id not in nodes:
                    raise ValueError(f"Ontology {node.id} refers to missing child {child_id}")
                parents[child_id].append(node.id)
        self._parents = {key: tuple(sorted(value)) for key, value in parents.items()}

        # Kahn's algorithm avoids recursion limits on long custom hierarchies.
        indegrees = {class_id: len(value) for class_id, value in parents.items()}
        ready = deque(class_id for class_id, count in indegrees.items() if count == 0)
        visited = 0
        while ready:
            visited += 1
            for child_id in nodes[ready.popleft()].child_ids:
                indegrees[child_id] -= 1
                if indegrees[child_id] == 0:
                    ready.append(child_id)
        if visited != len(nodes):
            raise ValueError("Ontology contains a cycle")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AudioSetOntology":
        """Load bundled official data or an explicitly supplied custom JSON file.

        Bundled content is checked against the pinned SHA-256. For a custom
        file, provenance identifies its absolute path and actual content hash;
        it never inherits the official file's revision or license claims.
        """
        if path is None:
            resources = files("waves_sed.resources")
            payload = resources.joinpath("audioset_ontology.json").read_bytes()
            provenance = json.loads(
                resources.joinpath("audioset_ontology.provenance.json").read_text(encoding="utf-8")
            )
            digest = hashlib.sha256(payload).hexdigest()
            if provenance.get("sha256") != digest:
                raise ValueError("Bundled AudioSet ontology SHA-256 mismatch")
            provenance = {**provenance, "kind": "bundled", "sha256": digest}
        else:
            source = Path(path).resolve()
            payload = source.read_bytes()
            provenance = {
                "kind": "custom",
                "source": str(source),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        try:
            entries = json.loads(payload.decode("utf-8"), object_pairs_hook=_json_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Ontology must contain valid UTF-8 JSON") from exc
        if not isinstance(entries, list) or not entries:
            raise ValueError("Ontology must be a nonempty list of nodes")
        nodes = {}
        names = set()
        for position, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Ontology node {position} must be an object")
            node = AudioSetNode(
                id=_text(entry.get("id"), f"node {position} id"),
                name=_text(entry.get("name"), f"node {position} name"),
                description=_text(entry.get("description"), "description", allow_empty=True),
                child_ids=_strings(entry.get("child_ids"), "child_ids"),
                restrictions=_strings(entry.get("restrictions"), "restrictions"),
            )
            if node.id in nodes:
                raise ValueError(f"Duplicate ontology ID: {node.id}")
            if node.name in names:
                raise ValueError(f"Duplicate ontology name: {node.name}")
            if "citation_uri" in entry:
                _text(entry["citation_uri"], "citation_uri", allow_empty=True)
            if "positive_examples" in entry:
                _strings(entry["positive_examples"], "positive_examples")
            nodes[node.id] = node
            names.add(node.name)
        if path is None and provenance.get("node_count") != len(nodes):
            raise ValueError("Bundled AudioSet ontology node count mismatch")
        provenance = {**provenance, "node_count": len(nodes)}
        return cls(nodes, provenance)

    def get(self, class_id: str) -> AudioSetNode:
        try:
            return self.nodes[class_id]
        except KeyError as exc:
            raise ValueError(f"Unknown ontology class ID: {class_id}") from exc

    def resolve_name(self, name: str) -> str:
        try:
            return self._names[name]
        except KeyError as exc:
            raise ValueError(f"Unknown ontology class name: {name}") from exc

    def _traverse(self, class_id: str, include_self: bool, *, parents: bool) -> tuple[str, ...]:
        self.get(class_id)
        visited = set()
        remaining = [class_id]
        while remaining:
            current = remaining.pop()
            if current in visited:
                continue
            visited.add(current)
            remaining.extend(self._parents[current] if parents else self.nodes[current].child_ids)
        if not include_self:
            visited.remove(class_id)
        return tuple(sorted(visited))

    def descendants(self, class_id: str, include_self: bool = True) -> tuple[str, ...]:
        return self._traverse(class_id, include_self, parents=False)

    def ancestors(self, class_id: str, include_self: bool = True) -> tuple[str, ...]:
        return self._traverse(class_id, include_self, parents=True)

    def vocabulary_coverage(self, class_ids: Sequence[str], class_names: Sequence[str]) -> dict:
        """Report vocabulary differences by ID without changing either label set.

        Missing IDs and name mismatches preserve the supplied classifier order.
        Matching IDs remain valid even when their display names have changed.
        """
        if len(class_ids) != len(class_names):
            raise ValueError("Vocabulary class IDs and names must have the same length")
        if any(not isinstance(value, str) or not value.strip() for value in class_ids):
            raise ValueError("Vocabulary class IDs must be nonempty strings")
        if any(not isinstance(value, str) or not value.strip() for value in class_names):
            raise ValueError("Vocabulary class names must be nonempty strings")
        if len(set(class_ids)) != len(class_ids):
            raise ValueError("Vocabulary class IDs must be unique")
        missing = []
        mismatches = []
        for class_id, class_name in zip(class_ids, class_names, strict=True):
            if class_id not in self.nodes:
                missing.append(class_id)
            elif self.nodes[class_id].name != class_name:
                mismatches.append(
                    {
                        "id": class_id,
                        "model_name": class_name,
                        "ontology_name": self.nodes[class_id].name,
                    }
                )
        return {
            "class_count": len(class_ids),
            "ontology_present_count": len(class_ids) - len(missing),
            "ontology_missing_class_ids": missing,
            "name_mismatches": mismatches,
        }
