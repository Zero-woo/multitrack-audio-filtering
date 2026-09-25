"""Portable raw scores. No torch or model dependency is needed to read this cache."""

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import numpy as np

SCHEMA_VERSION = 1


@dataclass
class FramePrediction:
    probabilities: np.ndarray
    frame_start_seconds: np.ndarray
    frame_end_seconds: np.ndarray
    class_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    metadata: dict

    def validate(self) -> None:
        scores = np.asarray(self.probabilities)
        start, end = np.asarray(self.frame_start_seconds), np.asarray(self.frame_end_seconds)
        if scores.ndim != 2 or not all(scores.shape):
            raise ValueError("Probabilities must have nonempty shape [time, classes]")
        if not np.issubdtype(scores.dtype, np.floating):
            raise ValueError("Probabilities must be floating point")
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("Probabilities must be finite and in [0, 1]")
        if start.shape != (len(scores),) or end.shape != start.shape:
            raise ValueError("Frame time dimensions do not match probabilities")
        if any(
            not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
            for values in (start, end)
        ):
            raise ValueError("Frame times must be real numbers")
        if (
            not np.isfinite(start).all()
            or not np.isfinite(end).all()
            or (start < 0).any()
            or (end <= start).any()
            or (start[1:] < end[:-1]).any()
        ):
            raise ValueError("Frame supports must be finite, positive, ordered and nonoverlapping")
        for values, kind in [(self.class_ids, "IDs"), (self.class_names, "names")]:
            if (
                len(values) != scores.shape[1]
                or any(not isinstance(v, str) or not v.strip() for v in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"Class {kind} must be unique nonempty strings matching columns")
        if not isinstance(self.metadata, dict):
            raise ValueError("Prediction metadata must be a JSON object")
        try:
            json.dumps(self.metadata, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ValueError("Prediction metadata must be finite JSON data") from error

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(
                    stream,
                    schema_version=np.array(SCHEMA_VERSION, dtype=np.int64),
                    probabilities=np.asarray(self.probabilities, dtype=np.float32),
                    frame_start_seconds=np.asarray(self.frame_start_seconds, dtype=np.float64),
                    frame_end_seconds=np.asarray(self.frame_end_seconds, dtype=np.float64),
                    class_ids=np.asarray(self.class_ids, dtype=np.str_),
                    class_names=np.asarray(self.class_names, dtype=np.str_),
                    metadata_json=np.array(
                        json.dumps(self.metadata, ensure_ascii=False, allow_nan=False)
                    ),
                )
            except BaseException:
                stream.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "FramePrediction":
        try:
            with np.load(path, allow_pickle=False) as data:
                if data["schema_version"].item() != SCHEMA_VERSION:
                    raise ValueError("Unsupported prediction schema version")
                result = cls(
                    probabilities=data["probabilities"],
                    frame_start_seconds=data["frame_start_seconds"],
                    frame_end_seconds=data["frame_end_seconds"],
                    class_ids=tuple(data["class_ids"].tolist()),
                    class_names=tuple(data["class_names"].tolist()),
                    metadata=json.loads(data["metadata_json"].item()),
                )
            result.validate()
            return result
        except (KeyError, TypeError, ValueError, BadZipFile, EOFError) as error:
            raise ValueError(f"Invalid prediction cache {path}: {error}") from error
