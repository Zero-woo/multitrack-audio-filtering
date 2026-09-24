from pathlib import Path
from typing import Protocol

from waves_sed.prediction import FramePrediction


class SEDBackend(Protocol):
    """Return unsmoothed frame probabilities, independent of temporal references."""

    def predict(self, audio_path: str | Path) -> FramePrediction: ...
