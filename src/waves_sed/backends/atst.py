"""Frozen adapter for the official PretrainedSED ATST-F Strong student."""

from importlib.metadata import version
from pathlib import Path

import numpy as np

from waves_sed import __version__
from waves_sed.audio import frame_grid, load_audio
from waves_sed.labels import load_labels
from waves_sed.prediction import FramePrediction
from waves_sed.provenance import (
    CHECKPOINT_URL,
    PREPROCESSING,
    UPSTREAM_REVISION,
    sha256,
    verify_checkpoint,
)


class ATSTBackend:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = verify_checkpoint(self.checkpoint)
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        self.device = device
        self.model = None

    def _load_model(self):
        import torch
        from torch import nn

        from waves_sed._vendor.pretrained_sed.models.atstframe.ATSTF_wrapper import ATSTWrapper

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")

        class StrongStudent(nn.Module):
            """Exactly the upstream PredictionsWrapper default linear-head inference path."""

            def __init__(self):
                super().__init__()
                self.model = ATSTWrapper()
                self.strong_head = nn.Linear(768, 447)
                # The checkpoint contains this unused head; validate it as well.
                self.weak_head = nn.Linear(768, 447)

            def forward(self, audio):
                embeddings = self.model(self.model.mel_forward(audio))
                if embeddings.shape[1:] != (250, 768):
                    raise RuntimeError(f"Unexpected ATST embedding shape: {embeddings.shape}")
                return self.strong_head(embeddings).transpose(1, 2)

        model = StrongStudent()
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        # The official checkpoint predates persisted torchaudio mel buffers.
        # Only these two deterministic buffers may be absent; heads never fall back to random.
        expected_missing = {
            "model.atst_mel.mel_transform.spectrogram.window",
            "model.atst_mel.mel_transform.mel_scale.fb",
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if set(missing) != expected_missing or unexpected:
            raise ValueError(
                f"Checkpoint structure mismatch: missing={missing}, unexpected={unexpected}"
            )
        model.eval().requires_grad_(False)
        self.model = model.to(self.device)
        return self.model

    def predict(self, audio_path: str | Path) -> FramePrediction:
        import torch

        audio_path = Path(audio_path).resolve()
        audio_sha256 = sha256(audio_path)
        audio, audio_metadata = load_audio(audio_path)
        starts, ends = frame_grid(len(audio))
        model = self.model if self.model is not None else self._load_model()
        chunks = []
        with torch.inference_mode():
            for offset in range(0, len(audio), 160000):
                chunk = torch.from_numpy(audio[offset : offset + 160000]).unsqueeze(0)
                chunk = torch.nn.functional.pad(chunk, (0, 160000 - chunk.shape[1]))
                logits = model(chunk.to(self.device))
                if logits.shape != (1, 447, 250):
                    raise RuntimeError(f"Unexpected model output shape: {logits.shape}")
                chunks.append(logits.sigmoid()[0].T.cpu().numpy())
        probabilities = np.concatenate(chunks, axis=0)[: len(starts)].astype(np.float32)
        ids, names = load_labels()
        prediction = FramePrediction(
            probabilities,
            starts,
            ends,
            ids,
            names,
            {
                "backend": "atst_f_strong",
                "package_version": __version__,
                "upstream_revision": UPSTREAM_REVISION,
                "checkpoint_url": CHECKPOINT_URL,
                "checkpoint_sha256": self.checkpoint_sha256,
                "audio_path": str(audio_path),
                "audio_sha256": audio_sha256,
                "preprocessing": PREPROCESSING,
                "device": self.device,
                "frame_hop_seconds": 0.04,
                "chunk_seconds": 10.0,
                "frame_time_convention": "half-open support [start,end); final frame clipped",
                "smoothing": None,
                "threshold": None,
                "model_frozen": True,
                "dependency_versions": {
                    name: version(name)
                    for name in ("torch", "torchaudio", "numpy", "librosa", "einops")
                },
                **audio_metadata,
            },
        )
        prediction.validate()
        return prediction
