"""Opt-in real weights test: WAVES_SED_CHECKPOINT must name the official checkpoint."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_real_checkpoint_stereo_multichunk_partial_tail(tmp_path):
    checkpoint = os.environ.get("WAVES_SED_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set WAVES_SED_CHECKPOINT to run the real frozen model")
    torch = pytest.importorskip("torch")
    sf = pytest.importorskip("soundfile")
    from waves_sed.backends.atst import ATSTBackend
    from waves_sed.prediction import FramePrediction

    torch.set_num_threads(4)
    sample_rate = 22050
    sample_count = 220721  # just above 10 seconds, and fractional at 16 kHz
    t = np.arange(sample_count) / sample_rate
    audio = 0.1 * np.sin(2 * np.pi * 220 * t)
    path = tmp_path / "stereo_22050.wav"
    sf.write(path, np.column_stack((audio, audio * 0.5)), sample_rate, subtype="FLOAT")
    backend = ATSTBackend(checkpoint, device="cpu")
    prediction = backend.predict(path)
    assert prediction.probabilities.shape == (251, 447)
    assert prediction.probabilities.dtype == np.float32
    assert prediction.frame_start_seconds[-1] == 10.0
    assert prediction.frame_end_seconds[-1] == sample_count / sample_rate
    assert not backend.model.training
    assert all(not p.requires_grad and p.grad is None for p in backend.model.parameters())
    assert prediction.metadata["model_frozen"] is True
    assert prediction.metadata["original_channels"] == 2
    cache = tmp_path / "raw.npz"
    prediction.save(cache)
    loaded = FramePrediction.load(cache)
    np.testing.assert_array_equal(loaded.probabilities, prediction.probabilities)
