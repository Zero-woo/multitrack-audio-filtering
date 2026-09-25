import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("librosa")

from waves_sed.audio import load_audio  # noqa: E402


def test_stereo_mono_mean_and_resampling(tmp_path):
    path = tmp_path / "stereo.wav"
    # Opposite channels must cancel, rather than flattening channels into a longer stream.
    t = np.arange(22050) / 22050
    left = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, np.column_stack((left, -left)), 22050, subtype="FLOAT")
    audio, info = load_audio(path)
    assert audio.shape == (16000,)
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, 0, atol=1e-8)
    assert info["original_channels"] == 2
    assert info["original_sample_rate"] == 22050
    assert info["duration_seconds"] == 1.0


@pytest.mark.parametrize(
    "samples", [np.array([], dtype=np.float32), np.array([0.0, np.nan], dtype=np.float32)]
)
def test_reject_empty_or_nonfinite_waveform(tmp_path, samples):
    path = tmp_path / "invalid.wav"
    sf.write(path, samples, 16000, subtype="FLOAT")
    with pytest.raises(ValueError):
        load_audio(path)
