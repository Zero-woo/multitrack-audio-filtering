"""Audio preprocessing and frame geometry; the grid works without torch."""

from pathlib import Path

import numpy as np


def frame_grid(num_samples: int, sample_rate: int = 16000, frame_samples: int = 640):
    """Half-open frame support, retaining a partial final frame but never padding."""
    if any(
        not isinstance(v, int) or isinstance(v, bool) or v <= 0
        for v in (num_samples, sample_rate, frame_samples)
    ):
        raise ValueError("Sample count, sample rate and frame size must be positive integers")
    count = (num_samples + frame_samples - 1) // frame_samples
    starts = np.arange(count, dtype=np.float64) * frame_samples / sample_rate
    ends = (
        np.minimum((np.arange(count, dtype=np.float64) + 1) * frame_samples, num_samples)
        / sample_rate
    )
    return starts, ends


def load_audio(path: str | Path):
    """Match upstream: mono mean and librosa/soxr HQ resampling to 16 kHz."""
    import librosa
    import soundfile as sf

    path = Path(path)
    info = sf.info(path)
    if info.frames == 0:
        raise ValueError("Audio contains no samples")
    try:
        audio, _ = librosa.load(path, sr=16000, mono=True, dtype=np.float32, res_type="soxr_hq")
    except librosa.util.exceptions.ParameterError as error:
        raise ValueError(f"Invalid audio: {error}") from error
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains nonfinite samples")
    return audio, {
        "original_sample_rate": info.samplerate,
        "original_channels": info.channels,
        "original_samples": info.frames,
        "original_duration_seconds": info.duration,
        "inference_sample_rate": 16000,
        "inference_samples": len(audio),
        "resampled_duration_seconds": len(audio) / 16000,
        "duration_seconds": info.duration,
    }
