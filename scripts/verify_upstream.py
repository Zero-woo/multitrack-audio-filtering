"""Compare the adapter with an unchanged, pinned PretrainedSED checkout on CPU.

Run with an environment containing the installed waves-stem-sed[atst] package:
    python scripts/verify_upstream.py --upstream .cache/PretrainedSED \
        --checkpoint checkpoints/ATST-F_strong_1.pt --audio example.wav

Only the first ten seconds are compared. A shorter --seconds prefix is padded
to ten seconds on both sides. Upstream imports live only in a child process;
neither the production adapter nor this process adds upstream to sys.path.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 160000
TOLERANCE = 1e-6
MEL_BUFFERS = {
    "model.atst_mel.mel_transform.spectrogram.window",
    "model.atst_mel.mel_transform.mel_scale.fb",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Compare this much of the first chunk, padded to 10 seconds (0 < seconds <= 10).",
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU PyTorch threads (default: 4).")
    parser.add_argument("--upstream-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.seconds <= 10 or round(args.seconds * SAMPLE_RATE) < 1:
        parser.error("--seconds must span at least one sample and be at most 10 seconds")
    if args.threads < 1:
        parser.error("--threads must be positive")
    for name in ("upstream", "checkpoint", "audio"):
        path = getattr(args, name).resolve()
        if not path.exists():
            parser.error(f"--{name} does not exist: {path}")
        setattr(args, name, path)
    return args


def first_chunk(waveform, seconds):
    import numpy as np

    valid = waveform[: round(seconds * SAMPLE_RATE)]
    if valid.ndim != 1 or not len(valid) or not np.isfinite(valid).all():
        raise ValueError("Audio must decode to a finite, nonempty mono waveform")
    return np.pad(valid, (0, CHUNK_SAMPLES - len(valid))), len(valid)


def run_upstream(args):
    """Run only in the isolated child; never invoke the download-aware loader."""
    import librosa
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.upstream))
    from models.atstframe.ATSTF_wrapper import ATSTWrapper
    from models.prediction_wrapper import PredictionsWrapper

    model = PredictionsWrapper(ATSTWrapper(), checkpoint=None)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != MEL_BUFFERS or unexpected:
        raise ValueError(f"Upstream checkpoint mismatch: {missing=}, {unexpected=}")
    del state
    model.eval().requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Upstream model is not frozen in evaluation mode")

    waveform, _ = librosa.core.load(args.audio, sr=SAMPLE_RATE, mono=True)
    chunk, valid_samples = first_chunk(waveform, args.seconds)
    audio = torch.from_numpy(chunk).unsqueeze(0)
    with torch.inference_mode():
        mel = model.mel_forward(audio)
        logits, _ = model(mel)
        probabilities = logits.sigmoid()
    np.savez(
        args.upstream_output,
        waveform=chunk,
        mel=mel.numpy(),
        probabilities=probabilities.numpy(),
        valid_samples=valid_samples,
    )


def verify_checkout(path, expected_revision):
    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(path), *arguments],
            text=True,
        ).strip()

    revision = git("rev-parse", "HEAD")
    if revision != expected_revision:
        raise ValueError(f"Expected upstream revision {expected_revision}, got {revision}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Upstream checkout has tracked modifications; use an unchanged checkout")
    return revision


def comparison(actual, reference):
    import numpy as np

    same_shape = actual.shape == reference.shape
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    max_abs = float(np.max(np.abs(actual - reference))) if same_shape and finite else None
    return {
        "adapter_shape": list(actual.shape),
        "upstream_shape": list(reference.shape),
        "max_abs_difference": max_abs,
        "passed": max_abs is not None and max_abs <= TOLERANCE,
    }


def verify(args):
    import numpy as np
    import torch

    from waves_sed.audio import load_audio
    from waves_sed.backends.atst import ATSTBackend
    from waves_sed.provenance import UPSTREAM_REVISION, sha256

    revision = verify_checkout(args.upstream, UPSTREAM_REVISION)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    # Validate the pinned checkpoint before the child loads any weights.
    adapter = ATSTBackend(args.checkpoint, device="cpu")

    with tempfile.TemporaryDirectory(prefix="waves-sed-upstream-") as temporary:
        output = Path(temporary) / "upstream.npz"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--upstream",
                str(args.upstream),
                "--checkpoint",
                str(args.checkpoint),
                "--audio",
                str(args.audio),
                "--seconds",
                str(args.seconds),
                "--threads",
                str(args.threads),
                "--upstream-output",
                str(output),
            ],
            check=True,
        )
        # The child has exited, so the two large models need not coexist in RAM.
        model = adapter._load_model()
        all_frozen = all(not parameter.requires_grad for parameter in model.parameters())
        all_eval = all(not module.training for module in model.modules())
        waveform, _ = load_audio(args.audio)
        chunk, valid_samples = first_chunk(waveform, args.seconds)
        audio = torch.from_numpy(chunk).unsqueeze(0)
        with torch.inference_mode():
            mel = model.model.mel_forward(audio)
            probabilities = model(audio).sigmoid()

        with np.load(output, allow_pickle=False) as reference:
            checks = {
                "waveform": comparison(chunk, reference["waveform"]),
                "mel": comparison(mel.numpy(), reference["mel"]),
                "probabilities": comparison(probabilities.numpy(), reference["probabilities"]),
            }
            same_length = valid_samples == int(reference["valid_samples"])

    passed = (
        all_frozen
        and all_eval
        and same_length
        and all(check["passed"] for check in checks.values())
    )
    result = {
        "passed": passed,
        "upstream_revision": revision,
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "audio": str(args.audio),
        "audio_sha256": sha256(args.audio),
        "device": "cpu",
        "torch_threads": args.threads,
        "requested_prefix_seconds": args.seconds,
        "valid_samples": valid_samples,
        "padded_chunk_samples": CHUNK_SAMPLES,
        "absolute_tolerance": TOLERANCE,
        "adapter_all_parameters_frozen": all_frozen,
        "adapter_all_modules_eval": all_eval,
        "same_valid_samples": same_length,
        "comparisons": checks,
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.upstream_output is not None:
        run_upstream(arguments)
    else:
        raise SystemExit(verify(arguments))
