import hashlib
from pathlib import Path

UPSTREAM_REVISION = "1aa47e482f7e89904cba2338999345025d8b4e36"
CHECKPOINT_NAME = "ATST-F_strong_1.pt"
CHECKPOINT_URL = (
    "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/" + CHECKPOINT_NAME
)
# Computed from the official release download, not a publisher-signed digest.
CHECKPOINT_SHA256 = "fbf2577958e3648d55ee8cea7e0e5260c4505fb0c13964e1ec81bc367cee5eda"
PREPROCESSING = "librosa-soxr_hq-mono-16k-zero-pad-10s-sigmoid-40ms-v1"


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_checkpoint(path: str | Path) -> str:
    digest = sha256(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: expected {CHECKPOINT_SHA256}, got {digest}")
    return digest


def download_checkpoint(path: str | Path) -> Path:
    """Download the pinned official weights and verify before making them available."""
    import shutil
    from tempfile import NamedTemporaryFile
    from urllib.request import urlopen

    path = Path(path)
    if path.exists():
        verify_checkpoint(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urlopen(CHECKPOINT_URL, timeout=60) as response:
                shutil.copyfileobj(response, stream)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        verify_checkpoint(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
