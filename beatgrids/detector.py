import librosa
import numpy as np
from pathlib import Path


def detect_beats(
    file_path: str | Path,
    hop_length: int = 512,
    start_bpm: float = 120.0,
) -> np.ndarray:
    """Detect beat positions in an audio file.

    Uses librosa for analysis only (downmixed, resampled copy).
    Returns an array of beat timestamps in seconds, sorted ascending.
    """
    y, sr = librosa.load(str(file_path), sr=None, mono=True)

    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
        units="frames",
    )

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    beat_times = np.sort(beat_times)

    return beat_times
