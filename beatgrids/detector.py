import librosa
import numpy as np
from pathlib import Path


def detect_beats(
    file_path: str | Path,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    refine: bool = True,
) -> np.ndarray:
    """Detect beat positions in an audio file.

    Uses librosa for analysis only (downmixed, resampled copy).
    Returns an array of beat timestamps in seconds, sorted ascending.

    If refine=True, snaps each beat to the nearest onset strength peak
    within a 30ms window for sub-frame precision.
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

    if refine and len(beat_times) > 0:
        beat_times = _refine_beats(y, sr, beat_times)

    return beat_times


def _refine_beats(
    y: np.ndarray, sr: int, beat_times: np.ndarray, fine_hop: int = 64
) -> np.ndarray:
    """Refine beat positions by finding onset strength peaks near each beat.

    Uses a small hop_length for higher temporal resolution (~1.5ms at 44100Hz).
    For each beat, finds the strongest onset within a 30ms window.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=fine_hop)
    window_sec = 0.030  # 30ms search window
    window_frames = int(window_sec * sr / fine_hop)

    refined = []
    for bt in beat_times:
        center_frame = int(bt * sr / fine_hop)
        start = max(0, center_frame - window_frames)
        end = min(len(onset_env), center_frame + window_frames + 1)

        if start >= end:
            refined.append(bt)
            continue

        peak_offset = np.argmax(onset_env[start:end])
        peak_frame = start + peak_offset
        peak_time = librosa.frames_to_time(peak_frame, sr=sr, hop_length=fine_hop)
        refined.append(float(peak_time))

    return np.array(refined)
