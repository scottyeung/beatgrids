import numpy as np
import soundfile as sf
import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def make_click_track(tmp_dir):
    """Generate a FLAC file with clicks at known beat positions.

    Returns (file_path, beat_times) where beat_times is the array of
    expected beat positions in seconds.
    """
    def _make(bpm=120.0, duration=10.0, sr=44100, bit_depth=16):
        beat_interval = 60.0 / bpm
        num_samples = int(duration * sr)
        audio = np.zeros(num_samples, dtype=np.float32)

        beat_times = []
        t = 0.0
        while t < duration:
            sample_idx = int(t * sr)
            if sample_idx < num_samples:
                # Short click: 1ms burst
                click_len = min(int(0.001 * sr), num_samples - sample_idx)
                audio[sample_idx:sample_idx + click_len] = 0.8
                beat_times.append(t)
            t += beat_interval

        path = tmp_dir / "click_track.flac"
        subtype = "PCM_16" if bit_depth == 16 else "PCM_24"
        sf.write(str(path), audio, sr, subtype=subtype)
        return path, np.array(beat_times)

    return _make


@pytest.fixture
def make_drift_track(tmp_dir):
    """Generate a FLAC with tempo drift (simulates vinyl rip).

    BPM varies linearly from start_bpm to end_bpm over the duration.
    Returns (file_path, beat_times).
    """
    def _make(start_bpm=119.0, end_bpm=121.0, duration=30.0, sr=44100):
        num_samples = int(duration * sr)
        audio = np.zeros(num_samples, dtype=np.float32)

        beat_times = []
        t = 0.0
        while t < duration:
            sample_idx = int(t * sr)
            if sample_idx < num_samples:
                click_len = min(int(0.001 * sr), num_samples - sample_idx)
                audio[sample_idx:sample_idx + click_len] = 0.8
                beat_times.append(t)
            # Linearly interpolate BPM
            progress = t / duration
            current_bpm = start_bpm + (end_bpm - start_bpm) * progress
            t += 60.0 / current_bpm

        path = tmp_dir / "drift_track.flac"
        sf.write(str(path), audio, sr, subtype="PCM_16")
        return path, np.array(beat_times)

    return _make
