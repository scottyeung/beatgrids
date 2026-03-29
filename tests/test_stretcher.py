import numpy as np
import soundfile as sf
import pytest
from pathlib import Path
from beatgrids.stretcher import (
    compute_segment_ratios,
    stretch_segment_ffmpeg,
    stretch_and_concat,
    _overlap_add_concat,
)


def test_overlap_add_preserves_length():
    """True overlap-add should not remove audio. Output length =
    sum(parts) - crossfade * (num_parts - 1)."""
    a = np.ones(100)
    b = np.ones(100)
    c = np.ones(100)
    cf = 10
    result = _overlap_add_concat([a, b, c], cf)
    expected_len = 300 - 10 * 2  # 280
    assert len(result) == expected_len


def test_overlap_add_single_part():
    """Single part should be returned unchanged."""
    a = np.array([1.0, 2.0, 3.0])
    result = _overlap_add_concat([a], 10)
    np.testing.assert_array_equal(result, a)


def test_compute_segment_ratios():
    """Each segment ratio = target / local BPM."""
    segments = [
        np.array([0.0, 0.5, 1.0, 1.5]),     # 120 BPM
        np.array([2.0, 2.52, 3.04, 3.56]),   # ~115.4 BPM
        np.array([4.0, 4.48, 4.96, 5.44]),   # 125 BPM
    ]
    target_bpm = 120.0
    ratios = compute_segment_ratios(segments, target_bpm)

    assert len(ratios) == 3
    assert abs(ratios[0] - 1.0) < 0.01
    assert ratios[1] > 1.0
    assert ratios[2] < 1.0


def test_stretch_segment_ffmpeg_preserves_sample_rate(make_click_track, tmp_dir):
    """Stretched segment should have same sample rate as input."""
    path, _ = make_click_track(bpm=120.0, duration=2.0, sr=44100)
    output_path = tmp_dir / "stretched.wav"

    stretch_segment_ffmpeg(str(path), str(output_path), ratio=1.02)

    info = sf.info(str(output_path))
    assert info.samplerate == 44100


def test_stretch_segment_ffmpeg_changes_duration(make_click_track, tmp_dir):
    """Stretching at ratio > 1 should shorten the audio."""
    path, _ = make_click_track(bpm=120.0, duration=5.0, sr=44100)
    original_info = sf.info(str(path))

    output_path = tmp_dir / "stretched.wav"
    stretch_segment_ffmpeg(str(path), str(output_path), ratio=1.05)

    stretched_info = sf.info(str(output_path))
    assert stretched_info.duration < original_info.duration


def test_stretch_and_concat_produces_output(make_drift_track, tmp_dir):
    """Full pipeline should produce a valid FLAC output."""
    path, beat_times = make_drift_track(
        start_bpm=119.0, end_bpm=121.0, duration=10.0
    )
    output_path = tmp_dir / "output.flac"

    stretch_and_concat(
        input_path=str(path),
        output_path=str(output_path),
        beat_times=beat_times,
        target_bpm=120.0,
        segment_size=16,
        engine="ffmpeg",
    )

    assert output_path.exists()
    info = sf.info(str(output_path))
    assert info.samplerate == 44100
    assert info.duration > 0
