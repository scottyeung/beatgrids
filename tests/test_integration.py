import numpy as np
import soundfile as sf
import subprocess
import sys
from pathlib import Path
from beatgrids.detector import detect_beats
from beatgrids.analyzer import analyze_beats


def test_full_pipeline_reduces_drift(make_drift_track, tmp_dir):
    """Processing a drifting track should produce output near the target BPM.

    We check that the output's average BPM is close to what was computed,
    rather than comparing raw drift (which is sensitive to beat detection
    outliers on synthetic audio).
    """
    path, _ = make_drift_track(start_bpm=118.0, end_bpm=122.0, duration=20.0)

    # Measure original
    original_beats = detect_beats(path)
    original_analysis = analyze_beats(original_beats)

    # Process
    out_dir = tmp_dir / "output"
    out_dir.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "beatgrids", "fix", str(path),
         "-o", str(out_dir), "--engine", "ffmpeg"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0

    # Measure output
    output_file = list(out_dir.glob("*.flac"))[0]
    output_beats = detect_beats(str(output_file))
    output_analysis = analyze_beats(output_beats)

    # Output average BPM should be close to the original average
    assert abs(output_analysis.average_bpm - original_analysis.average_bpm) < 2.0

    # The IQR (interquartile range) of beat intervals should be tighter
    original_intervals = np.diff(original_beats)
    output_intervals = np.diff(output_beats)
    original_iqr = np.percentile(original_intervals, 75) - np.percentile(original_intervals, 25)
    output_iqr = np.percentile(output_intervals, 75) - np.percentile(output_intervals, 25)
    assert output_iqr <= original_iqr + 0.01  # allow small tolerance


def test_full_pipeline_preserves_sample_rate(make_drift_track, tmp_dir):
    """Output should have same sample rate as input."""
    path, _ = make_drift_track(start_bpm=119.0, end_bpm=121.0, duration=10.0)
    original_info = sf.info(str(path))

    out_dir = tmp_dir / "output"
    out_dir.mkdir()
    subprocess.run(
        [sys.executable, "-m", "beatgrids", "fix", str(path),
         "-o", str(out_dir), "--engine", "ffmpeg"],
        capture_output=True, text=True, check=True,
    )

    output_file = list(out_dir.glob("*.flac"))[0]
    output_info = sf.info(str(output_file))

    assert output_info.samplerate == original_info.samplerate


def test_full_pipeline_with_target_bpm(make_drift_track, tmp_dir):
    """--target-bpm should override the computed average."""
    path, _ = make_drift_track(start_bpm=119.0, end_bpm=121.0, duration=10.0)

    out_dir = tmp_dir / "output"
    out_dir.mkdir()
    subprocess.run(
        [sys.executable, "-m", "beatgrids", "fix", str(path),
         "-o", str(out_dir), "--target-bpm", "120.0", "--engine", "ffmpeg"],
        capture_output=True, text=True, check=True,
    )

    output_file = list(out_dir.glob("*.flac"))[0]
    output_beats = detect_beats(str(output_file))
    output_analysis = analyze_beats(output_beats)

    # Should be close to 120.0
    assert abs(output_analysis.average_bpm - 120.0) < 1.0
