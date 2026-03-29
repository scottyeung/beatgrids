import numpy as np
import pytest
from beatgrids.analyzer import analyze_beats, AnalysisResult, build_segments


def test_analyze_beats_constant_tempo():
    """Evenly spaced beats at 120 BPM should report ~120 BPM."""
    beat_interval = 60.0 / 120.0  # 0.5s
    beat_times = np.arange(0.0, 30.0, beat_interval)
    result = analyze_beats(beat_times)

    assert abs(result.average_bpm - 120.0) < 0.1
    assert result.beat_count == len(beat_times)
    assert result.drift_pct < 0.1


def test_analyze_beats_variable_tempo():
    """Drifting tempo should report average and nonzero drift."""
    beat_times = []
    t = 0.0
    for i in range(60):
        beat_times.append(t)
        bpm = 118.0 + (122.0 - 118.0) * (i / 59)
        t += 60.0 / bpm
    beat_times = np.array(beat_times)
    result = analyze_beats(beat_times)

    assert 119.0 < result.average_bpm < 121.0
    assert result.drift_pct > 0.5
    assert result.min_bpm < result.max_bpm


def test_analyze_beats_too_few():
    """Fewer than 2 beats should raise ValueError."""
    with pytest.raises(ValueError, match="at least 2 beats"):
        analyze_beats(np.array([1.0]))


def test_build_segments_even_division():
    """60 beats with segment_size=16 -> 3 full + 1 partial (merged if <4)."""
    beat_times = np.arange(0.0, 30.0, 0.5)  # 60 beats
    segments = build_segments(beat_times, segment_size=16)

    # 60 / 16 = 3 full (48 beats) + 12 remaining -> 4 segments
    assert len(segments) == 4


def test_build_segments_short_final_merged():
    """Final segment with <4 beats is merged into previous."""
    beat_times = np.arange(0.0, 17.5, 0.5)  # 35 beats
    segments = build_segments(beat_times, segment_size=16)

    # 35 / 16 = 2 full (32 beats) + 3 remaining (<4) -> merged into segment 2
    assert len(segments) == 2
    assert len(segments[-1]) == 16 + 3  # merged


def test_build_segments_very_short_track():
    """Track with fewer than segment_size beats -> single segment."""
    beat_times = np.arange(0.0, 5.0, 0.5)  # 10 beats
    segments = build_segments(beat_times, segment_size=16)

    assert len(segments) == 1
    assert len(segments[0]) == 10
