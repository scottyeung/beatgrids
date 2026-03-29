import numpy as np
from beatgrids.detector import detect_beats


def test_detect_beats_finds_clicks(make_click_track):
    """Detector should find beats within 50ms of actual click positions."""
    path, expected_beats = make_click_track(bpm=120.0, duration=10.0)
    beat_times = detect_beats(path)

    assert len(beat_times) > 0
    # Each detected beat should be within 50ms of an expected beat
    for detected in beat_times:
        distances = np.abs(expected_beats - detected)
        assert distances.min() < 0.05, f"Beat at {detected:.3f}s not near any expected beat"

    # Should find at least 80% of expected beats (recall check)
    assert len(beat_times) >= len(expected_beats) * 0.8, (
        f"Only found {len(beat_times)}/{len(expected_beats)} expected beats"
    )


def test_detect_beats_returns_sorted_array(make_click_track):
    path, _ = make_click_track(bpm=120.0, duration=5.0)
    beat_times = detect_beats(path)

    assert len(beat_times) > 1
    assert np.all(np.diff(beat_times) > 0), "Beat times must be sorted ascending"


def test_detect_beats_with_custom_start_bpm(make_click_track):
    path, _ = make_click_track(bpm=90.0, duration=10.0)
    beat_times = detect_beats(path, start_bpm=90.0)

    assert len(beat_times) > 0
