from dataclasses import dataclass
import numpy as np


@dataclass
class AnalysisResult:
    average_bpm: float
    min_bpm: float
    max_bpm: float
    drift_pct: float
    beat_count: int
    beat_times: np.ndarray


def analyze_beats(beat_times: np.ndarray) -> AnalysisResult:
    """Compute BPM statistics from detected beat timestamps."""
    if len(beat_times) < 2:
        raise ValueError("Need at least 2 beats for analysis")

    intervals = np.diff(beat_times)
    bpms = 60.0 / intervals

    average_bpm = float(np.mean(bpms))
    min_bpm = float(np.min(bpms))
    max_bpm = float(np.max(bpms))

    # Drift as +/- half-range percentage of average
    half_range = (max_bpm - min_bpm) / 2.0
    drift_pct = (half_range / average_bpm) * 100.0 if average_bpm > 0 else 0.0

    return AnalysisResult(
        average_bpm=average_bpm,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        drift_pct=drift_pct,
        beat_count=len(beat_times),
        beat_times=beat_times,
    )


def build_segments(
    beat_times: np.ndarray, segment_size: int = 16
) -> list[np.ndarray]:
    """Split beat times into segments of N beats.

    If the final segment has fewer than 4 beats, it is merged
    into the preceding segment.
    """
    if len(beat_times) <= segment_size:
        return [beat_times]

    segments = []
    for i in range(0, len(beat_times), segment_size):
        segments.append(beat_times[i : i + segment_size])

    # Merge short final segment
    if len(segments) > 1 and len(segments[-1]) < 4:
        last = segments.pop()
        segments[-1] = np.concatenate([segments[-1], last])

    return segments
