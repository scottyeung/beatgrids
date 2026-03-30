# Beatgrids Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI tool that detects beats in vinyl rip FLAC files and time-stretches segment by segment to produce an even beat grid.

**Architecture:** Python CLI with librosa for beat detection, soundfile for lossless audio I/O, and ffmpeg atempo for segment-wise time-stretching. The pipeline: detect beats -> compute average BPM -> split into segments -> stretch each to target BPM -> concatenate with overlap-add crossfade -> write FLAC with preserved metadata.

**Tech Stack:** Python 3.10+, librosa, soundfile, numpy, ffmpeg, argparse

**Spec:** `docs/superpowers/specs/2026-03-29-beatgrids-design.md`

---

## File Structure

```
beatgrids/
  pyproject.toml          # Project config, dependencies, CLI entry point
  beatgrids/
    __init__.py           # Package init, version
    __main__.py           # python -m beatgrids entry point
    cli.py                # argparse CLI: analyze + fix commands
    detector.py           # Beat detection via librosa
    analyzer.py           # BPM calculation, drift stats, segment splitting
    stretcher.py          # Per-segment time-stretching via ffmpeg/rubberband
    output.py             # FLAC writing with metadata preservation
  tests/
    __init__.py
    conftest.py           # Shared fixtures (test audio generation)
    test_detector.py      # Beat detection tests
    test_analyzer.py      # BPM calculation tests
    test_stretcher.py     # Segment stretching tests
    test_output.py        # FLAC output + metadata tests
    test_cli.py           # CLI integration tests
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `beatgrids/__init__.py`
- Create: `.gitignore`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "beatgrids"
version = "0.1.0"
description = "Vinyl rip beat grid alignment tool"
requires-python = ">=3.10"
dependencies = [
    "librosa>=0.10",
    "soundfile>=0.12",
    "numpy>=1.24",
]

[project.scripts]
beatgrids = "beatgrids.cli:main"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
]
```

- [ ] **Step 2: Create `beatgrids/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Create `tests/__init__.py`**

Empty file.

- [ ] **Step 4: Create `tests/conftest.py`**

```python
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
```

- [ ] **Step 5: Create `.gitignore`**

```
__pycache__/
*.egg-info/
.eggs/
dist/
build/
*.pyc
.pytest_cache/
```

- [ ] **Step 6: Install project in dev mode**

Run: `cd /Users/scottyeung/Projects/beatgrids && pip install -e ".[dev]"`
Expected: Successful installation with all dependencies.

- [ ] **Step 7: Verify setup**

Run: `cd /Users/scottyeung/Projects/beatgrids && python -c "import beatgrids; print(beatgrids.__version__)"`
Expected: `0.1.0`

- [ ] **Step 8: Commit**

```bash
git init
git add .gitignore pyproject.toml beatgrids/__init__.py tests/__init__.py tests/conftest.py
git commit -m "feat: scaffold beatgrids project with test fixtures"
```

---

### Task 2: Beat Detector

**Files:**
- Create: `beatgrids/detector.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1: Write failing test for beat detection**

```python
# tests/test_detector.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/scottyeung/Projects/beatgrids && pytest tests/test_detector.py -v`
Expected: FAIL — `ImportError: cannot import name 'detect_beats'`

- [ ] **Step 3: Implement `detector.py`**

```python
# beatgrids/detector.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/scottyeung/Projects/beatgrids && pytest tests/test_detector.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add beatgrids/detector.py tests/test_detector.py
git commit -m "feat: add beat detector using librosa"
```

---

### Task 3: BPM Analyzer

**Files:**
- Create: `beatgrids/analyzer.py`
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: Write failing tests for BPM analysis**

```python
# tests/test_analyzer.py
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
    # Beats that speed up: 118 -> 122 BPM
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/scottyeung/Projects/beatgrids && pytest tests/test_analyzer.py -v`
Expected: FAIL — `ImportError: cannot import name 'analyze_beats'`

- [ ] **Step 3: Implement `analyzer.py`**

```python
# beatgrids/analyzer.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/scottyeung/Projects/beatgrids && pytest tests/test_analyzer.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add beatgrids/analyzer.py tests/test_analyzer.py
git commit -m "feat: add BPM analyzer with segment splitting"
```

---

### Task 4: Segment Stretcher

**Files:**
- Create: `beatgrids/stretcher.py`
- Create: `tests/test_stretcher.py`

- [ ] **Step 1: Write failing tests for stretching**

```python
# tests/test_stretcher.py
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
    # 3 segments with different local BPMs
    segments = [
        np.array([0.0, 0.5, 1.0, 1.5]),     # 120 BPM
        np.array([2.0, 2.52, 3.04, 3.56]),   # ~115.4 BPM
        np.array([4.0, 4.48, 4.96, 5.44]),   # 125 BPM
    ]
    target_bpm = 120.0
    ratios = compute_segment_ratios(segments, target_bpm)

    assert len(ratios) == 3
    assert abs(ratios[0] - 1.0) < 0.01   # 120/120
    assert ratios[1] > 1.0               # speed up (target > local)
    assert ratios[2] < 1.0               # slow down (target < local)


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
    # 5% speed up -> ~4.76s expected
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/scottyeung/Projects/beatgrids && pytest tests/test_stretcher.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_segment_ratios'`

- [ ] **Step 3: Implement `stretcher.py`**

```python
# beatgrids/stretcher.py
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from beatgrids.analyzer import build_segments


def compute_segment_ratios(
    segments: list[np.ndarray], target_bpm: float
) -> list[float]:
    """Compute stretch ratio for each segment: target_bpm / local_bpm."""
    ratios = []
    for seg in segments:
        if len(seg) < 2:
            ratios.append(1.0)
            continue
        intervals = np.diff(seg)
        local_bpm = 60.0 / float(np.mean(intervals))
        ratios.append(target_bpm / local_bpm)
    return ratios


def stretch_segment_ffmpeg(
    input_path: str, output_path: str, ratio: float
) -> None:
    """Time-stretch a single audio segment using ffmpeg atempo.

    ratio > 1.0 = speed up, ratio < 1.0 = slow down.
    ffmpeg atempo range is 0.5-2.0; chain filters if needed.
    """
    # Build atempo filter chain for ratios outside 0.5-2.0
    filters = []
    r = ratio
    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r /= 0.5
    filters.append(f"atempo={r:.6f}")

    filter_str = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", filter_str,
        "-ar", str(sf.info(input_path).samplerate),
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def stretch_segment_rubberband(
    input_path: str, output_path: str, ratio: float
) -> None:
    """Time-stretch using rubberband CLI.

    rubberband --tempo expects a tempo ratio (>1 = faster).
    """
    cmd = [
        "rubberband",
        "--tempo", f"{ratio:.6f}",
        input_path,
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def stretch_and_concat(
    input_path: str,
    output_path: str,
    beat_times: np.ndarray,
    target_bpm: float,
    segment_size: int = 16,
    engine: str = "ffmpeg",
) -> None:
    """Full stretch pipeline: segment, stretch, concatenate."""
    audio, sr = sf.read(input_path, dtype="float64")
    info = sf.info(input_path)
    subtype = info.subtype  # e.g. "PCM_24"
    channels = info.channels
    total_samples = len(audio)

    segments = build_segments(beat_times, segment_size)
    ratios = compute_segment_ratios(segments, target_bpm)

    # Define segment boundaries in samples — no overlap between segments.
    # Each segment runs from its first beat to the first beat of the next
    # segment (or end of audio for the last segment).
    boundaries = []
    for i, seg in enumerate(segments):
        start_sec = float(seg[0])
        if i + 1 < len(segments):
            end_sec = float(segments[i + 1][0])  # start of next segment
        else:
            # Last segment: extend one beat past last beat, capped at track end
            end_sec = float(seg[-1]) + 60.0 / target_bpm
        start_sample = int(start_sec * sr)
        end_sample = min(int(end_sec * sr), total_samples)
        boundaries.append((start_sample, end_sample))

    stretch_fn = (
        stretch_segment_ffmpeg if engine == "ffmpeg"
        else stretch_segment_rubberband
    )

    crossfade_samples = int(0.010 * sr)  # 10ms

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        stretched_parts = []

        # Head: audio before first beat
        first_beat_sample = int(float(beat_times[0]) * sr)
        if first_beat_sample > 0:
            head_audio = audio[:first_beat_sample]
            head_path = tmpdir / "head.wav"
            head_out = tmpdir / "head_stretched.wav"
            sf.write(str(head_path), head_audio, sr, subtype=subtype)
            stretch_fn(str(head_path), str(head_out), ratios[0])
            head_stretched, _ = sf.read(str(head_out), dtype="float64")
            stretched_parts.append(head_stretched)

        # Stretch each segment
        for i, ((start, end), ratio) in enumerate(zip(boundaries, ratios)):
            seg_audio = audio[start:end]
            seg_path = tmpdir / f"seg_{i:04d}.wav"
            seg_out = tmpdir / f"seg_{i:04d}_stretched.wav"
            sf.write(str(seg_path), seg_audio, sr, subtype=subtype)
            stretch_fn(str(seg_path), str(seg_out), ratio)
            seg_stretched, _ = sf.read(str(seg_out), dtype="float64")
            stretched_parts.append(seg_stretched)

        # Tail: audio after last beat
        last_beat_sample = int(float(beat_times[-1]) * sr)
        tail_start = boundaries[-1][1] if boundaries else last_beat_sample
        if tail_start < total_samples:
            tail_audio = audio[tail_start:]
            tail_path = tmpdir / "tail.wav"
            tail_out = tmpdir / "tail_stretched.wav"
            sf.write(str(tail_path), tail_audio, sr, subtype=subtype)
            stretch_fn(str(tail_path), str(tail_out), ratios[-1])
            tail_stretched, _ = sf.read(str(tail_out), dtype="float64")
            stretched_parts.append(tail_stretched)

        # Concatenate with overlap-add crossfade
        result = _overlap_add_concat(stretched_parts, crossfade_samples)

        sf.write(output_path, result, sr, subtype=subtype)


def _overlap_add_concat(
    parts: list[np.ndarray], crossfade_len: int
) -> np.ndarray:
    """Concatenate audio parts using true overlap-add crossfade.

    No audio is removed. The overlap region is additive: the tail of
    part N and the head of part N+1 are blended in-place. Total output
    length = sum(len(p)) - crossfade_len * (len(parts) - 1).
    """
    if not parts:
        return np.array([])
    if len(parts) == 1:
        return parts[0]

    # Calculate total output length
    total = sum(len(p) for p in parts) - crossfade_len * (len(parts) - 1)
    ndim = parts[0].ndim
    if ndim == 1:
        result = np.zeros(total, dtype=parts[0].dtype)
    else:
        result = np.zeros((total, parts[0].shape[1]), dtype=parts[0].dtype)

    pos = 0
    for i, part in enumerate(parts):
        if i == 0:
            result[:len(part)] = part
            pos = len(part)
        else:
            cf = min(crossfade_len, pos, len(part))
            fade_out = np.linspace(1.0, 0.0, cf)
            fade_in = np.linspace(0.0, 1.0, cf)

            # Blend the overlap region
            overlap_start = pos - cf
            if ndim == 1:
                result[overlap_start:pos] *= fade_out
                result[overlap_start:pos] += part[:cf] * fade_in
            else:
                result[overlap_start:pos] *= fade_out[:, np.newaxis]
                result[overlap_start:pos] += part[:cf] * fade_in[:, np.newaxis]

            # Append the non-overlapping remainder
            remainder = part[cf:]
            result[pos:pos + len(remainder)] = remainder
            pos += len(remainder)

    return result[:pos]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/scottyeung/Projects/beatgrids && pytest tests/test_stretcher.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add beatgrids/stretcher.py tests/test_stretcher.py
git commit -m "feat: add segment-wise time-stretching with crossfade"
```

---

### Task 5: FLAC Output with Metadata Preservation

**Files:**
- Create: `beatgrids/output.py`
- Create: `tests/test_output.py`

- [ ] **Step 1: Write failing tests for output**

```python
# tests/test_output.py
import soundfile as sf
import subprocess
from pathlib import Path
from beatgrids.output import resolve_output_path, copy_metadata


def test_resolve_output_path_default(tmp_dir):
    """Default output appends _gridded to filename."""
    input_path = tmp_dir / "track.flac"
    input_path.touch()
    result = resolve_output_path(str(input_path), output_dir=None)
    assert result == str(tmp_dir / "track_gridded.flac")


def test_resolve_output_path_custom_dir(tmp_dir):
    """Custom output dir keeps original filename."""
    input_path = tmp_dir / "track.flac"
    input_path.touch()
    out_dir = tmp_dir / "output"
    out_dir.mkdir()
    result = resolve_output_path(str(input_path), output_dir=str(out_dir))
    assert result == str(out_dir / "track.flac")


def test_resolve_output_path_collision(tmp_dir):
    """Collision appends _1, _2, etc."""
    input_path = tmp_dir / "track.flac"
    input_path.touch()
    existing = tmp_dir / "track_gridded.flac"
    existing.touch()
    result = resolve_output_path(str(input_path), output_dir=None)
    assert result == str(tmp_dir / "track_gridded_1.flac")


def test_copy_metadata_preserves_tags(make_click_track, tmp_dir):
    """Vorbis comments should be copied from source to destination."""
    src, _ = make_click_track(bpm=120.0, duration=2.0)

    # Add a tag to source using ffmpeg
    tagged_src = tmp_dir / "tagged.flac"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(src),
        "-metadata", "ARTIST=Test Artist",
        "-metadata", "TITLE=Test Title",
        str(tagged_src),
    ], capture_output=True, check=True)

    # Create a destination file (copy of source without tags)
    dst = tmp_dir / "dest.flac"
    audio, sr = sf.read(str(tagged_src))
    sf.write(str(dst), audio, sr)

    copy_metadata(str(tagged_src), str(dst))

    # Verify tags exist in destination
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format_tags",
         "-of", "compact", str(dst)],
        capture_output=True, text=True,
    )
    assert "Test Artist" in result.stdout
    assert "Test Title" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/scottyeung/Projects/beatgrids && pytest tests/test_output.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_output_path'`

- [ ] **Step 3: Implement `output.py`**

```python
# beatgrids/output.py
import subprocess
import tempfile
from pathlib import Path


def resolve_output_path(input_path: str, output_dir: str | None) -> str:
    """Determine output file path with collision handling."""
    src = Path(input_path)

    if output_dir:
        out = Path(output_dir) / src.name
    else:
        out = src.with_stem(f"{src.stem}_gridded")

    # Handle collisions: use a base stem for counter suffixes
    if not out.exists():
        return str(out)

    base_stem = out.stem
    counter = 1
    while True:
        candidate = out.with_stem(f"{base_stem}_{counter}")
        if not candidate.exists():
            return str(candidate)
        counter += 1


def copy_metadata(source_path: str, dest_path: str) -> None:
    """Copy all FLAC metadata from source to dest.

    Copies Vorbis comments, embedded artwork (PICTURE blocks),
    and other metadata. Uses ffmpeg to remux: keeps dest audio,
    copies metadata + artwork streams from source.
    """
    tmp_out = dest_path + ".tmp.flac"
    cmd = [
        "ffmpeg", "-y",
        "-i", dest_path,
        "-i", source_path,
        "-map", "0:a",          # audio from dest
        "-map", "1:v?",         # artwork/picture from source (if present)
        "-map_metadata", "1",   # metadata from source
        "-c:a", "copy",         # no re-encoding audio
        "-c:v", "copy",         # no re-encoding artwork
        tmp_out,
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    Path(tmp_out).replace(dest_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/scottyeung/Projects/beatgrids && pytest tests/test_output.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add beatgrids/output.py tests/test_output.py
git commit -m "feat: add FLAC output with metadata preservation"
```

---

### Task 6: CLI

**Files:**
- Create: `beatgrids/cli.py`
- Create: `beatgrids/__main__.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for CLI**

```python
# tests/test_cli.py
import subprocess
import sys
from pathlib import Path


def test_cli_analyze(make_click_track):
    """analyze command should print BPM stats."""
    path, _ = make_click_track(bpm=120.0, duration=10.0)
    result = subprocess.run(
        [sys.executable, "-m", "beatgrids", "analyze", str(path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Average BPM" in result.stdout
    assert "Detected beats" in result.stdout


def test_cli_fix_produces_output(make_drift_track, tmp_dir):
    """fix command should produce a _gridded FLAC."""
    path, _ = make_drift_track(start_bpm=119.0, end_bpm=121.0, duration=10.0)
    out_dir = tmp_dir / "output"
    out_dir.mkdir()

    result = subprocess.run(
        [sys.executable, "-m", "beatgrids", "fix", str(path),
         "-o", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0

    outputs = list(out_dir.glob("*.flac"))
    assert len(outputs) == 1


def test_cli_fix_batch(make_click_track, tmp_dir):
    """--batch should process all FLACs in directory."""
    # Create 2 FLAC files in a batch dir
    batch_dir = tmp_dir / "batch"
    batch_dir.mkdir()
    for name in ["track1", "track2"]:
        path, _ = make_click_track(bpm=120.0, duration=5.0)
        import shutil
        shutil.copy(str(path), str(batch_dir / f"{name}.flac"))

    out_dir = tmp_dir / "output"
    out_dir.mkdir()

    result = subprocess.run(
        [sys.executable, "-m", "beatgrids", "fix", "--batch", str(batch_dir),
         "-o", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0

    outputs = list(out_dir.glob("*.flac"))
    assert len(outputs) == 2


def test_cli_no_args_shows_help():
    """Running with no args should show usage."""
    result = subprocess.run(
        [sys.executable, "-m", "beatgrids"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0 or "usage" in result.stderr.lower() or "usage" in result.stdout.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/scottyeung/Projects/beatgrids && pytest tests/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `cli.py`**

```python
# beatgrids/cli.py
import argparse
import sys
from pathlib import Path

from beatgrids.detector import detect_beats
from beatgrids.analyzer import analyze_beats, build_segments, AnalysisResult
from beatgrids.stretcher import stretch_and_concat, compute_segment_ratios
from beatgrids.output import resolve_output_path, copy_metadata


def print_analysis(name: str, result: AnalysisResult, segments, ratios):
    print(f"Analyzing: {name}")
    print(f"  Detected beats: {result.beat_count}")
    print(f"  Average BPM: {result.average_bpm:.2f}")
    print(f"  BPM range: {result.min_bpm:.1f} - {result.max_bpm:.1f} "
          f"(drift: +/-{result.drift_pct:.1f}%)")
    print(f"  Segments: {len(segments)} ({len(segments[0])} beats each)")

    if result.drift_pct > 3.0:
        print("  WARNING: High drift (>3%). Beat detection may be unreliable.")


def do_analyze(args):
    path = Path(args.file)
    beat_times = detect_beats(path, start_bpm=args.target_bpm or 120.0)
    result = analyze_beats(beat_times)
    segments = build_segments(beat_times, args.segment_beats)
    ratios = compute_segment_ratios(segments, args.target_bpm or result.average_bpm)

    print_analysis(path.name, result, segments, ratios)

    # Show per-segment ratios
    print("\n  Per-segment stretch ratios:")
    for i, (seg, ratio) in enumerate(zip(segments, ratios)):
        local_bpm = 60.0 / float((seg[-1] - seg[0]) / (len(seg) - 1)) if len(seg) > 1 else 0
        print(f"    Segment {i+1}: {local_bpm:.2f} BPM -> ratio {ratio:.4f}")


def process_single(file_path: str, args) -> bool:
    """Process a single FLAC file. Returns True on success."""
    path = Path(file_path)
    try:
        beat_times = detect_beats(path, start_bpm=args.target_bpm or 120.0)
        result = analyze_beats(beat_times)
        target_bpm = args.target_bpm or result.average_bpm
        segments = build_segments(beat_times, args.segment_beats)
        ratios = compute_segment_ratios(segments, target_bpm)

        print_analysis(path.name, result, segments, ratios)

        output_path = resolve_output_path(str(path), args.output)

        print(f"Stretching segments...", end=" ", flush=True)
        stretch_and_concat(
            input_path=str(path),
            output_path=output_path,
            beat_times=beat_times,
            target_bpm=target_bpm,
            segment_size=args.segment_beats,
            engine=args.engine,
        )

        # Copy metadata from source
        copy_metadata(str(path), output_path)
        print("done")
        print(f"Output: {output_path}")

        if args.verify:
            print("Verifying output...", end=" ", flush=True)
            verify_beats = detect_beats(output_path)
            verify_result = analyze_beats(verify_beats)
            print(f"Output drift: +/-{verify_result.drift_pct:.2f}%")

        return True
    except Exception as e:
        print(f"ERROR processing {path.name}: {e}", file=sys.stderr)
        return False


def do_fix(args):
    if args.batch:
        batch_dir = Path(args.batch)
        flacs = sorted(batch_dir.glob("*.flac"))
        if not flacs:
            print(f"No FLAC files found in {batch_dir}", file=sys.stderr)
            sys.exit(1)

        successes, failures = 0, 0
        for f in flacs:
            print(f"\n--- {f.name} ---")
            if process_single(str(f), args):
                successes += 1
            else:
                failures += 1

        print(f"\nBatch complete: {successes} succeeded, {failures} failed")
        if failures > 0:
            sys.exit(1)
    else:
        if not process_single(args.file, args):
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="beatgrids",
        description="Vinyl rip beat grid alignment tool",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common options
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--segment-beats", type=int, default=16,
                        help="Beats per segment (default: 16)")
    common.add_argument("--target-bpm", type=float, default=None,
                        help="Override target BPM (default: auto)")
    common.add_argument("--engine", choices=["ffmpeg", "rubberband"],
                        default="ffmpeg", help="Stretch engine")

    # analyze
    analyze_parser = subparsers.add_parser(
        "analyze", parents=[common],
        help="Detect beats and report BPM stats",
    )
    analyze_parser.add_argument("file", help="FLAC file to analyze")
    analyze_parser.set_defaults(func=do_analyze)

    # fix
    fix_parser = subparsers.add_parser(
        "fix", parents=[common],
        help="Stretch to even grid",
    )
    fix_parser.add_argument("file", nargs="?", help="FLAC file to fix")
    fix_parser.add_argument("-o", "--output", help="Output directory")
    fix_parser.add_argument("--batch", help="Process all FLACs in directory")
    fix_parser.add_argument("--verify", action="store_true",
                            help="Verify output grid alignment")
    fix_parser.set_defaults(func=do_fix)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `beatgrids/__main__.py`**

```python
# beatgrids/__main__.py
from beatgrids.cli import main
main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/scottyeung/Projects/beatgrids && pytest tests/test_cli.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add beatgrids/cli.py beatgrids/__main__.py tests/test_cli.py
git commit -m "feat: add CLI with analyze and fix commands"
```

---

### Task 7: Full Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
import numpy as np
import soundfile as sf
import subprocess
import sys
from pathlib import Path
from beatgrids.detector import detect_beats
from beatgrids.analyzer import analyze_beats


def test_full_pipeline_reduces_drift(make_drift_track, tmp_dir):
    """Processing a drifting track should reduce tempo drift."""
    path, _ = make_drift_track(start_bpm=118.0, end_bpm=122.0, duration=20.0)

    # Measure original drift
    original_beats = detect_beats(path)
    original_analysis = analyze_beats(original_beats)

    # Process
    out_dir = tmp_dir / "output"
    out_dir.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "beatgrids", "fix", str(path),
         "-o", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0

    # Measure output drift
    output_file = list(out_dir.glob("*.flac"))[0]
    output_beats = detect_beats(str(output_file))
    output_analysis = analyze_beats(output_beats)

    # Output drift should be lower than input drift
    assert output_analysis.drift_pct < original_analysis.drift_pct


def test_full_pipeline_preserves_sample_rate(make_drift_track, tmp_dir):
    """Output should have same sample rate as input."""
    path, _ = make_drift_track(start_bpm=119.0, end_bpm=121.0, duration=10.0)
    original_info = sf.info(str(path))

    out_dir = tmp_dir / "output"
    out_dir.mkdir()
    subprocess.run(
        [sys.executable, "-m", "beatgrids", "fix", str(path),
         "-o", str(out_dir)],
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
         "-o", str(out_dir), "--target-bpm", "120.0"],
        capture_output=True, text=True, check=True,
    )

    output_file = list(out_dir.glob("*.flac"))[0]
    output_beats = detect_beats(str(output_file))
    output_analysis = analyze_beats(output_beats)

    # Should be close to 120.0
    assert abs(output_analysis.average_bpm - 120.0) < 1.0
```

- [ ] **Step 2: Run integration tests**

Run: `cd /Users/scottyeung/Projects/beatgrids && pytest tests/test_integration.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/scottyeung/Projects/beatgrids && pytest -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add full pipeline integration tests"
```

---

### Task 8: Manual Smoke Test

- [ ] **Step 1: Test with a real FLAC file**

If a real vinyl rip FLAC is available, run:

```bash
cd /Users/scottyeung/Projects/beatgrids
beatgrids analyze /path/to/vinyl_rip.flac
beatgrids fix /path/to/vinyl_rip.flac -o ./output --verify
```

Verify:
- BPM analysis looks reasonable
- Output file is created
- Verification drift is lower than input drift
- Output plays correctly in a media player
- Load in DJ software and confirm grid alignment

- [ ] **Step 2: Final commit if any fixes were needed**
