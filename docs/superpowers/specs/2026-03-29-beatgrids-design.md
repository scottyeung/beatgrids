# Beatgrids: Vinyl Rip Beat Grid Alignment Tool

## Problem

Vinyl rips suffer from turntable speed drift — the motor, belt, and physical medium cause micro-tempo variations throughout the track. DJ software (Traktor, Serato, Rekordbox) assumes constant tempo and lays an evenly-spaced beat grid, but it drifts out of alignment with the actual beats as the track plays.

## Solution

A Python CLI tool that detects beats across a FLAC file, computes the average BPM, and time-stretches the audio segment by segment so every beat lands on an even grid at that averaged BPM. The output is a new FLAC file with consistent tempo that grids perfectly in any DJ software.

## Architecture

```
Input FLAC
    |
    v
+---------------------+
|  Beat Detection     |  librosa.beat.beat_track()
|  (full track)       |  -> array of beat timestamps (seconds)
+----------+----------+
           |
           v
+---------------------+
|  Average BPM Calc   |  mean(inter-beat intervals)
|                     |  -> target BPM (raw float, no rounding)
+----------+----------+
           |
           v
+---------------------+
|  Segment Splitting  |  Group beats into segments of N beats
|                     |  Calculate local BPM per segment
+----------+----------+
           |
           v
+---------------------+
|  Per-Segment        |  ratio = target_BPM / local_BPM
|  Time-Stretch       |  ffmpeg atempo filter per segment
+----------+----------+
           |
           v
+---------------------+
|  Concatenate +      |  Join segments with overlap-add
|  Crossfade          |  crossfade (no audio removed)
+----------+----------+
           |
           v
+---------------------+
|  Output FLAC        |  Preserve all metadata, tags, artwork
|                     |  Write to output directory
+---------------------+
```

## Components

### 1. Beat Detector (`detector.py`)

Loads a FLAC file via librosa **for analysis only** (downmixed/resampled copy). Audio I/O for stretching uses `soundfile` to preserve the original sample rate and bit depth. Runs `beat_track()` to get an array of beat frame positions, converts to timestamps in seconds.

**Inputs:** File path, optional `hop_length` and `start_bpm` overrides.

**Outputs:** Array of beat timestamps, detected BPM estimate.

**Edge cases:**
- Tracks with fewer than 16 beats: use segment size of 4 or process as a single segment.
- Beat detection confidence: report the standard deviation of inter-beat intervals as a "drift" metric. High drift (>3%) triggers a warning.

### 2. BPM Calculator (`analyzer.py`)

Takes beat timestamps, computes inter-beat intervals, derives average BPM.

**Outputs:**
- Average BPM (raw float, not rounded)
- BPM range (min/max local BPM)
- Drift percentage (how much the tempo varies)
- Total beat count

### 3. Segment Stretcher (`stretcher.py`)

Splits the audio at segment boundaries (every N beats, default 16). For each segment:

1. Calculate local BPM from the inter-beat intervals within that segment.
2. Compute stretch ratio: `target_bpm / local_bpm`.
3. Extract segment audio to a temp WAV **matching the source format** (same sample rate, bit depth).
4. Apply ffmpeg `atempo` filter with the computed ratio.
5. Collect stretched segments.

**Head and tail handling:** Audio before the first detected beat ("head") and after the last detected beat ("tail") are not dropped. The head is stretched at the first segment's ratio; the tail at the last segment's ratio.

**Final segment handling:** The last segment may contain fewer than N beats. If it contains fewer than 4 beats, it is merged with the preceding segment. Local BPM for a segment is computed as the mean of inter-beat intervals within that segment.

**Concatenation:** Segments are joined using overlap-add crossfading (10ms overlap). The crossfade region is additive — no audio is removed, so cumulative timing is not affected.

**Intermediate format:** All temporary WAV files match the source FLAC's sample rate and bit depth (e.g. 24-bit/96kHz). librosa is used only for beat detection (downmixed analysis copy); `soundfile` handles all audio I/O to preserve resolution.

**ffmpeg atempo constraints:** Limited to 0.5x-2.0x range. Vinyl drift is typically <2%, so this is not a concern. If a segment somehow exceeds the range, chain multiple atempo filters.

**Rubberband fallback:** When `--engine rubberband` is passed, use `rubberband` CLI instead of ffmpeg atempo for higher-quality stretching. Same segment logic, different stretch command. Recommended for critical listening — ffmpeg's atempo uses WSOLA (optimized for speech), while rubberband is designed for music.

**Temporary file cleanup:** All temporary files are created in a system temp directory (`tempfile.TemporaryDirectory`) and cleaned up on completion or failure.

### 4. FLAC Writer (`output.py`)

Writes the concatenated audio to a new FLAC file. Copies all metadata from the source:
- Vorbis comments (artist, title, album, etc.)
- Embedded artwork (PICTURE blocks)
- ReplayGain tags if present

**Output naming:**
- Default: `<original_name>_gridded.flac` in the same directory.
- With `-o`: write to specified directory, keeping original filename.
- Collision handling: if output file exists, append `_1`, `_2`, etc.

### 5. CLI (`cli.py`)

Entry point using `argparse`.

**Commands:**

```
beatgrids analyze <file>
    Detect beats and report BPM stats without modifying audio.
    Shows per-segment stretch ratios for preview.

beatgrids fix <file> [options]
    Detect beats, stretch to even grid, output corrected FLAC.

beatgrids fix --batch <directory> [options]
    Process all FLAC files in directory.
    Errors on individual files are logged; processing continues.
    A summary of successes and failures is printed at the end.
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | same dir | Output directory |
| `--segment-beats` | 16 | Beats per segment |
| `--target-bpm` | auto | Override target BPM (default: computed average) |
| `--engine` | ffmpeg | Stretch engine: `ffmpeg` or `rubberband` |
| `--verify` | false | Run beat detection on output to confirm grid alignment |

**Example output:**

```
Analyzing: Is It Cool (Theo Parrish Re-Edit).flac
  Detected beats: 532
  Average BPM: 119.49
  BPM range: 118.8 - 120.1 (drift: +/-0.5%)
  Segments: 33 (16 beats each)
Stretching segments... [================] 33/33
Output: corrected/Is It Cool (Theo Parrish Re-Edit)_gridded.flac
```

## Project Structure

```
beatgrids/
  pyproject.toml
  beatgrids/
    __init__.py
    cli.py          # argparse entry point
    detector.py     # beat detection via librosa
    analyzer.py     # BPM calculation and stats
    stretcher.py    # segment-wise time-stretching
    output.py       # FLAC writing with metadata preservation
```

## Dependencies

**Required:**
- Python 3.10+
- `librosa` — beat detection
- `soundfile` — audio I/O (FLAC read/write)
- `numpy` — numerical operations
- `ffmpeg` — time-stretching (must be on PATH)

**Optional:**
- `rubberband-cli` — higher quality time-stretch engine

## Testing Strategy

- Unit tests for BPM calculation (known intervals -> expected BPM).
- Unit tests for segment splitting (edge cases: short tracks, odd beat counts).
- Integration test: process a short FLAC, verify output has consistent inter-beat intervals.
- Manual validation: load output in DJ software, confirm grid alignment.

## Limitations

- Beat detection depends on librosa's algorithm. Sparse, ambient, or polyrhythmic tracks may produce poor results. The `analyze` command lets users verify before committing.
- Time-stretching is destructive. Original timing nuances from the vinyl are flattened.
- Very large drift (>3%) suggests the vinyl rip has more fundamental issues (wrong RPM, heavy warping) and may need manual intervention.
