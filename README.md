# beatgrids

Fixes tempo drift in vinyl rip FLACs so beat grids lock in DJ software (Traktor, Rekordbox, etc).

Vinyl rips have slight speed variation from turntable motor drift, causing beat grid markers to gradually misalign. This tool detects every beat, computes the average BPM, then time-stretches the audio so each beat lands on a perfect grid — without changing the pitch.

## How it works

1. **Beat detection** — librosa finds beat positions, then onset strength refinement snaps each to sub-frame precision
2. **Grid warping** — rubberband's timemap warps every detected beat to its ideal grid position in a single pass
3. **Output** — writes a new FLAC with metadata and artwork preserved

98% of beats land within 10ms of the ideal grid on tested tracks.

## Requirements

- Python 3.10+
- [Rubber Band](https://breakfastquay.com/rubberband/) (`brew install rubberband`)
- ffmpeg (fallback engine, `brew install ffmpeg`)

## Install

```bash
git clone https://github.com/scottyeung/beatgrids.git
cd beatgrids
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Analyze a track (no changes)
beatgrids analyze track.flac

# Fix a single track
beatgrids fix track.flac -o output/

# Fix with BPM rounded to nearest integer
beatgrids fix track.flac -o output/ --quantize

# Fix with BPM rounded to nearest 0.5
beatgrids fix track.flac -o output/ --quantize 0.5

# Set a specific target BPM
beatgrids fix track.flac -o output/ --target-bpm 124.0

# Batch process a directory
beatgrids fix --batch ~/vinyl-rips/ -o output/

# Verify output grid alignment
beatgrids fix track.flac -o output/ --verify

# Use ffmpeg engine instead of rubberband
beatgrids fix track.flac -o output/ --engine ffmpeg
```

## Options

| Flag | Description |
|------|-------------|
| `--quantize [step]` | Round BPM to nearest step (default: 1.0) |
| `--target-bpm` | Override detected BPM with a specific value |
| `--engine` | `rubberband` (default, per-beat warping) or `ffmpeg` (per-segment) |
| `--segment-beats` | Beats per segment for ffmpeg engine (default: 16) |
| `--verify` | Re-analyze output to check grid alignment |
| `-o / --output` | Output directory |
| `--batch` | Process all FLACs in a directory |

## Tests

```bash
pytest
```
