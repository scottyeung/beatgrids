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


def stretch_with_timemap(
    input_path: str,
    output_path: str,
    beat_times: np.ndarray,
    target_bpm: float,
) -> None:
    """Warp audio so each detected beat lands on an ideal grid.

    Uses rubberband's --timemap to map every beat from its source
    position to the ideal grid position. This gives per-beat correction
    instead of per-segment averaging.
    """
    info = sf.info(input_path)
    sr = info.samplerate
    total_frames = info.frames

    ideal_interval = 60.0 / target_bpm
    first_beat = float(beat_times[0])

    # Build timemap: (source_frame, target_frame) pairs
    timemap = [(0, 0)]  # anchor start of file

    for i, bt in enumerate(beat_times):
        src = int(float(bt) * sr)
        tgt = int((first_beat + i * ideal_interval) * sr)
        timemap.append((src, tgt))

    # Tail: keep duration after last beat unchanged
    last_src = int(float(beat_times[-1]) * sr)
    last_tgt = int((first_beat + (len(beat_times) - 1) * ideal_interval) * sr)
    tail_frames = total_frames - last_src
    total_output_frames = last_tgt + tail_frames
    timemap.append((total_frames, total_output_frames))

    time_ratio = total_output_frames / total_frames

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write timemap file
        map_path = tmpdir / "timemap.txt"
        with open(map_path, "w") as f:
            for src, tgt in timemap:
                f.write(f"{src} {tgt}\n")

        # Convert to WAV for rubberband
        tmp_in = tmpdir / "input.wav"
        tmp_out = tmpdir / "output.wav"
        audio, _ = sf.read(input_path, dtype="float64")
        sf.write(str(tmp_in), audio, sr, subtype=info.subtype)

        cmd = [
            "rubberband",
            "--timemap", str(map_path),
            "-t", f"{time_ratio:.10f}",
            str(tmp_in),
            str(tmp_out),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        stretched, _ = sf.read(str(tmp_out), dtype="float64")
        sf.write(output_path, stretched, sr, subtype=info.subtype)


def stretch_and_concat(
    input_path: str,
    output_path: str,
    beat_times: np.ndarray,
    target_bpm: float,
    segment_size: int = 16,
    engine: str = "ffmpeg",
) -> None:
    """Full stretch pipeline.

    engine="rubberband" uses per-beat timemap warping (recommended).
    engine="ffmpeg" uses per-segment uniform stretching (fallback).
    """
    if engine == "rubberband":
        return stretch_with_timemap(
            input_path, output_path, beat_times, target_bpm
        )

    # ffmpeg path: segment-based stretching
    audio, sr = sf.read(input_path, dtype="float64")
    info = sf.info(input_path)
    subtype = info.subtype
    total_samples = len(audio)

    segments = build_segments(beat_times, segment_size)
    ratios = compute_segment_ratios(segments, target_bpm)

    # Define segment boundaries in samples — no overlap between segments.
    boundaries = []
    for i, seg in enumerate(segments):
        start_sec = float(seg[0])
        if i + 1 < len(segments):
            end_sec = float(segments[i + 1][0])
        else:
            end_sec = float(seg[-1]) + 60.0 / target_bpm
        start_sample = int(start_sec * sr)
        end_sample = min(int(end_sec * sr), total_samples)
        boundaries.append((start_sample, end_sample))

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
            stretch_segment_ffmpeg(str(head_path), str(head_out), ratios[0])
            head_stretched, _ = sf.read(str(head_out), dtype="float64")
            stretched_parts.append(head_stretched)

        # Stretch each segment
        for i, ((start, end), ratio) in enumerate(zip(boundaries, ratios)):
            seg_audio = audio[start:end]
            seg_path = tmpdir / f"seg_{i:04d}.wav"
            seg_out = tmpdir / f"seg_{i:04d}_stretched.wav"
            sf.write(str(seg_path), seg_audio, sr, subtype=subtype)
            stretch_segment_ffmpeg(str(seg_path), str(seg_out), ratio)
            seg_stretched, _ = sf.read(str(seg_out), dtype="float64")
            stretched_parts.append(seg_stretched)

        # Tail: audio after last beat
        tail_start = boundaries[-1][1] if boundaries else int(float(beat_times[-1]) * sr)
        if tail_start < total_samples:
            tail_audio = audio[tail_start:]
            tail_path = tmpdir / "tail.wav"
            tail_out = tmpdir / "tail_stretched.wav"
            sf.write(str(tail_path), tail_audio, sr, subtype=subtype)
            stretch_segment_ffmpeg(str(tail_path), str(tail_out), ratios[-1])
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
