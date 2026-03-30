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


def quantize_bpm(bpm: float, step: float) -> float:
    """Round BPM to the nearest multiple of step."""
    return round(bpm / step) * step


def process_single(file_path: str, args) -> bool:
    """Process a single FLAC file. Returns True on success."""
    path = Path(file_path)
    try:
        beat_times = detect_beats(path, start_bpm=args.target_bpm or 120.0)
        result = analyze_beats(beat_times)
        target_bpm = args.target_bpm or result.average_bpm
        if args.quantize is not None:
            target_bpm = quantize_bpm(target_bpm, args.quantize)
            print(f"  Quantized BPM: {target_bpm:.2f} (step={args.quantize})")
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
                        default="rubberband",
                        help="Stretch engine (default: rubberband)")

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
    fix_parser.add_argument("--quantize", type=float, nargs="?", const=1.0,
                            default=None,
                            help="Round BPM to nearest step (default step: 1.0)")
    fix_parser.add_argument("--verify", action="store_true",
                            help="Verify output grid alignment")
    fix_parser.set_defaults(func=do_fix)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
