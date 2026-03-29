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
    """fix command should produce a FLAC in the output dir."""
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
