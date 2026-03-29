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
