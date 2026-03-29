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
