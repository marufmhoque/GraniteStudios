"""Video ingestion: containers and editing codecs are recognised.

Clips are encoded here with the bundled ffmpeg, so these are real files in
each codec rather than stand-ins. An encoder the bundled build lacks skips
its case instead of failing.
"""

from __future__ import annotations

import pytest

from src.engine.universal_loader import UniversalLoader
from src.video.ffmpeg_utils import FFmpegError, is_video, probe, run_ffmpeg

_SOURCE = ["-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=25"]

CASES = [
    ("h264.mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p"], "H.264", False),
    ("h264.mov", ["-c:v", "libx264", "-pix_fmt", "yuv420p"], "H.264", False),
    ("h264.mkv", ["-c:v", "libx264", "-pix_fmt", "yuv420p"], "H.264", False),
    ("vp9.webm", ["-c:v", "libvpx-vp9", "-b:v", "200k"], "VP9", False),
    ("prores422.mov", ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le"], "ProRes 422", True),
    ("prores4444.mov", ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"], "ProRes 4444", True),
    ("dnxhd.mov", ["-c:v", "dnxhd", "-b:v", "36M", "-s", "1920x1080", "-r", "25",
                   "-pix_fmt", "yuv422p"], "DNxHD", True),
]


@pytest.mark.slow
@pytest.mark.parametrize("name,codec_args,label,intermediate", CASES, ids=[c[0] for c in CASES])
def test_codec_is_recognised(tmp_path, name, codec_args, label, intermediate):
    path = tmp_path / name
    try:
        run_ffmpeg([*_SOURCE, *codec_args, str(path)])
    except FFmpegError as exc:
        pytest.skip(f"bundled ffmpeg cannot encode this: {str(exc)[-120:]}")

    info = probe(path)
    assert info.width > 0 and info.height > 0 and info.duration_sec > 0.5
    assert info.codec_label == label, f"fourcc {info.codec_fourcc!r} read as {info.codec_label!r}"
    assert info.is_intermediate_codec is intermediate
    assert is_video(path)


def test_scan_videos_separates_clips_from_stills(tmp_path):
    for name in ("a.mp4", "b.MOV", "c.mkv", "d.webm", "still.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    loader = UniversalLoader()
    assert [p.name for p in loader.scan_videos(tmp_path)] == ["a.mp4", "b.MOV", "c.mkv", "d.webm"]
    assert [p.name for p in loader.scan_folder(tmp_path)] == ["still.jpg"]
    assert loader.scan_videos(tmp_path / "missing") == []
