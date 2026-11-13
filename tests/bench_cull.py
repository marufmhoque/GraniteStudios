"""Time a cull over many RAW files.

    python tests/bench_cull.py                 # 1,000 frames per format
    python tests/bench_cull.py --frames 200 --no-eyes
    python tests/bench_cull.py --baseline      # also time the old full-demosaic path

This is a measuring tool, not a test. It builds a folder of hard links to
the sample files from ``tests/fetch_media.py`` -- so a thousand "frames"
cost no disk space -- and runs the real cull over it.

Read the numbers for what they are: every frame in a run is the same file,
and after the first read it comes from the operating system's cache. That
makes this a measurement of decode and analysis throughput, which is what
the code controls. It is not a measurement of reading a thousand different
files from a memory card or a hard disk.

Recorded results
----------------
13 November 2026. Intel Core i5, 8th generation U-series (4 cores, 8
threads), 32 GB RAM, 7 workers. 1,000 frames per run, eye checking on (face
detection, 478-point mesh, keeper score). Seconds per 1,000 frames:

    camera                 embedded preview   first build   final
    Sony a6400 (ARW)          1616 x 1080         100          42
    Samsung S10 (DNG)         4016 x 1892          95          51
    Olympus E-M1 II (ORF)     3200 x 2400         115          53
    Canon EOS R6 (CR3)        3408 x 2272         111          61
    Panasonic GH5S (RW2)      1920 x 1920         163          85
    Fujifilm X-S10 (RAF)      4416 x 2944         166          91
    Nikon Z50 II (NEF)        3712 x 3712         224          96
    Canon 5D IV (CR2)         6720 x 4480         258         100
    Pentax K-1 II (PEF)       4800 x 3200         151         104

Against a 90 second target that is five of nine cameras. Against the earlier
five minute target it is all nine. The Panasonic frame contains two faces,
which is why a small preview still costs 85 s: each face adds a mesh pass.

Two files have no usable embedded preview and are developed at half size
instead. First build only, not re-run: iPhone XS DNG 291 s, Hasselblad H3D
616 s per 1,000.

"First build" decoded each preview once in colour at full size. "Final"
decodes greyscale at full size for sharpness and reduced colour for faces,
and finds faces on an 800 px copy.

What did not help, so it is not tried again blind: running OpenCV
single-threaded inside each worker (under 3 percent), and finding faces at
640 px instead of 800 (about 8 percent, at the cost of small faces).
Measuring faces on the greyscale decode alone would remove a decode, but it
moved the eye aspect ratio on two of four reference faces (0.155 to 0.227 on
one), so the thresholds would no longer hold.

For comparison, fully developing one of these files takes 0.8 to 7.8 s.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.fetch_media import MEDIA_DIR, available  # noqa: E402

BENCH_DIR = MEDIA_DIR / "bench"


def build_folder(sample, frames: int) -> Path:
    folder = BENCH_DIR / sample.path.stem
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for index in range(frames):
        target = folder / f"IMG_{index:04d}{sample.path.suffix}"
        try:
            os.link(sample.path, target)
        except OSError:
            shutil.copyfile(sample.path, target)
    return folder


def machine() -> str:
    try:
        import psutil

        ram = f"{psutil.virtual_memory().total / 1024 ** 3:.0f} GB RAM"
    except Exception:
        ram = "RAM unknown"
    return f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical cores, {ram}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--no-eyes", action="store_true", help="skip the face / eye check")
    parser.add_argument("--baseline", action="store_true", help="time the full-demosaic path too")
    parser.add_argument("--only", nargs="*", default=[], help="substrings of sample names")
    args = parser.parse_args()

    from src.ai_tools.culling import SmartCull
    from src.engine import fast_preview
    from src.engine.universal_loader import UniversalLoader

    samples = [s for s in available() if not args.only or any(o in s.name for o in args.only)]
    if not samples:
        print("No samples present. Run: python tests/fetch_media.py")
        return 1

    eyes = not args.no_eyes
    print(f"Machine: {machine()}")
    print(f"Workers: {fast_preview.default_workers()}   Frames per run: {args.frames}   Eye check: {eyes}")
    print()
    print(f"{'sample':24s} {'preview':15s} {'native':>12s} {'total s':>8s} {'ms/frame':>9s} {'per 1,000':>10s}"
          + ("  demosaic ms" if args.baseline else ""))

    culler = SmartCull()
    for sample in samples:
        probe = fast_preview.load_preview(sample.path, max_side=None, exact=True)
        folder = build_folder(sample, args.frames)
        try:
            started = time.perf_counter()
            groups = culler.cull_folder(folder, check_eyes=eyes)
            elapsed = time.perf_counter() - started
            scored = sum(len(g.frames) for g in groups)
            assert scored == args.frames, f"scored {scored} of {args.frames}"
            per_frame = elapsed / args.frames * 1e3
            line = (
                f"{sample.name:24s} {probe.source if probe else 'unreadable':15s} "
                f"{'x'.join(map(str, probe.native_size)) if probe else '-':>12s} "
                f"{elapsed:8.1f} {per_frame:9.1f} {per_frame:9.0f}s"
            )
            if args.baseline:
                loader = UniversalLoader()
                t = time.perf_counter()
                full = loader.load_array(sample.path)
                line += f"  {(time.perf_counter() - t) * 1e3:10.0f}" + ("" if full is not None else " (failed)")
            print(line, flush=True)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
