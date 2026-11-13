"""Fetch the real camera files the ``media`` tests need.

    python tests/fetch_media.py            # everything
    python tests/fetch_media.py cr3 dng    # only matching names

The files are public-domain (CC0) samples from https://raw.pixls.us. They are
written to ``~/.granitestudios/test-media/raw`` (override with the
``GRANITE_TEST_MEDIA`` environment variable) and each is verified against the
SHA-256 published alongside it. They are kept out of the repository folder on
purpose: it may sit on a synced drive, which would upload 190 MB of samples
and make every timing measure the sync client instead of the code. Nothing here runs during a normal
test session: tests that need a file skip when it is absent.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MEDIA_DIR = Path(os.environ.get("GRANITE_TEST_MEDIA", Path.home() / ".granitestudios" / "test-media"))
RAW_DIR = MEDIA_DIR / "raw"
_BASE = "https://raw.pixls.us/getfile.php"


@dataclass(frozen=True)
class Sample:
    name: str          # local filename
    remote: str        # path under getfile.php
    sha256: str
    megabytes: float
    note: str = ""

    folder: str = "raw"
    base: str = _BASE

    @property
    def path(self) -> Path:
        return MEDIA_DIR / self.folder / self.name

    @property
    def url(self) -> str:
        return f"{self.base}/{urllib.parse.quote(self.remote)}"

    @property
    def present(self) -> bool:
        return self.path.exists()


SAMPLES = [
    Sample("canon_eos_r6.cr3", "4661/nice/Canon - EOS R6 - 3:2.CR3",
           "6156ee9cfd1a61e08719a8309095d05162891fd2adb8dd1505b31d4716dcee05", 9.36),
    Sample("canon_5d_mark_iv.cr2", "981/nice/Canon - EOS 5D Mark IV - sRAW2 (sRAW) (3:2).CR2",
           "25d928f9e9a65525ac796bdff13e93b7b36b627568d7a17338e12dc44424078a", 14.51),
    Sample("nikon_z50_2.nef", "7767/nice/Nikon - Z50_2 - 8bit compressed (1:1).NEF",
           "99b2a569356fb084d6946fa6e0b501b014e6e004be0a8e8d564a7b9dab91857e", 8.05,
           "2025 body; newer than the bundled LibRaw"),
    Sample("sony_a6400.arw", "3406/nice/Sony - ILCE-6400 - 12bit 12bit compressed (3:2).ARW",
           "8a37c80fc47c6026efa2e55279220e79460e2f470b1aa48b440f8f2d47873aa6", 23.62),
    Sample("fuji_x_s10.raf", "4190/nice/Fujifilm - X-S10 - 14bit 14bit compressed (3:2).RAF",
           "9419d1408ebf850395e5dd563beb96309466546252caf541466034c9cb7e9724", 17.81),
    Sample("panasonic_gh5s.rw2", "2610/nice/Panasonic - DC-GH5S - 1:1.RW2",
           "3fdf6ce5e14382801e8998e66b290b2f65d0c4fe4a199922742eb12c72fc8403", 14.07),
    Sample("olympus_em1_ii.orf", "1993/nice/Olympus - E-M1MarkII - 16bit (4:3).ORF",
           "0b433019e5fa61548d7eb0eb7bb90894251542c056da72355202382696d0ceb8", 16.46),
    Sample("pentax_k1_ii.pef", "7133/nice/Pentax - K-1 Mark II - 14bit (3:2).PEF",
           "fe2330254c5313d4dde97cbd01e2987f6f1137f1c9c687d30e951f4668979381", 22.86),
    Sample("hasselblad_h3d.3fr", "2851/nice/Hasselblad - H3D - 16bit (4:3).3FR",
           "34fdb93cb215e4f419ae0f7252c57b881fa381003124b4f9c143d27741f211b3", 51.11),
    Sample("apple_iphone_xs.dng", "3372/nice/Apple - iPhone XS - 16bit (4:3).dng",
           "50bc64ee8cfbbfb6c0531e29ab428ff985506a16fe2d1d2e74656f21b54505f6", 9.81,
           "phone DNG; predates ProRAW"),
    Sample("samsung_s10.dng", "3839/nice/Samsung - SM-G973U - 16bit 16bit (2.1132075471698).dng",
           "6492b7e0ac8e012524c0c83606a104cc9bf3b68f30fa60c7b73ffbe4dc584ab1", 1.26,
           "phone DNG"),
]


_GITHUB = "https://raw.githubusercontent.com"

# Reference faces: sample images that ship with OpenCV and its test data.
FACES = [
    Sample("grace_hopper_227.png", "opencv/opencv_extra/4.x/testdata/dnn/grace_hopper_227.png",
           "f9f5173e165918db972a36b61f73d9548f43a9859ebbcc4b5603c887f5bb6e44", 0.09,
           "frontal, glasses, neutral", "faces", _GITHUB),
    Sample("messi5.jpg", "opencv/opencv/4.x/samples/data/messi5.jpg",
           "1d570e49654e84c7a943918537bd9e5e1ef82920152e147c834006e235be97c9", 0.07,
           "small face, looking down, mouth open", "faces", _GITHUB),
    Sample("basketball1.png", "opencv/opencv/4.x/samples/data/basketball1.png",
           "ba06f6701f7260998b430c39b6557f775497e6ce7b1a74f0b7ea6af371bf54a6", 0.13,
           "small face, three-quarter view", "faces", _GITHUB),
    Sample("pose.png", "opencv/opencv_extra/4.x/testdata/dnn/pose.png",
           "447278bfa797dbc4d53b351c4978b63d980ba46ed92c4213c850f487d2765416", 0.15,
           "frontal, broad smile", "faces", _GITHUB),
]


def face(name: str) -> Sample:
    return next(s for s in FACES if s.name == name)


def available() -> list[Sample]:
    return [s for s in SAMPLES if s.present]


def fetch(sample: Sample, timeout: int = 300) -> bool:
    if sample.present:
        return True
    sample.path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(sample.url, headers={"User-Agent": "sample-fetcher"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except Exception as exc:  # network trouble is not a test failure
        print(f"  FAILED {sample.name}: {exc}")
        return False
    digest = hashlib.sha256(data).hexdigest()
    if digest != sample.sha256:
        print(f"  REJECTED {sample.name}: checksum mismatch")
        return False
    sample.path.write_bytes(data)
    return True


def main(argv: list[str]) -> int:
    wanted = [s for s in SAMPLES + FACES if not argv or any(a.lower() in s.name for a in argv)]
    total = sum(s.megabytes for s in wanted if not s.present)
    print(f"{len(wanted)} sample(s), {total:.0f} MB to download into {MEDIA_DIR}")
    ok = 0
    for sample in wanted:
        print(f"  {sample.name}  ({sample.megabytes:.1f} MB)")
        ok += fetch(sample)
    print(f"{ok}/{len(wanted)} present")
    return 0 if ok == len(wanted) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
