"""Phone formats, on files built here to the published layouts.

These prove the parsers and the gain-map arithmetic are self-consistent.
They do not prove compatibility with a file from a real phone; the PRD
records each of these formats as unverified until one is tested.
"""

from __future__ import annotations

import io
from datetime import datetime

import cv2
import numpy as np
import pytest
from PIL import Image

from src.engine import phone_media as pm
from src.engine.universal_loader import UniversalLoader
from tests.conftest import assert_originals_untouched, make_scene, snapshot
from tests.fetch_media import SAMPLES


def _jpeg(size=(320, 240), seed=0, xmp: str | None = None, grey: int | None = None) -> bytes:
    if grey is None:
        image = Image.fromarray(cv2.cvtColor(make_scene(seed, size), cv2.COLOR_BGR2RGB))
    else:
        image = Image.new("L", size, grey)
    buffer = io.BytesIO()
    extra = {"xmp": xmp.encode("utf-8")} if xmp else {}
    image.save(buffer, "JPEG", quality=95, **extra)
    return buffer.getvalue()


def _xmp(body: str) -> str:
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f"{body}</rdf:RDF></x:xmpmeta>"
    )


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> bytes:
    from src.video.ffmpeg_utils import run_ffmpeg

    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
                "-pix_fmt", "yuv420p", str(path)])
    return path.read_bytes()


# --------------------------------------------------------------------- #
# Live Photos
# --------------------------------------------------------------------- #

def test_live_photos_pair_by_name(tmp_path):
    for name in ("IMG_1.HEIC", "IMG_1.MOV", "IMG_2.jpg", "IMG_2.mov", "IMG_3.heic", "clip.mov"):
        (tmp_path / name).write_bytes(b"x")
    pairs = pm.find_live_photo_pairs(tmp_path)
    assert {k.name: v.name for k, v in pairs.items()} == {"IMG_1.HEIC": "IMG_1.MOV", "IMG_2.jpg": "IMG_2.mov"}
    assert pm.find_live_photo_pairs(tmp_path / "missing") == {}


# --------------------------------------------------------------------- #
# Motion Photos
# --------------------------------------------------------------------- #

def test_motion_photo_located_from_the_container_list(tmp_path, clip):
    xmp = _xmp(
        '<rdf:Description GCamera:MotionPhoto="1"><Container:Directory><rdf:Seq>'
        '<rdf:li><Container:Item Item:Mime="image/jpeg" Item:Semantic="Primary"/></rdf:li>'
        f'<rdf:li><Container:Item Item:Mime="video/mp4" Item:Semantic="MotionPhoto" Item:Length="{len(clip)}"/></rdf:li>'
        "</rdf:Seq></Container:Directory></rdf:Description>"
    )
    still = _jpeg(xmp=xmp)
    path = tmp_path / "PXL_0001.MP.jpg"
    path.write_bytes(still + clip)
    assert pm.motion_video_range(path) == (len(still), len(clip))


def test_motion_photo_located_from_the_legacy_offset(tmp_path, clip):
    still = _jpeg(xmp=_xmp(f'<rdf:Description GCamera:MicroVideo="1" GCamera:MicroVideoOffset="{len(clip)}"/>'))
    path = tmp_path / "MVIMG_0001.jpg"
    path.write_bytes(still + clip)
    assert pm.motion_video_range(path) == (len(still), len(clip))


def test_motion_photo_found_structurally_when_the_xmp_is_missing(tmp_path, clip):
    still = _jpeg()
    path = tmp_path / "samsung.jpg"
    path.write_bytes(still + clip)
    assert pm.motion_video_range(path) == (len(still), len(clip))


def test_an_ordinary_still_has_no_motion_clip(tmp_path):
    path = tmp_path / "plain.jpg"
    path.write_bytes(_jpeg())
    assert pm.motion_video_range(path) is None
    assert pm.motion_video_range(tmp_path / "missing.jpg") is None


def test_extracting_the_clip_leaves_the_still_untouched(tmp_path, clip):
    folder = tmp_path / "shoot"
    folder.mkdir()
    path = folder / "PXL_0001.MP.jpg"
    path.write_bytes(_jpeg() + clip)
    before = snapshot(folder)
    out = pm.extract_motion_video(path, folder / "shoot-edit" / "PXL_0001.mp4")
    assert out.read_bytes() == clip
    assert_originals_untouched(before, folder)


# --------------------------------------------------------------------- #
# Gain maps
# --------------------------------------------------------------------- #

def _ultra_hdr(tmp_path, params: str, map_grey: int = 255, sdr_seed: int = 1):
    gain_bytes = _jpeg(size=(160, 120), grey=map_grey, xmp=_xmp(f"<rdf:Description {params}/>"))
    primary_xmp = _xmp(
        '<rdf:Description hdrgm:Version="1.0"><Container:Directory><rdf:Seq>'
        '<rdf:li><Container:Item Item:Semantic="Primary" Item:Mime="image/jpeg"/></rdf:li>'
        f'<rdf:li><Container:Item Item:Semantic="GainMap" Item:Mime="image/jpeg" Item:Length="{len(gain_bytes)}"/></rdf:li>'
        "</rdf:Seq></Container:Directory></rdf:Description>"
    )
    path = tmp_path / "ultra.jpg"
    path.write_bytes(_jpeg(seed=sdr_seed, xmp=primary_xmp) + gain_bytes)
    return path


def test_gain_map_parameters_are_read(tmp_path):
    path = _ultra_hdr(
        tmp_path,
        'hdrgm:Version="1.0" hdrgm:GainMapMin="0" hdrgm:GainMapMax="2.5" hdrgm:Gamma="1" '
        'hdrgm:OffsetSDR="0.015625" hdrgm:OffsetHDR="0.015625" '
        'hdrgm:HDRCapacityMin="0" hdrgm:HDRCapacityMax="2.5"',
    )
    gain = pm.read_gain_map(path)
    assert gain is not None
    assert gain.gain_max == (2.5, 2.5, 2.5) and gain.capacity_max == 2.5
    assert gain.image.shape == (120, 160)
    assert gain.max_boost == pytest.approx(2 ** 2.5)
    assert pm.inspect(path).gain_map


def test_per_channel_values_written_as_a_sequence(tmp_path):
    seq = "<hdrgm:GainMapMax><rdf:Seq><rdf:li>1.0</rdf:li><rdf:li>2.0</rdf:li><rdf:li>3.0</rdf:li></rdf:Seq></hdrgm:GainMapMax>"
    packet = _xmp(f'<rdf:Description hdrgm:HDRCapacityMax="3.0">{seq}</rdf:Description>')
    assert pm._xmp_values(packet, "hdrgm:GainMapMax") == [1.0, 2.0, 3.0]
    assert pm._triple([1.0, 2.0, 3.0], 0.0) == (1.0, 2.0, 3.0)


def test_a_plain_jpeg_has_no_gain_map(tmp_path):
    path = tmp_path / "plain.jpg"
    path.write_bytes(_jpeg())
    assert pm.read_gain_map(path) is None
    assert pm.load_hdr(path) is None


def test_gain_arithmetic_at_the_extremes():
    sdr = np.full((4, 4, 3), 0.25, np.float32)
    offset = 1 / 64

    def gain(value):
        return pm.GainMap(image=np.full((4, 4), value, np.float32),
                          gain_min=(0, 0, 0), gain_max=(2, 2, 2), capacity_min=0.0, capacity_max=2.0)

    # No recovery: the offsets cancel and the SDR image comes back.
    np.testing.assert_allclose(pm.apply_gain_map(sdr, gain(0.0)), sdr, rtol=1e-5)
    # Full recovery on a capable display: two stops, so four times brighter.
    np.testing.assert_allclose(pm.apply_gain_map(sdr, gain(1.0)), (0.25 + offset) * 4 - offset, rtol=1e-5)
    # A display with half the range (in stops) applies half the boost.
    np.testing.assert_allclose(
        pm.apply_gain_map(sdr, gain(1.0), display_boost=2.0), (0.25 + offset) * 2 - offset, rtol=1e-5
    )
    # An SDR display applies none.
    np.testing.assert_allclose(pm.apply_gain_map(sdr, gain(1.0), display_boost=1.0), sdr, rtol=1e-5)


def test_gamma_is_undone_before_the_boost():
    sdr = np.full((2, 2, 3), 0.5, np.float32)
    gain = pm.GainMap(image=np.full((2, 2), 0.25, np.float32), gain_max=(2, 2, 2),
                      gamma=(0.5, 0.5, 0.5), capacity_max=2.0,
                      offset_sdr=(0, 0, 0), offset_hdr=(0, 0, 0))
    # 0.25 ** (1 / 0.5) = 0.0625 of the way to two stops.
    np.testing.assert_allclose(pm.apply_gain_map(sdr, gain), 0.5 * 2 ** (2 * 0.0625), rtol=1e-5)


def test_load_hdr_end_to_end_exceeds_sdr_white(tmp_path):
    path = _ultra_hdr(tmp_path, 'hdrgm:GainMapMax="2" hdrgm:HDRCapacityMax="2"', map_grey=255)
    hdr = pm.load_hdr(path)
    assert hdr is not None and hdr.dtype == np.float32
    assert hdr.max() > 1.5, "a two-stop gain map produced no headroom"
    assert pm.load_hdr(path, display_boost=1.0).max() <= 1.01


def test_gain_map_is_resized_to_the_image():
    sdr = np.full((40, 60, 3), 0.2, np.float32)
    gain = pm.GainMap(image=np.ones((10, 15), np.float32), gain_max=(1, 1, 1), capacity_max=1.0)
    assert pm.apply_gain_map(sdr, gain).shape == (40, 60, 3)


# --------------------------------------------------------------------- #
# Deep HEIC
# --------------------------------------------------------------------- #

def test_ten_bit_heic_keeps_its_precision(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    ramp = np.linspace(0, 65535, 512, dtype=np.uint16)
    data = np.repeat(np.repeat(ramp[None, :, None], 64, axis=0), 3, axis=2)
    path = tmp_path / "deep.heic"
    try:
        heif = pillow_heif.from_bytes(mode="RGB;16", size=(512, 64), data=data.tobytes())
        heif.save(str(path), quality=-1, bit_depth=10)
    except Exception as exc:
        pytest.skip(f"this build cannot encode 10-bit HEIC: {exc}")

    loaded = pm.load_heif_float(path)
    assert loaded is not None
    array, depth = loaded
    assert depth == 10
    assert array.dtype == np.float32 and array.shape == (64, 512, 3)
    row = array[32, :, 1]
    assert len(np.unique(np.round(row * 1023))) > 300, "the ramp collapsed to 8-bit steps"
    assert pm.inspect(path).bit_depth == 10


# --------------------------------------------------------------------- #
# ProRAW detection and metadata
# --------------------------------------------------------------------- #

def test_gain_table_detection_never_raises(tmp_path):
    junk = tmp_path / "junk.dng"
    junk.write_bytes(b"not a tiff")
    assert pm.has_profile_gain_table(junk) is False


@pytest.mark.media
def test_ordinary_dngs_report_no_gain_table():
    present = [s for s in SAMPLES if s.name.endswith(".dng") and s.present]
    if not present:
        pytest.skip("no DNG samples downloaded")
    assert not any(pm.has_profile_gain_table(s.path) for s in present)


def test_rich_metadata_is_read(tmp_path):
    exif = Image.Exif()
    exif[0x010F] = "TestCam"
    exif[0x0110] = "TestCam Z9"
    sub = exif.get_ifd(0x8769)
    sub[0x920A] = 50.0
    sub[0xA405] = 75
    sub[0xA431] = "BODY123"
    sub[0xA435] = "LENS456"
    sub[0x9003] = "2025:06:01 09:30:00"
    sub[0x9011] = "+02:00"
    path = tmp_path / "meta.jpg"
    Image.fromarray(cv2.cvtColor(make_scene(1), cv2.COLOR_BGR2RGB)).save(path, "JPEG", exif=exif)

    meta = UniversalLoader().read_metadata(path)
    assert meta.camera_label == "Testcam Z9"
    assert meta.focal_label == "50mm" and meta.focal_length_35mm == 75
    assert meta.body_serial == "BODY123" and meta.lens_serial == "LENS456"
    assert meta.captured_at == datetime(2025, 6, 1, 9, 30)
    assert meta.utc_offset == "+02:00"
    assert meta.as_dict()["focal_length"] == "50mm"


@pytest.mark.media
def test_raw_metadata_includes_the_capture_time():
    sample = next((s for s in SAMPLES if s.name == "canon_eos_r6.cr3" and s.present), None)
    if sample is None:
        pytest.skip("sample not downloaded")
    meta = UniversalLoader().read_metadata(sample.path)
    assert meta.captured_at is not None and "Canon" in meta.camera_label
