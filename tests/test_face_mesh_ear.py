"""Eye aspect ratio: the arithmetic exactly, then the model on real faces.

Two different kinds of claim are tested here and should not be confused.

* The geometry functions are checked on constructed points with known
  answers. Those tests are exact.
* The model is checked on four reference faces, each also with its eyes
  painted over. That shows the measurement responds to closed eyes and pins
  the numbers it currently produces. The "closed" images are synthetic, so
  this is not validation on real blinks.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.ai_tools import face_mesh as fm
from src.ai_tools import model_store
from tests.fetch_media import FACES, face

FACE_SCALE = 3   # the reference faces are small; work on an enlarged copy


# --------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------- #

def _eye(width: float, height: float) -> np.ndarray:
    """Six points p1..p6 for an eye of the given width and lid opening."""
    return np.array([
        [0, 0], [width / 3, -height / 2], [2 * width / 3, -height / 2],
        [width, 0], [2 * width / 3, height / 2], [width / 3, height / 2],
    ], dtype=np.float64)


@pytest.mark.parametrize("width,height,expected", [
    (30.0, 9.0, 0.30),     # open
    (30.0, 4.5, 0.15),     # squinting
    (30.0, 0.0, 0.00),     # closed
    (60.0, 18.0, 0.30),    # the same eye, twice as large
])
def test_ear_matches_the_formula(width, height, expected):
    assert fm.eye_aspect_ratio(_eye(width, height)) == pytest.approx(expected, abs=1e-9)


def test_ear_ignores_position_scale_and_rotation():
    eye = _eye(30.0, 9.0)
    theta = np.radians(37.0)
    turn = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    moved = (eye * 4.2) @ turn.T + [512.0, -77.0]
    assert fm.eye_aspect_ratio(moved) == pytest.approx(fm.eye_aspect_ratio(eye), abs=1e-9)


def test_ear_of_a_degenerate_eye_is_zero_not_an_error():
    assert fm.eye_aspect_ratio(np.zeros((6, 2))) == 0.0


def _mesh_with(points: dict[int, tuple[float, float]]) -> np.ndarray:
    mesh = np.zeros((fm.MESH_POINTS, 3), np.float32)
    for index, (x, y) in points.items():
        mesh[index, :2] = (x, y)
    return mesh


def test_ear_pair_reads_the_right_points():
    points = {}
    for indices, opening in ((fm.RIGHT_EYE, 9.0), (fm.LEFT_EYE, 3.0)):
        for index, xy in zip(indices, _eye(30.0, opening)):
            points[index] = tuple(xy)
    right, left = fm.ear_pair(_mesh_with(points))
    assert right == pytest.approx(0.30, abs=1e-6) and left == pytest.approx(0.10, abs=1e-6)


def test_mouth_ratio():
    closed = {fm.MOUTH_CORNERS[0]: (0, 0), fm.MOUTH_CORNERS[1]: (40, 0)}
    for upper, lower in zip(fm.MOUTH_UPPER, fm.MOUTH_LOWER):
        closed[upper], closed[lower] = (20, 0), (20, 0)
    assert fm.mouth_aspect_ratio(_mesh_with(closed)) == pytest.approx(0.0)
    opened = dict(closed)
    for upper, lower in zip(fm.MOUTH_UPPER, fm.MOUTH_LOWER):
        opened[upper], opened[lower] = (20, -10), (20, 10)
    assert fm.mouth_aspect_ratio(_mesh_with(opened)) == pytest.approx(0.5)


@pytest.mark.parametrize("angle", [-30.0, -5.0, 0.0, 12.0, 40.0])
def test_head_roll_from_the_eye_corners(angle):
    theta = np.radians(angle)
    mesh = _mesh_with({
        fm.RIGHT_EYE_OUTER: (0.0, 0.0),
        fm.LEFT_EYE_OUTER: (100 * np.cos(theta), 100 * np.sin(theta)),
    })
    assert fm.head_roll(mesh) == pytest.approx(angle, abs=1e-4)


def test_thresholds_leave_a_band_between_closed_and_open():
    assert fm.EAR_CLOSED_BELOW < fm.EAR_OPEN_ABOVE
    result = fm.FaceMeshResult(np.zeros((fm.MESH_POINTS, 3), np.float32), 0.17, 0.10, 0.0, 0.0)
    assert result.ear == 0.17, "the frame is judged on its more open eye"
    assert not result.eyes_closed and not result.eyes_open
    assert 0.0 < result.eye_openness < 1.0


def test_a_wink_is_not_a_blink():
    result = fm.FaceMeshResult(np.zeros((fm.MESH_POINTS, 3), np.float32), 0.28, 0.03, 0.0, 0.0)
    assert not result.eyes_closed


def test_the_expression_point_list_is_what_the_network_expects():
    assert len(fm._EXPRESSION_POINTS) == 146
    assert len(set(fm._EXPRESSION_POINTS)) == 146
    assert max(fm._EXPRESSION_POINTS) == fm.MESH_POINTS - 1
    assert len(fm.EXPRESSION_NAMES) == 52


def test_without_the_model_the_feature_is_off_not_broken(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "MODEL_DIR", tmp_path)
    mesh = fm.FaceMesh(download=False)
    assert not mesh.available and mesh.load_error
    assert mesh.analyze_face(np.zeros((100, 100, 3), np.uint8), object()) is None


def test_a_damaged_archive_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(model_store, "ensure_model", lambda spec, download=True, timeout=300: tmp_path / "bad.task")
    (tmp_path / "bad.task").write_bytes(b"this is not a zip file")
    mesh = fm.FaceMesh(download=False)
    assert not mesh.available and "unreadable" in mesh.load_error


# --------------------------------------------------------------------- #
# The model, on reference faces
# --------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def engines():
    from src.ai_tools.face_analysis import FaceAnalyzer

    if not (fm.FaceMesh.is_cached() and all(f.present for f in FACES)):
        pytest.skip("face-mesh model or reference faces not present")
    detector, mesh = FaceAnalyzer(download=False), fm.FaceMesh(download=False)
    if not (detector.available and mesh.available):
        pytest.skip("face models not cached")
    return detector, mesh


def _load(name: str) -> np.ndarray:
    image = cv2.imread(str(face(name).path))
    return cv2.resize(image, None, fx=FACE_SCALE, fy=FACE_SCALE, interpolation=cv2.INTER_CUBIC)


def _paint_eyes_shut(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Cover both eyes with nearby skin and draw a lash line.

    A stand-in for a blink, used because no labelled photographs of real
    closed eyes were available.
    """
    out = image.copy()
    for ring in (fm.RIGHT_EYE_RING, fm.LEFT_EYE_RING):
        outline = np.round(landmarks[list(ring), :2]).astype(np.int32)
        x, y, w, h = cv2.boundingRect(outline)
        below = image[y + h + max(2, h // 2): y + h + max(4, h), x: x + w]
        skin = below.reshape(-1, 3).mean(axis=0) if below.size else np.array([150.0, 150.0, 180.0])
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [outline], 255)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=max(1, h // 3))
        soft = cv2.GaussianBlur(mask, (0, 0), max(1.0, h / 3))[:, :, None] / 255.0
        out = (out * (1 - soft) + skin * soft).astype(np.uint8)
        lower = np.round(landmarks[list(ring[:9]), :2]).astype(np.int32)
        cv2.polylines(out, [lower], False, tuple(int(c * 0.45) for c in skin), max(1, h // 4), cv2.LINE_AA)
    return out


def _measure(engines, image):
    detector, mesh = engines
    faces = detector.detect(image)
    assert faces, "no face detected"
    return mesh.analyze_face(image, faces[0])


@pytest.mark.model
@pytest.mark.media
@pytest.mark.parametrize("name", [f.name for f in FACES])
def test_open_eyes_read_as_open(engines, name):
    result = _measure(engines, _load(name))
    assert result is not None
    assert not result.eyes_closed, f"open eyes flagged as closed (EAR {result.ear:.3f})"


@pytest.mark.model
@pytest.mark.media
@pytest.mark.parametrize("name", ["grace_hopper_227.png", "basketball1.png", "pose.png"])
def test_painted_shut_eyes_read_as_closed(engines, name):
    image = _load(name)
    shut = _paint_eyes_shut(image, _measure(engines, image).landmarks)
    result = _measure(engines, shut)
    assert result.eyes_closed, f"painted-shut eyes not flagged (EAR {result.ear:.3f})"


@pytest.mark.model
@pytest.mark.media
@pytest.mark.parametrize("name", [f.name for f in FACES])
def test_closing_the_eyes_always_lowers_ear(engines, name):
    image = _load(name)
    opened = _measure(engines, image)
    closed = _measure(engines, _paint_eyes_shut(image, opened.landmarks))
    assert closed.ear < opened.ear


@pytest.mark.model
@pytest.mark.media
def test_known_limit_a_small_downward_gaze_is_ambiguous(engines):
    """Recorded so it is not rediscovered: a 30 px face looking down reads
    0.155 open and about 0.14 painted shut. Both sit beside the threshold,
    so this frame is not reliably classified either way."""
    image = _load("messi5.jpg")
    opened = _measure(engines, image)
    closed = _measure(engines, _paint_eyes_shut(image, opened.landmarks))
    assert opened.ear < fm.EAR_OPEN_ABOVE, "this face no longer reads low; update the note"
    assert abs(closed.ear - fm.EAR_CLOSED_BELOW) < 0.04


@pytest.mark.model
@pytest.mark.media
def test_smile_is_told_from_an_open_mouth(engines):
    smiling = _measure(engines, _load("pose.png"))
    neutral = _measure(engines, _load("grace_hopper_227.png"))
    shouting = _measure(engines, _load("messi5.jpg"))
    assert smiling.smile_source == "expression"
    assert smiling.is_smiling and smiling.smile > 0.8
    assert not neutral.is_smiling and neutral.smile < 0.2
    assert not shouting.is_smiling, "an open mouth mid-action was read as a smile"


@pytest.mark.model
@pytest.mark.media
@pytest.mark.parametrize("applied", [-25.0, -15.0, -8.0, 8.0, 15.0, 25.0])
def test_head_roll_tracks_a_known_rotation(engines, applied):
    image = _load("pose.png")
    upright = _measure(engines, image).roll_deg
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), applied, 1.0)
    turned = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    result = _measure(engines, turned)
    # OpenCV rotates counter-clockwise for a positive angle; roll is
    # reported clockwise-positive, hence the sign.
    assert result.roll_deg - upright == pytest.approx(-applied, abs=2.0)
    assert not result.eyes_closed, "turning the head changed the blink verdict"


@pytest.mark.model
@pytest.mark.media
def test_mesh_lands_on_the_face(engines):
    detector, _ = engines
    image = _load("pose.png")
    found = detector.detect(image)[0]
    result = _measure(engines, image)
    eyes = result.landmarks[[fm.RIGHT_EYE_OUTER, fm.LEFT_EYE_OUTER], :2]
    assert found.x < eyes[:, 0].min() and eyes[:, 0].max() < found.x + found.w
    assert found.y < eyes[:, 1].min() and eyes[:, 1].max() < found.y + found.h
    x, y, w, h = result.eye_box()
    assert w > 0 and h > 0 and found.x - found.w < x < found.x + found.w
