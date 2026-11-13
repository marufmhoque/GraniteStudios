"""The keeper score and the face-aware cull."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ai_tools import culling
from src.ai_tools.culling import CullResult, SmartCull, assign_scores, composition_score
from tests.conftest import make_scene, write_jpeg
from tests.fetch_media import FACES, face


def _frame(name="a.jpg", sharpness=100.0, **fields) -> CullResult:
    frame = CullResult(path=Path(name), sharpness=sharpness)
    for key, value in fields.items():
        setattr(frame, key, value)
    return frame


# --------------------------------------------------------------------- #
# Score arithmetic
# --------------------------------------------------------------------- #

def test_weights_are_the_specified_ones_and_sum_to_one():
    weights = (culling.WEIGHT_SHARPNESS, culling.WEIGHT_EYES, culling.WEIGHT_SMILE, culling.WEIGHT_COMPOSITION)
    assert weights == (0.40, 0.35, 0.15, 0.10)
    assert sum(weights) == pytest.approx(1.0)


def test_score_is_the_weighted_sum_of_normalised_terms():
    best = _frame("best.jpg", face_sharpness=200.0, min_eye_openness=1.0, smile=1.0, composition=1.0)
    other = _frame("other.jpg", face_sharpness=100.0, min_eye_openness=0.5, smile=0.2, composition=0.8)
    assign_scores([best, other])
    # Sharpness is scaled by the batch's 95th percentile (195 here).
    top = float(np.percentile([200.0, 100.0], 95))
    assert best.score == pytest.approx(1.0)
    assert other.score == pytest.approx(0.40 * (100.0 / top) + 0.35 * 0.5 + 0.15 * 0.2 + 0.10 * 0.8)


def test_every_score_stays_between_zero_and_one():
    frames = [
        _frame("a.jpg", 5000.0, face_sharpness=9000.0, min_eye_openness=3.0, smile=2.0, composition=1.0),
        _frame("b.jpg", 0.0, face_sharpness=0.0, min_eye_openness=-1.0, smile=-0.5, composition=0.0),
        _frame("c.jpg", 300.0),
    ]
    assign_scores(frames)
    assert all(0.0 <= f.score <= 1.0 for f in frames)


def test_a_blink_costs_more_than_a_little_softness():
    """The point of weighting eyes at 0.35: a slightly softer frame with
    eyes open should beat a crisper one mid-blink."""
    crisp_blink = _frame("blink.jpg", face_sharpness=200.0, min_eye_openness=0.0, smile=0.5, composition=1.0)
    softer_open = _frame("open.jpg", face_sharpness=150.0, min_eye_openness=1.0, smile=0.5, composition=1.0)
    assign_scores([crisp_blink, softer_open])
    assert softer_open.score > crisp_blink.score


def test_a_missing_term_is_left_out_not_scored_as_zero():
    with_smile = _frame("a.jpg", face_sharpness=100.0, min_eye_openness=1.0, smile=1.0, composition=1.0)
    no_smile = _frame("b.jpg", face_sharpness=100.0, min_eye_openness=1.0, smile=None, composition=1.0)
    assign_scores([with_smile, no_smile])
    assert no_smile.score == pytest.approx(with_smile.score)


def test_frames_without_faces_fall_back_to_whole_frame_sharpness():
    frames = [_frame("a.jpg", 400.0), _frame("b.jpg", 200.0), _frame("c.jpg", 0.0)]
    assign_scores(frames)
    assert frames[0].score > frames[1].score > frames[2].score == 0.0


def test_empty_batch_is_fine():
    assign_scores([])


# --------------------------------------------------------------------- #
# Composition heuristic
# --------------------------------------------------------------------- #

def test_composition_full_marks_for_a_face_with_room():
    assert composition_score((400, 300, 200, 240), (1200, 800)) == 1.0


def test_composition_penalises_a_face_cut_by_the_edge():
    inside = composition_score((400, 300, 200, 240), (1200, 800))
    cut = composition_score((-100, 300, 200, 240), (1200, 800))
    assert cut < inside - 0.3


def test_composition_penalises_no_headroom():
    assert composition_score((400, 5, 200, 240), (1200, 800)) < composition_score((400, 200, 200, 240), (1200, 800))


def test_composition_never_leaves_the_unit_interval():
    for box in [(-500, -500, 100, 100), (0, 0, 5000, 5000), (10, 10, 0, 0)]:
        assert 0.0 <= composition_score(box, (1200, 800)) <= 1.0


# --------------------------------------------------------------------- #
# Region sharpness
# --------------------------------------------------------------------- #

def test_region_sharpness_looks_only_inside_the_box():
    """A frame that is sharp everywhere except the face must score low on
    face sharpness, however crisp the background."""
    scene = cv2.cvtColor(make_scene(3, (900, 600)), cv2.COLOR_BGR2GRAY)
    box = (350, 200, 200, 200)
    soft_face = scene.copy()
    x, y, w, h = box
    soft_face[y:y + h, x:x + w] = cv2.GaussianBlur(scene[y:y + h, x:x + w], (0, 0), 6)
    assert culling._region_sharpness(soft_face, box) < 0.2 * culling._region_sharpness(scene, box)
    whole = lambda g: float(cv2.Laplacian(g, cv2.CV_64F).var())  # noqa: E731
    assert whole(soft_face) > 0.7 * whole(scene), "the background should still dominate the whole-frame score"


def test_region_sharpness_handles_boxes_off_the_frame():
    gray = np.zeros((100, 100), np.uint8)
    assert culling._region_sharpness(gray, (-50, -50, 20, 20)) == 0.0
    assert culling._region_sharpness(gray, (95, 95, 50, 50)) == 0.0


# --------------------------------------------------------------------- #
# End to end, with the real models
# --------------------------------------------------------------------- #

@pytest.fixture
def portraits(tmp_path):
    from src.ai_tools.face_analysis import FaceAnalyzer
    from src.ai_tools.face_mesh import FaceMesh

    if not all(f.present for f in FACES) or not FaceMesh.is_cached():
        pytest.skip("reference faces or face-mesh model not present")
    if not FaceAnalyzer(download=False).available:
        pytest.skip("face detector not cached")
    folder = tmp_path / "Portraits"
    folder.mkdir()
    image = cv2.resize(cv2.imread(str(face("pose.png").path)), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    write_jpeg(folder / "sharp.jpg", image, quality=95)
    write_jpeg(folder / "soft.jpg", cv2.GaussianBlur(image, (0, 0), 4), quality=95)
    write_jpeg(folder / "no_face.jpg", make_scene(1, (900, 700)))
    return folder


@pytest.mark.model
@pytest.mark.media
def test_cull_measures_faces_with_the_mesh(portraits):
    frames = {f.path.name: f for f in SmartCull().score_frames(sorted(portraits.iterdir()), check_eyes=True)}
    sharp, soft, empty = frames["sharp.jpg"], frames["soft.jpg"], frames["no_face.jpg"]

    assert sharp.face_count == 1 and sharp.eye_method == "mesh"
    assert sharp.min_ear is not None and not sharp.eyes_closed
    assert sharp.smile is not None and sharp.smile > 0.5
    assert sharp.frame_size is not None and sharp.faces[0].is_main
    assert sharp.face_sharpness > soft.face_sharpness
    assert sharp.score > soft.score

    assert empty.face_count == 0 and empty.eye_method == "none"
    assert empty.face_sharpness is None and 0.0 <= empty.score <= 1.0


@pytest.mark.model
@pytest.mark.media
def test_cull_falls_back_to_the_heuristic_without_the_mesh(portraits):
    frames = SmartCull().score_frames([portraits / "sharp.jpg"], check_eyes=True, use_mesh=False)
    assert frames[0].eye_method == "heuristic"
    assert frames[0].min_ear is None and frames[0].min_eye_openness is not None


@pytest.mark.model
@pytest.mark.media
def test_face_boxes_refer_to_the_recorded_frame_size(portraits):
    frame = SmartCull().score_frames([portraits / "sharp.jpg"], check_eyes=True)[0]
    width, height = frame.frame_size
    x, y, w, h = frame.faces[0].box
    assert 0 <= x and x + w <= width and 0 <= y and y + h <= height

    # The face was found on an 800 px copy. Its box must come back in the
    # full frame's pixels, matching a detection run at full size.
    from src.ai_tools.face_analysis import FaceAnalyzer

    assert max(width, height) > culling.DETECT_MAX_SIDE, "the test image is too small to prove anything"
    full = FaceAnalyzer(download=False).detect(cv2.imread(str(portraits / "sharp.jpg")), measure_eyes=False)[0]
    assert w == pytest.approx(full.w, rel=0.15) and x == pytest.approx(full.x, abs=0.15 * full.w)
