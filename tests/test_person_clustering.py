"""Person clustering: the arithmetic exactly, then the recogniser's
self-consistency on reference faces."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ai_tools import model_store
from src.ai_tools import person_clustering as pc
from src.ai_tools.culling import CullResult, FaceObservation
from tests.fetch_media import FACES, face


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _blob(centre, count, spread, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([_unit(centre + rng.normal(0, spread, len(centre))) for _ in range(count)])


# --------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------- #

def test_cosine_distance_basics():
    d = pc.cosine_distances(np.array([[1, 0], [2, 0], [0, 3], [-1, 0]], np.float32))
    assert d[0, 1] == pytest.approx(0.0, abs=1e-6)     # same direction, any length
    assert d[0, 2] == pytest.approx(1.0, abs=1e-6)     # perpendicular
    assert d[0, 3] == pytest.approx(2.0, abs=1e-6)     # opposite
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0.0, atol=1e-6)
    assert pc.cosine_distances(np.zeros((0, 4), np.float32)).shape == (0, 0)


def test_three_well_separated_groups_are_recovered():
    axes = np.eye(16, dtype=np.float32)
    points = np.vstack([_blob(axes[i], 6, 0.03, i) for i in range(3)])
    labels = pc.cluster_embeddings(points, eps=0.2, min_samples=2)
    assert set(labels) == {0, 1, 2}
    for group in range(3):
        assert len(set(labels[group * 6:(group + 1) * 6])) == 1


def test_a_face_seen_once_is_noise_not_a_person():
    axes = np.eye(8, dtype=np.float32)
    points = np.vstack([_blob(axes[0], 5, 0.02, 1), axes[3][None]])
    labels = pc.cluster_embeddings(points, eps=0.2, min_samples=2)
    assert labels[-1] == pc.NOISE and set(labels[:5]) == {0}


def test_border_points_join_but_do_not_bridge_clusters():
    # Two tight groups on a line with one point midway. The midpoint is
    # within reach of one member of each group but has only three
    # neighbours itself, so with min_samples 4 it is a border point: it
    # joins a cluster and must not merge the two.
    line = lambda xs: np.array([[x, 0.0] for x in xs])  # noqa: E731
    points = np.vstack([line([0.0, 0.1, 0.2]), line([1.0]), line([1.8, 1.9, 2.0])])
    distances = np.abs(points[:, None, 0] - points[None, :, 0])
    labels = pc.dbscan(distances, eps=0.85, min_samples=4)
    assert labels[3] != pc.NOISE
    assert labels[0] == labels[1] == labels[2]
    assert labels[4] == labels[5] == labels[6]
    assert labels[0] != labels[4], "a border point bridged two clusters"


def test_labels_are_deterministic():
    points = np.vstack([_blob(np.eye(8, dtype=np.float32)[i], 4, 0.03, i) for i in range(3)])
    first = pc.cluster_embeddings(points, 0.2, 2)
    assert np.array_equal(first, pc.cluster_embeddings(points, 0.2, 2))
    assert first[0] == 0, "clusters should be numbered in the order they are met"


def test_empty_input():
    assert pc.dbscan(np.zeros((0, 0)), 0.5, 2).shape == (0,)


# --------------------------------------------------------------------- #
# Frames to people
# --------------------------------------------------------------------- #

def _frame(name: str, *embeddings) -> CullResult:
    frame = CullResult(path=Path(name), sharpness=1.0)
    frame.faces = [FaceObservation(box=(10 * i, 0, 50, 60), confidence=0.9, embedding=e)
                   for i, e in enumerate(embeddings)]
    return frame


def test_frames_are_grouped_by_who_is_in_them():
    a, b = np.eye(8, dtype=np.float32)[0], np.eye(8, dtype=np.float32)[4]
    near = lambda v, seed: _blob(v, 1, 0.02, seed)[0]  # noqa: E731
    frames = [
        _frame("1.jpg", near(a, 1)),
        _frame("2.jpg", near(a, 2), near(b, 3)),
        _frame("3.jpg", near(b, 4)),
        _frame("4.jpg", None),                      # a face with no embedding
        _frame("5.jpg"),                            # no faces
    ]
    index, matrix = pc.cluster_frames(frames, eps=0.2, min_samples=2)
    assert matrix.shape == (4, 8)
    assert len(index.people) == 2
    assert index.paths_of(0) == ["1.jpg", "2.jpg"] and index.paths_of(1) == ["2.jpg", "3.jpg"]
    assert frames[1].person_ids == [0, 1]
    assert frames[3].person_ids == [] and frames[3].faces[0].person_id is None
    assert index.label(0) == "Person 1"


def test_names_are_the_users_and_survive_a_save(tmp_path):
    a = np.eye(8, dtype=np.float32)[0]
    frames = [_frame("1.jpg", a), _frame("2.jpg", a)]
    index, matrix = pc.cluster_frames(frames, eps=0.2, min_samples=2)
    index.rename(0, "  Bride ")
    edit = tmp_path / "Shoot-edit"
    index.save(edit, matrix)
    assert (edit / pc.INDEX_FILENAME).exists() and (edit / pc.EMBEDDINGS_FILENAME).exists()

    loaded = pc.PeopleIndex.load(edit)
    assert loaded.label(0) == "Bride"
    assert loaded.paths_of(0) == ["1.jpg", "2.jpg"]
    assert loaded.people[0][0].box == (0, 0, 50, 60)
    loaded.rename(0, "")
    assert loaded.label(0) == "Person 1"


def test_forget_removes_everything_stored_about_faces(tmp_path):
    a = np.eye(8, dtype=np.float32)[0]
    index, matrix = pc.cluster_frames([_frame("1.jpg", a), _frame("2.jpg", a)], 0.2, 2)
    index.save(tmp_path, matrix)
    (tmp_path / "IMG_1_graded.jpg").write_bytes(b"an export")
    assert pc.PeopleIndex.forget(tmp_path) == 2
    assert [p.name for p in tmp_path.iterdir()] == ["IMG_1_graded.jpg"]
    assert pc.PeopleIndex.forget(tmp_path) == 0


def test_a_damaged_index_loads_as_empty(tmp_path):
    (tmp_path / pc.INDEX_FILENAME).write_text("{ not json", encoding="utf-8")
    assert pc.PeopleIndex.load(tmp_path).people == {}
    assert pc.PeopleIndex.load(tmp_path / "missing").people == {}


def test_without_the_model_embedding_is_off_not_broken(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "MODEL_DIR", tmp_path)
    embedder = pc.FaceEmbedder(download=False)
    assert not embedder.available
    assert embedder.embed(np.zeros((50, 50, 3), np.uint8), np.zeros(15, np.float32)) is None


# --------------------------------------------------------------------- #
# The recogniser, on reference faces
# --------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def recogniser():
    from src.ai_tools.face_analysis import FaceAnalyzer

    if not (pc.FaceEmbedder.is_cached() and all(f.present for f in FACES)):
        pytest.skip("recognition model or reference faces not present")
    detector, embedder = FaceAnalyzer(download=False), pc.FaceEmbedder(download=False)
    if not (detector.available and embedder.available):
        pytest.skip("face models not cached")

    def embed(image):
        found = detector.detect(image, measure_eyes=False)
        return embedder.embed(image, found[0].detector_row()) if found else None

    return embed


def _variants(image: np.ndarray) -> list[np.ndarray]:
    h, w = image.shape[:2]
    turn = cv2.getRotationMatrix2D((w / 2, h / 2), 6.0, 1.0)
    return [
        cv2.convertScaleAbs(image, alpha=1.15, beta=12),
        cv2.convertScaleAbs(image, alpha=0.85, beta=-10),
        cv2.GaussianBlur(image, (0, 0), 1.2),
        cv2.warpAffine(image, turn, (w, h), borderMode=cv2.BORDER_REPLICATE),
    ]


@pytest.mark.model
@pytest.mark.media
def test_the_same_face_stays_closer_to_itself_than_to_others(recogniser):
    """Self-consistency only. The variants are edits of one photograph, not
    different photographs of one person, so this is not a recognition
    accuracy figure."""
    images = {
        f.name: cv2.resize(cv2.imread(str(face(f.name).path)), None, fx=3, fy=3,
                           interpolation=cv2.INTER_CUBIC)
        for f in FACES
    }
    base = {name: recogniser(image) for name, image in images.items()}
    assert all(v is not None and v.shape == (pc.EMBEDDING_SIZE,) for v in base.values())
    assert all(np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4) for v in base.values())

    within, between = [], []
    for name, image in images.items():
        for variant in _variants(image):
            vector = recogniser(variant)
            if vector is not None:
                within.append(float(base[name] @ vector))
        between += [float(base[name] @ base[other]) for other in images if other != name]

    assert min(within) > max(between), (
        f"weakest same-face match {min(within):.2f} did not beat "
        f"strongest different-face match {max(between):.2f}"
    )
    assert min(within) > pc.SAME_PERSON_SIMILARITY
    assert max(between) < 1.0 - pc.DEFAULT_EPS, "two different people fall inside one cluster radius"


@pytest.mark.model
@pytest.mark.media
def test_reference_faces_cluster_into_four_people(recogniser):
    vectors, owners = [], []
    for sample in FACES:
        image = cv2.resize(cv2.imread(str(sample.path)), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        for picture in [image, *_variants(image)]:
            vector = recogniser(picture)
            if vector is not None:
                vectors.append(vector)
                owners.append(sample.name)
    labels = pc.cluster_embeddings(np.vstack(vectors))
    groups = {name: {int(l) for l, o in zip(labels, owners) if o == name} for name in set(owners)}
    assert all(len(g) == 1 and pc.NOISE not in g for g in groups.values()), groups
    assert len({next(iter(g)) for g in groups.values()}) == len(FACES)
