"""Grouping photographs by who is in them, entirely on this machine.

Each main face gets a 128-number embedding from SFace, a recognition
network. Faces whose embeddings point the same way are the same person, so
clustering the embeddings groups a shoot by subject. Nothing is uploaded and
no names are guessed: a cluster is "Person 3" until the photographer calls
it "Bride".

Privacy
-------
An embedding is biometric data. It is derived only from the photographs in
the folder being worked on, kept only in that folder's edit subfolder, and
removed by :meth:`PeopleIndex.forget`. There is no gallery of known faces
and nothing carries over from one shoot to the next.

How far this has been checked
-----------------------------
The clustering arithmetic is tested exactly, on constructed points. The
recognition model itself was checked only for self-consistency: the same
face under small changes (brightness, a few degrees of rotation, mild blur)
stays far closer to itself than to three other people. That shows the
pipeline is wired correctly. It is not a measurement of recognition
accuracy across a real event, which needs many photographs of the same
people and was not available.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.ai_tools import model_store
from src.ai_tools.model_store import ModelSpec

logger = logging.getLogger(__name__)

SFACE_SPEC = ModelSpec(
    filename="face_recognition_sface_2021dec.onnx",
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    size=38696353,
    zoo_dir="face_recognition_sface",
    label="the face-recognition model",
)

EMBEDDING_SIZE = 128

# SFace's published operating point treats cosine similarity >= 0.363 as the
# same person. Density clustering chains neighbours together, so one loose
# link can merge two people; the radius used here is tighter than that.
SAME_PERSON_SIMILARITY = 0.363
DEFAULT_EPS = 0.45            # cosine distance, i.e. similarity >= 0.55
DEFAULT_MIN_SAMPLES = 2

INDEX_FILENAME = "people.granite.json"
EMBEDDINGS_FILENAME = "people.granite.npz"

NOISE = -1


# --------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------- #

class FaceEmbedder:
    """Turns an aligned face into a unit-length identity vector.

    Safe to call from several threads: each gets its own network.
    """

    def __init__(self, download: bool = True) -> None:
        self._path = model_store.ensure_model(SFACE_SPEC, download=download)
        self._local = threading.local()
        self._load_error = None if self._path else "face-recognition model unavailable"

    @property
    def available(self) -> bool:
        return self._path is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @staticmethod
    def is_cached() -> bool:
        return model_store.is_cached(SFACE_SPEC)

    @staticmethod
    def download_mb() -> float:
        return model_store.total_download_mb([SFACE_SPEC])

    def _net(self):
        net = getattr(self._local, "net", None)
        if net is None and self._path is not None:
            try:
                net = self._local.net = cv2.FaceRecognizerSF.create(str(self._path), "")
            except Exception as exc:
                self._load_error = f"face-recognition model failed to load: {exc}"
                logger.warning(self._load_error)
                self._path = None
        return net

    def embed(self, bgr: np.ndarray, detector_row: np.ndarray | None) -> np.ndarray | None:
        """Embedding for one face, or None if it could not be computed.

        ``detector_row`` is ``Face.detector_row()``: the box and the five
        landmarks the recogniser uses to align the face.
        """
        if detector_row is None:
            return None
        net = self._net()
        if net is None:
            return None
        try:
            aligned = net.alignCrop(bgr, np.asarray(detector_row, dtype=np.float32).reshape(1, -1))
            vector = np.asarray(net.feature(aligned), dtype=np.float32).reshape(-1)
        except Exception as exc:
            logger.debug("Face embedding failed: %s", exc)
            return None
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else None


# --------------------------------------------------------------------- #
# Clustering: pure functions
# --------------------------------------------------------------------- #

def cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance (0 identical, 1 unrelated, 2 opposite)."""
    e = np.asarray(embeddings, dtype=np.float32)
    if e.ndim != 2 or e.shape[0] == 0:
        return np.zeros((0, 0), np.float32)
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    unit = e / np.where(norms > 0, norms, 1.0)
    return np.clip(1.0 - unit @ unit.T, 0.0, 2.0)


def dbscan(distances: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Density clustering on a precomputed distance matrix.

    Returns one label per row; ``NOISE`` (-1) marks points in no cluster.
    A point with at least ``min_samples`` neighbours within ``eps`` (itself
    included) is a core point; clusters are the connected groups of core
    points plus the border points they reach. Labels are numbered from 0 in
    the order clusters are first met, so the result is deterministic.
    """
    n = distances.shape[0]
    labels = np.full(n, NOISE, dtype=np.int64)
    if n == 0:
        return labels
    near = distances <= eps
    core = near.sum(axis=1) >= min_samples
    cluster = 0
    for seed in range(n):
        if labels[seed] != NOISE or not core[seed]:
            continue
        labels[seed] = cluster
        frontier = [seed]
        while frontier:
            point = frontier.pop()
            if not core[point]:
                continue                      # border points do not spread
            for neighbour in np.flatnonzero(near[point]):
                if labels[neighbour] == NOISE:
                    labels[neighbour] = cluster
                    frontier.append(int(neighbour))
        cluster += 1
    return labels


def cluster_embeddings(
    embeddings: np.ndarray, eps: float = DEFAULT_EPS, min_samples: int = DEFAULT_MIN_SAMPLES
) -> np.ndarray:
    """Cluster identity embeddings; returns a label per row."""
    return dbscan(cosine_distances(embeddings), eps, min_samples)


# --------------------------------------------------------------------- #
# A shoot's people
# --------------------------------------------------------------------- #

@dataclass
class FaceRef:
    """One face in one photograph."""

    path: str
    face_index: int
    box: tuple[int, int, int, int]


@dataclass
class PeopleIndex:
    """Who appears where, for one folder."""

    people: dict[int, list[FaceRef]] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)
    unassigned: list[FaceRef] = field(default_factory=list)   # seen only once

    def label(self, person_id: int) -> str:
        return self.names.get(person_id) or f"Person {person_id + 1}"

    def rename(self, person_id: int, name: str) -> None:
        name = name.strip()
        if name:
            self.names[person_id] = name
        else:
            self.names.pop(person_id, None)

    def paths_of(self, person_id: int) -> list[str]:
        """Photographs this person appears in, in order, without repeats."""
        seen: dict[str, None] = {}
        for ref in self.people.get(person_id, []):
            seen.setdefault(ref.path)
        return list(seen)

    def by_size(self) -> list[int]:
        """Person ids, most-photographed first."""
        return sorted(self.people, key=lambda pid: (-len(self.paths_of(pid)), pid))

    # -- persistence: the edit folder only, never beside the originals ---- #

    def save(self, edit_folder: str | Path, embeddings: np.ndarray | None = None) -> Path:
        edit_folder = Path(edit_folder)
        edit_folder.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "names": {str(k): v for k, v in self.names.items()},
            "people": {
                str(pid): [[r.path, r.face_index, list(r.box)] for r in refs]
                for pid, refs in self.people.items()
            },
            "unassigned": [[r.path, r.face_index, list(r.box)] for r in self.unassigned],
        }
        target = edit_folder / INDEX_FILENAME
        target.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        if embeddings is not None:
            np.savez_compressed(edit_folder / EMBEDDINGS_FILENAME, embeddings=embeddings)
        return target

    @classmethod
    def load(cls, edit_folder: str | Path) -> "PeopleIndex":
        """Read a saved index. A missing or damaged file gives an empty one."""
        try:
            raw = json.loads((Path(edit_folder) / INDEX_FILENAME).read_text(encoding="utf-8"))
            ref = lambda item: FaceRef(str(item[0]), int(item[1]), tuple(int(v) for v in item[2]))  # noqa: E731
            return cls(
                people={int(k): [ref(i) for i in v] for k, v in raw.get("people", {}).items()},
                names={int(k): str(v) for k, v in raw.get("names", {}).items()},
                unassigned=[ref(i) for i in raw.get("unassigned", [])],
            )
        except Exception:
            return cls()

    @staticmethod
    def forget(edit_folder: str | Path) -> int:
        """Delete everything stored about faces for this folder.

        Returns how many files were removed. The photographs themselves are
        not touched; only this app's own derived files are.
        """
        removed = 0
        for name in (INDEX_FILENAME, EMBEDDINGS_FILENAME):
            target = Path(edit_folder) / name
            if target.exists():
                target.unlink()
                removed += 1
        return removed


def cluster_frames(
    frames, eps: float = DEFAULT_EPS, min_samples: int = DEFAULT_MIN_SAMPLES
) -> tuple[PeopleIndex, np.ndarray]:
    """Group the faces of a scored batch into people.

    ``frames`` are ``CullResult`` objects scored with an embedder. Each
    face's ``person_id`` and each frame's ``person_ids`` are filled in.
    Returns the index and the embedding matrix (for saving).
    """
    refs: list[FaceRef] = []
    owners: list[tuple[object, object]] = []
    vectors: list[np.ndarray] = []
    for frame in frames:
        frame.person_ids = []
        for index, face in enumerate(getattr(frame, "faces", []) or []):
            face.person_id = None
            if face.embedding is None:
                continue
            refs.append(FaceRef(str(frame.path), index, tuple(int(v) for v in face.box)))
            owners.append((frame, face))
            vectors.append(face.embedding)

    index = PeopleIndex()
    if not vectors:
        return index, np.zeros((0, EMBEDDING_SIZE), np.float32)

    matrix = np.vstack(vectors).astype(np.float32)
    labels = cluster_embeddings(matrix, eps, min_samples)
    for ref, (frame, face), label in zip(refs, owners, labels):
        if label == NOISE:
            index.unassigned.append(ref)
            continue
        person = int(label)
        face.person_id = person
        if person not in frame.person_ids:
            frame.person_ids.append(person)
        index.people.setdefault(person, []).append(ref)
    return index, matrix
