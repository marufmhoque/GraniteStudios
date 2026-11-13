"""Blur triage and orientation, on synthetic frames with known answers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.ai_tools import blur_analysis, orientation
from tests.conftest import make_architecture, make_scene, rotate_keep_size


def test_sharp_frame_is_sharp():
    report = blur_analysis.analyze(make_scene(1))
    assert report.is_sharp and not report.is_blurry


def test_defocused_frame_is_flagged():
    report = blur_analysis.analyze(cv2.GaussianBlur(make_scene(1), (0, 0), 9))
    assert report.is_blurry
    assert report.verdict in (blur_analysis.OUT_OF_FOCUS, blur_analysis.MOTION_BLUR)


def test_motion_blur_is_told_apart_from_defocus():
    kernel = np.zeros((31, 31), np.float32)
    kernel[15, :] = 1.0 / 31
    smeared = cv2.filter2D(make_scene(1), -1, kernel)
    report = blur_analysis.analyze(smeared)
    assert report.verdict == blur_analysis.MOTION_BLUR


def test_blurring_lowers_the_score():
    sharp = blur_analysis.sharpness_score(make_scene(2))
    soft = blur_analysis.sharpness_score(cv2.GaussianBlur(make_scene(2), (0, 0), 3))
    assert soft < sharp * 0.5


def test_unreadable_path_reports_an_error_instead_of_raising(tmp_path):
    report = blur_analysis.analyze(tmp_path / "missing.jpg")
    assert report.error


@pytest.mark.parametrize("tilt", [-8.0, -3.0, 4.0, 9.0])
def test_known_tilt_round_trips_to_under_a_degree(tilt):
    tilted = rotate_keep_size(make_architecture(), tilt)
    fixed, result = orientation.auto_rotate(tilted, analyzer=None, pose_analyzer=None)
    assert result.needs_rotation
    residual, _ = orientation.line_tilt(fixed)
    assert residual is not None and abs(residual) < 1.0


def test_level_frame_is_left_alone():
    _, result = orientation.auto_rotate(make_architecture(), analyzer=None, pose_analyzer=None)
    assert not result.needs_rotation


def test_blank_padding_is_not_read_as_scene_structure():
    """Wedges left by an uncropped rotation have perfectly straight edges at
    the rotation angle; they must not be measured as a tilt."""
    padded = orientation.rotate_image(make_scene(3), 11.0, crop=False)
    angle, _ = orientation.line_tilt(padded)
    assert angle is None or abs(abs(angle) - 11.0) > 2.0


def test_crop_leaves_no_blank_corners():
    cropped = orientation.rotate_image(np.full((400, 600, 3), 200, np.uint8), 7.0, crop=True)
    corners = [cropped[0, 0], cropped[0, -1], cropped[-1, 0], cropped[-1, -1]]
    assert all(int(c.min()) > 150 for c in corners)
