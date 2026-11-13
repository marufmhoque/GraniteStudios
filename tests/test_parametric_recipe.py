"""The recipe format: round trips, tolerance of bad input, and INV-7 --
editing writes nothing until it is asked to."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.engine import develop_pipeline as dp
from src.engine import parametric_recipe as pr
from src.engine.parametric_recipe import Recipe
from tests.conftest import assert_originals_untouched, make_scene, snapshot


def _tree(folder) -> dict[str, int]:
    return {str(p.relative_to(folder)): p.stat().st_size for p in sorted(folder.rglob("*")) if p.is_file()}


def _busy_recipe() -> Recipe:
    r = Recipe()
    r.basic.exposure, r.basic.contrast, r.basic.temperature = 1.25, -30.0, 18.0
    r.basic.clarity, r.basic.dehaze, r.basic.vibrance = 22.0, 10.0, -15.0
    r.curve.master = [[0.0, 0.02], [0.4, 0.35], [1.0, 0.97]]
    r.curve.blue = [[0.0, 0.0], [0.5, 0.56], [1.0, 1.0]]
    r.hsl.hue[0], r.hsl.saturation[5], r.hsl.luminance[3] = 12.0, -40.0, 25.0
    r.grade.shadows = pr.GradeWheel(220.0, 30.0, -5.0)
    r.grade.balance = 20.0
    r.geometry.quarter_turns, r.geometry.rotation = 1, -2.5
    r.geometry.crop = [0.1, 0.1, 0.8, 0.75]
    r.lens.profile, r.lens.vignette = True, 15.0
    r.detail.sharpen_amount = 45.0
    r.look = pr.builtin_looks()["warm_film"]
    r.lut = pr.Lut("looks/teal.cube", 60.0)
    r.layers = [{"kind": "adjustment", "name": "Skin"}]
    return r


# --------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------- #

def test_a_new_recipe_is_neutral():
    assert Recipe().is_neutral
    assert pr.from_dict(pr.to_dict(Recipe())).is_neutral


def test_every_section_survives_a_round_trip():
    recipe = _busy_recipe()
    again = pr.loads(pr.dumps(recipe))
    assert pr.to_dict(again) == pr.to_dict(recipe)
    assert again.load_error is None and not again.is_neutral


def test_copy_is_independent():
    recipe = _busy_recipe()
    clone = recipe.copy()
    clone.basic.exposure = -3.0
    clone.hsl.hue[0] = 99.0
    assert recipe.basic.exposure == 1.25 and recipe.hsl.hue[0] == 12.0


def test_reset_returns_one_section_to_neutral():
    recipe = _busy_recipe()
    recipe.reset("basic")
    assert recipe.basic == pr.Basic()
    assert recipe.hsl.saturation[5] == -40.0, "resetting one section touched another"
    with pytest.raises(KeyError):
        recipe.reset("nonsense")


def test_the_serialised_form_is_stable_and_readable():
    text = pr.dumps(_busy_recipe())
    assert text == pr.dumps(pr.loads(text))
    assert json.loads(text)["basic"]["exposure"] == 1.25


# --------------------------------------------------------------------- #
# Bad input never raises and never guesses
# --------------------------------------------------------------------- #

def test_out_of_range_numbers_are_clamped():
    recipe = pr.from_dict({"basic": {"exposure": 99, "contrast": -5000}, "geometry": {"rotation": 720}})
    assert recipe.basic.exposure == 5.0
    assert recipe.basic.contrast == -100.0
    assert recipe.geometry.rotation == 45.0


@pytest.mark.parametrize("bad", ["a lot", None, [], {}, float("nan"), float("inf")])
def test_wrong_types_fall_back_to_the_default(bad):
    recipe = pr.from_dict({"basic": {"exposure": bad}, "detail": {"sharpen_radius": bad}})
    assert recipe.basic.exposure == 0.0
    assert recipe.detail.sharpen_radius == 1.0


def test_unknown_fields_are_ignored():
    recipe = pr.from_dict({"basic": {"exposure": 1.0, "sparkle": 11}, "future_section": {"x": 1}})
    assert recipe.basic.exposure == 1.0 and recipe.load_error is None


@pytest.mark.parametrize("data", [None, 7, "recipe", [1, 2, 3]])
def test_a_recipe_that_is_not_an_object_is_neutral_with_an_error(data):
    recipe = pr.from_dict(data)
    assert recipe.is_neutral and recipe.load_error


@pytest.mark.parametrize("sections", [
    {"basic": "nope"}, {"curve": 5}, {"hsl": []}, {"grade": "x"}, {"geometry": None},
    {"lens": 3}, {"detail": "y"}, {"look": []}, {"lut": 9}, {"layers": "z"},
])
def test_a_section_of_the_wrong_shape_is_skipped(sections):
    assert pr.from_dict(sections).is_neutral


def test_text_that_is_not_json():
    recipe = pr.loads("{ this is not json")
    assert recipe.is_neutral and "JSON" in recipe.load_error


def test_a_recipe_from_a_newer_version_loads_with_a_warning():
    recipe = pr.from_dict({"version": pr.RECIPE_VERSION + 4, "basic": {"exposure": 0.5}})
    assert recipe.basic.exposure == 0.5
    assert "newer" in recipe.load_error


def test_curves_are_cleaned_up():
    recipe = pr.from_dict({"curve": {"master": [[0.8, 0.9], [0.2, 0.1], [0.2, 0.3], [1.5, 2.0], "junk", [0.5]]}})
    assert recipe.curve.master == [[0.2, 0.3], [0.8, 0.9], [1.0, 1.0]]
    assert pr.from_dict({"curve": {"master": [[0.5, 0.5]]}}).curve.master == [[0.0, 0.0], [1.0, 1.0]]


def test_hsl_lists_of_the_wrong_length():
    recipe = pr.from_dict({"hsl": {"hue": [10, 20], "saturation": list(range(20)), "luminance": "x"}})
    assert recipe.hsl.hue == [10.0, 20.0, 0, 0, 0, 0, 0, 0]
    assert len(recipe.hsl.saturation) == 8 and recipe.hsl.luminance == [0.0] * 8


@pytest.mark.parametrize("crop,expected", [
    ([0.1, 0.2, 0.5, 0.5], [0.1, 0.2, 0.5, 0.5]),
    ([0.8, 0.8, 0.9, 0.9], [0.8, 0.8, pytest.approx(0.2), pytest.approx(0.2)]),   # trimmed to the frame
    ([0.5, 0.5, 0.0, 0.3], None),                                                 # empty
    ("wide", None), ([1, 2], None),
])
def test_crop_is_validated(crop, expected):
    assert pr.from_dict({"geometry": {"crop": crop}}).geometry.crop == expected


def test_a_look_needs_a_real_matrix():
    assert pr.from_dict({"look": {"matrix": [[1, 0], [0, 1]]}}).look.matrix is None
    assert pr.from_dict({"look": {"matrix": "identity"}}).look.matrix is None
    good = pr.from_dict({"look": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "gamma": -4}})
    assert good.look.matrix is not None and good.look.gamma == 1.0


# --------------------------------------------------------------------- #
# Sidecars
# --------------------------------------------------------------------- #

def test_the_sidecar_lives_in_the_edit_folder_under_the_full_filename(tmp_path):
    edit = tmp_path / "Shoot-edit"
    raw, jpeg = tmp_path / "IMG_1.CR2", tmp_path / "IMG_1.JPG"
    assert pr.sidecar_path(raw, edit) == edit / "IMG_1.CR2.granite.json"
    assert pr.sidecar_path(raw, edit) != pr.sidecar_path(jpeg, edit)
    assert pr.sidecar_path(raw, edit).parent == edit


def test_save_then_load(tmp_path):
    edit = tmp_path / "Shoot-edit"
    recipe = _busy_recipe()
    written = pr.save(recipe, tmp_path / "IMG_1.jpg", edit)
    assert written == edit / "IMG_1.jpg.granite.json"
    assert [p.name for p in edit.iterdir()] == ["IMG_1.jpg.granite.json"], "a temporary file was left behind"
    assert pr.to_dict(pr.load(tmp_path / "IMG_1.jpg", edit)) == pr.to_dict(recipe)


def test_saving_a_neutral_recipe_removes_the_file(tmp_path):
    edit = tmp_path / "Shoot-edit"
    source = tmp_path / "IMG_1.jpg"
    pr.save(_busy_recipe(), source, edit)
    assert pr.save(Recipe(), source, edit) is None
    assert not pr.sidecar_path(source, edit).exists()
    assert pr.save(Recipe(), source, edit) is None      # and is fine when there is none


def test_a_missing_sidecar_is_a_neutral_recipe(tmp_path):
    recipe = pr.load(tmp_path / "IMG_1.jpg", tmp_path / "nowhere")
    assert recipe.is_neutral and recipe.load_error is None


def test_a_damaged_sidecar_does_not_stop_the_photo_opening(tmp_path):
    edit = tmp_path / "Shoot-edit"
    edit.mkdir()
    source = tmp_path / "IMG_1.jpg"
    pr.sidecar_path(source, edit).write_bytes(b"\x00\xff\x00 truncated")
    recipe = pr.load(source, edit)
    assert recipe.is_neutral and recipe.load_error


# --------------------------------------------------------------------- #
# INV-7: nothing reaches the disk until it is asked for
# --------------------------------------------------------------------- #

def test_editing_and_previewing_write_nothing(library):
    """Change every kind of setting and render previews. The folder, and
    everything under it, must be exactly as it was."""
    before_files = _tree(library)
    before_hashes = snapshot(library)
    source_path = library / "IMG_0001.jpg"
    edit = library / "Shoot-edit"

    recipe = pr.load(source_path, edit)
    source = dp.open_source(source_path)
    for step in range(4):
        recipe.basic.exposure = 0.3 * step
        recipe.basic.clarity = 10.0 * step
        recipe.hsl.saturation[step] = -20.0
        recipe.geometry.rotation = 1.5 * step
        recipe.look = pr.builtin_looks()["noir"]
        dp.render_preview(source, recipe, max_side=400)

    assert _tree(library) == before_files, "editing or previewing wrote a file"
    assert not edit.exists(), "the edit folder was created without an export or a save"
    assert_originals_untouched(before_hashes, library)


def test_only_an_explicit_save_writes_and_only_the_sidecar(library):
    before = snapshot(library)
    source_path = library / "IMG_0001.jpg"
    edit = library / "Shoot-edit"
    recipe = Recipe()
    recipe.basic.exposure = 0.5

    pr.save(recipe, source_path, edit)
    assert [p.name for p in edit.iterdir()] == ["IMG_0001.jpg.granite.json"]
    assert_originals_untouched(before, library)


def test_only_an_explicit_export_writes_pixels(library):
    before = snapshot(library)
    source_path = library / "IMG_0001.jpg"
    edit = library / "Shoot-edit"
    recipe = Recipe()
    recipe.basic.exposure = 0.5

    out = dp.export(source_path, recipe, edit / "IMG_0001.jpg")
    assert out.exists() and [p.name for p in edit.iterdir()] == ["IMG_0001.jpg"]
    exported = cv2.imread(str(out))
    original = cv2.imread(str(source_path))
    assert exported.mean() > original.mean() + 5, "the export does not show the edit"
    assert_originals_untouched(before, library)


def test_export_refuses_to_overwrite_the_original(library):
    before = snapshot(library)
    source_path = library / "IMG_0001.jpg"
    with pytest.raises(ValueError, match="original"):
        dp.export(source_path, Recipe(), source_path)
    assert_originals_untouched(before, library)


# --------------------------------------------------------------------- #
# Looks
# --------------------------------------------------------------------- #

def test_the_six_original_grades_are_available_as_looks():
    looks = pr.builtin_looks()
    assert set(looks) == {
        "cinematic_teal_orange", "warm_film", "cool_matte", "bleach_bypass", "vintage_fade", "noir",
    }
    recipe = pr.with_look("noir")
    assert recipe.look.name == "noir" and not recipe.is_neutral
    with pytest.raises(KeyError):
        pr.with_look("sepia_dream")


@pytest.mark.parametrize("name", [
    "cinematic_teal_orange", "warm_film", "cool_matte", "bleach_bypass", "vintage_fade", "noir",
])
def test_a_look_renders_exactly_as_the_original_grade_did(name):
    """The grades moved from baked-in pixels to recipe settings. The result
    has to be the same picture."""
    from src.ai_tools.color_engine import ColorEngine

    image = cv2.cvtColor(make_scene(2, (320, 240)), cv2.COLOR_BGR2RGB)
    expected = ColorEngine().apply_lut_filter(image, name)
    rendered = dp.render(dp.from_array(image), pr.with_look(name))
    assert np.abs(rendered.astype(int) - expected.astype(int)).max() <= 1
