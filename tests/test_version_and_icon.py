"""The version number and the Windows shortcut icon."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from src import __version__
from src.engine import app_icon, create_shortcut
from tools import bump_version as bv

ROOT = Path(__file__).resolve().parent.parent


def test_version_has_two_decimals_and_every_copy_agrees():
    assert bv.FORMAT.match(__version__), __version__
    assert bv.mismatches() == []


def test_the_readme_publishes_the_version():
    assert f"**Current version: v{__version__}**" in (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("old,new", [("2.00", "2.01"), ("2.09", "2.10"), ("2.99", "3.00")])
def test_an_update_adds_one_hundredth(old, new):
    assert bv.bumped(old) == new


def test_bump_rewrites_every_copy(tmp_path):
    for name, *_ in bv.PLACES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    bv.write(bv.bumped(bv.current(tmp_path)), tmp_path)
    assert bv.current(tmp_path) == bv.bumped(__version__)
    assert bv.mismatches(tmp_path) == []


def test_check_refuses_an_update_that_kept_its_version(tmp_path, capsys):
    previous = tmp_path / "old_init.py"
    previous.write_text(f'__version__ = "{__version__}"\n', encoding="utf-8")
    assert bv.main(["--check", "--previous", str(previous)]) == 1
    previous.write_text('__version__ = "0.1.0"\n', encoding="utf-8")   # before two decimals
    assert bv.main(["--check", "--previous", str(previous)]) == 0


def test_the_ico_is_the_logo_square_at_every_size():
    ico = ROOT / "icon.ico"
    with Image.open(ico) as image:
        assert set(image.info["sizes"]) == set(app_icon.ICO_SIZES)
        image.size = (256, 256)
        image.load()
        assert image.getchannel("A").getbbox() is not None      # not blank


def test_an_ico_is_made_when_the_install_has_none(tmp_path, monkeypatch):
    shutil.copy(ROOT / "icon.png", tmp_path / "icon.png")
    monkeypatch.setattr(app_icon, "resource_path", lambda name: tmp_path / name)
    monkeypatch.setattr(app_icon, "USER_ICON", tmp_path / "user" / "icon.ico")
    made = app_icon.windows_icon()
    assert made == tmp_path / "user" / "icon.ico" and made.exists()
    with Image.open(made) as image:
        assert image.format == "ICO"


def test_a_shortcut_pointing_at_the_png_is_found(tmp_path, monkeypatch):
    monkeypatch.setattr(create_shortcut.platform, "system", lambda: "Windows")
    monkeypatch.setattr(create_shortcut, "_windows_shortcut_dirs", lambda: [tmp_path / "a", tmp_path / "b"])
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    # .lnk strings are UTF-16: one old shortcut, one already on the .ico.
    (tmp_path / "a" / "Granite Studios.lnk").write_bytes(b"L\0\0\0" + r"C:\x\icon.png,0".encode("utf-16-le"))
    (tmp_path / "b" / "Granite Studios.lnk").write_bytes(b"L\0\0\0" + r"C:\x\icon.ico,0".encode("utf-16-le"))
    assert create_shortcut.stale_shortcuts() == [tmp_path / "a" / "Granite Studios.lnk"]
