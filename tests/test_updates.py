"""Finding a newer version on GitHub and handing over to the updater.

The network is never touched: ``urlopen`` and ``Popen`` are replaced.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from src import __version__
from src.engine import updates
from tools import bump_version

ROOT = Path(__file__).resolve().parent.parent


class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def prefs_file(tmp_path, monkeypatch):
    path = tmp_path / "updates.json"
    monkeypatch.setattr(updates, "PREFS_PATH", path)
    return path


def test_versions_compare_as_numbers():
    assert updates.is_newer("2.01", "2.00")
    assert updates.is_newer("2.10", "2.09")
    assert updates.is_newer("3.00", "2.99")
    assert not updates.is_newer("2.00", "2.00")
    assert not updates.is_newer("2.00", "2.01")
    assert updates.is_newer("2.00", "0.1.0")        # installs from before 2.00
    assert not updates.is_newer(None, "2.00")
    assert not updates.is_newer("garbage", "2.00")


def test_the_version_is_read_from_githubs_copy(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"], seen["agent"] = request.full_url, request.get_header("User-agent")
        return _Reply(b'"""Granite."""\n__version__ = "2.07"\n')

    monkeypatch.setattr(updates.urllib.request, "urlopen", fake_urlopen)
    assert updates.latest_version() == "2.07"
    assert seen["url"] == updates.VERSION_URL
    assert seen["agent"] == "GraniteStudios-updater"      # nothing identifying


@pytest.mark.parametrize("reply", [OSError("offline"), b"<html>rate limited</html>", b'__version__ = "0.1.0"'])
def test_an_unreadable_reply_is_no_update(monkeypatch, reply):
    def fake_urlopen(request, timeout):
        if isinstance(reply, Exception):
            raise reply
        return _Reply(reply)

    monkeypatch.setattr(updates.urllib.request, "urlopen", fake_urlopen)
    assert updates.latest_version() is None


def test_preferences_round_trip_and_survive_damage(prefs_file):
    assert updates.load_prefs() == {"auto_check": True, "last_check": 0.0}
    updates.save_prefs({"auto_check": False, "last_check": 123.0})
    assert updates.load_prefs() == {"auto_check": False, "last_check": 123.0}
    prefs_file.write_text("{not json", encoding="utf-8")
    assert updates.load_prefs()["auto_check"] is True
    prefs_file.write_text('{"auto_check": "no", "last_check": "soon"}', encoding="utf-8")
    assert updates.load_prefs() == {"auto_check": True, "last_check": 0.0}


def test_the_check_runs_at_most_once_a_day_and_can_be_off():
    day = updates.CHECK_INTERVAL_S
    assert updates.check_due({"auto_check": True, "last_check": 0.0}, now=day)
    assert not updates.check_due({"auto_check": True, "last_check": day}, now=day + 3600)
    assert updates.check_due({"auto_check": True, "last_check": day}, now=2 * day)
    assert not updates.check_due({"auto_check": False, "last_check": 0.0}, now=10 * day)


def test_a_development_checkout_never_updates_itself(monkeypatch):
    assert not updates.can_self_update()
    opened = []
    monkeypatch.setattr(updates.webbrowser, "open", opened.append)
    monkeypatch.setattr(updates.subprocess, "Popen", lambda *a, **k: pytest.fail("ran the updater"))
    assert updates.launch_update() is False
    assert opened == [updates.INSTALLER_URL]


def test_an_installed_copy_runs_its_updater_and_asks_to_reopen(tmp_path, monkeypatch):
    (tmp_path / updates.UPDATER_NAME).write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setattr(updates.platform, "system", lambda: "Windows")
    monkeypatch.setattr(updates, "get_base_path", lambda: tmp_path)
    monkeypatch.setattr(updates, "install_dir", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(updates.subprocess, "Popen", lambda args, **kw: calls.append((args, kw)))
    assert updates.can_self_update()
    assert updates.launch_update() is True
    args, kw = calls[0]
    assert args == ["cmd.exe", "/c", str(tmp_path / updates.UPDATER_NAME)]
    assert kw["env"]["GRANITE_RELAUNCH"] == "1"
    assert kw["creationflags"] == getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


# -- the window ---------------------------------------------------------------- #

def test_the_window_shows_its_version_and_offers_a_newer_one(harness, prefs_file):
    w = harness.window
    assert w.version_button.text() == f"v{__version__}"
    assert w.update_button.isHidden()

    w._on_update_result(__version__, True)
    assert w.update_button.isHidden()
    assert "latest version" in w.status_label.text()

    w._on_update_result(None, True)
    assert "Could not reach GitHub" in w.status_label.text()

    newer = bump_version.bumped(__version__)
    w._on_update_result(newer, False)
    assert not w.update_button.isHidden()
    assert w.update_button.text() == f"Update to v{newer}"


def test_turning_off_the_automatic_check_is_remembered(harness, prefs_file):
    w = harness.window
    assert w.auto_update_action.isChecked()
    w.auto_update_action.setChecked(False)
    assert updates.load_prefs()["auto_check"] is False
    assert not updates.check_due(updates.load_prefs())


def test_building_the_window_does_not_check(monkeypatch, prefs_file, harness):
    # The installer's start-up check builds a window; it must stay offline.
    monkeypatch.setattr(updates, "latest_version", lambda *a, **k: pytest.fail("checked on construction"))
    assert harness.window.available_version is None


# -- the installer --------------------------------------------------------------- #

def test_the_installer_keeps_an_updater_and_reopens_the_app():
    install = (ROOT / "install" / "install.ps1").read_text(encoding="utf-8")
    bootstrap = (ROOT / "install" / "Install-GraniteStudios.bat").read_text(encoding="utf-8")
    assert "Update Granite Studios.bat" in install and "Update Granite Studios.lnk" in install
    assert "GRANITE_RELAUNCH" in install and "Waiting for Granite Studios to close" in install
    assert "/src/__init__.py" in bootstrap and "You already have the latest version" in bootstrap
    # Run from the install folder it relaunches from TEMP, since the update
    # replaces it mid-run; and no answer at all counts as "no".
    assert 'call "%TEMP%\GraniteStudios-update.bat"' in bootstrap
    assert "[string](Read-Host" in bootstrap
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.bat text eol=crlf" in attributes and "*.ps1 text eol=crlf" in attributes
    assert updates.VERSION_URL.endswith("/main/src/__init__.py")
