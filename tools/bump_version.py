"""Raise Granite Studios' version by 0.01, everywhere it is written.

The version lives in ``src/__init__.py`` as major.minor with two decimals
("2.00", "2.01", ... "2.99", "3.00"). Every update that ships raises it by
0.01. This script rewrites the copies that must match it: ``pyproject.toml``,
the "Current version" line at the top of the README, and the installer's
fallback.

Usage::

    python tools/bump_version.py            # 2.00 -> 2.01
    python tools/bump_version.py --set 2.10
    python tools/bump_version.py --check    # every copy matches; exit 1 if not
    python tools/bump_version.py --check --previous old__init__.py
                                            # ...and the version went up
"""

from __future__ import annotations

import argparse
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEP = Decimal("0.01")
FORMAT = re.compile(r"^\d+\.\d{2}$")

# (file, pattern with one group around the version, how to write it back)
PLACES = [
    ("src/__init__.py", r'__version__ = "([^"]+)"', '__version__ = "{v}"'),
    ("pyproject.toml", r'(?m)^version = "([^"]+)"', 'version = "{v}"'),
    ("README.md", r"\*\*Current version: v([^*]+)\*\*", "**Current version: v{v}**"),
    ("install/install.ps1", r"\$AppVersion = '([^']+)'", "$AppVersion = '{v}'"),
]


def parse(text: str) -> Decimal | None:
    """The version in ``src/__init__.py`` text, or None if it is not in the
    two-decimal form (versions before 2.00 were not)."""
    match = re.search(PLACES[0][1], text)
    if not match or not FORMAT.match(match.group(1)):
        return None
    try:
        return Decimal(match.group(1))
    except InvalidOperation:
        return None


def current(root: Path = ROOT) -> str:
    version = parse((root / PLACES[0][0]).read_text(encoding="utf-8"))
    if version is None:
        raise SystemExit("src/__init__.py has no two-decimal __version__")
    return f"{version:.2f}"


def bumped(version: str) -> str:
    return f"{Decimal(version) + STEP:.2f}"


def write(version: str, root: Path = ROOT) -> None:
    if not FORMAT.match(version):
        raise SystemExit(f"{version!r} is not a two-decimal version such as 2.01")
    for name, pattern, template in PLACES:
        path = root / name
        with open(path, encoding="utf-8", newline="") as handle:   # keep its line endings
            text = handle.read()
        new, count = re.subn(pattern, lambda _m: template.format(v=version), text, count=1)
        if count != 1:
            raise SystemExit(f"{name}: the version line was not found")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(new)


def mismatches(root: Path = ROOT) -> list[str]:
    """Every copy that disagrees with ``src/__init__.py``."""
    version = current(root)
    problems = []
    for name, pattern, _template in PLACES[1:]:
        match = re.search(pattern, (root / name).read_text(encoding="utf-8"))
        found = match.group(1) if match else None
        if found != version:
            problems.append(f"{name} says {found!r}, src/__init__.py says {version!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", metavar="X.YY", help="set this version instead of adding 0.01")
    parser.add_argument("--check", action="store_true", help="verify only; change nothing")
    parser.add_argument("--previous", type=Path,
                        help="with --check: the previous src/__init__.py; the version must be higher")
    args = parser.parse_args(argv)

    if args.check:
        problems = mismatches()
        if args.previous is not None:
            old = parse(args.previous.read_text(encoding="utf-8"))
            if old is not None and Decimal(current()) <= old:
                problems.append(f"the version is still {current()}: an update must raise it "
                                f"(python tools/bump_version.py)")
        for problem in problems:
            print(problem)
        if not problems:
            print(f"ok: version {current()}")
        return 1 if problems else 0

    old = current()
    new = args.set or bumped(old)
    write(new)
    print(f"{old} -> {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
