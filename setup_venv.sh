#!/usr/bin/env bash
# Create the project venv on CPython 3.14 FREE-THREADED (python3.14t, PEP 703).
# Windows users: use setup_venv.ps1 instead.
#
# One venv directory per PLATFORM. A checkout shared between Windows and WSL (a /mnt/c
# path) cannot hold both in `.venv`: the second run overwrites `pyvenv.cfg` and leaves
# `Scripts/`+`Lib/` beside `bin/`+`lib64/`. Set VENV to keep them apart:
#     VENV=.venv-linux bash setup_venv.sh
set -euo pipefail

PYTHON="${PYTHON:-python3.14t}"
VENV="${VENV:-.venv}"

if [ -e "$VENV/Scripts" ]; then
    if [ -e "$VENV/bin" ]; then
        echo "$VENV holds BOTH layouts (Scripts/ and bin/): the two setup scripts have" >&2
        echo "already overwritten each other's pyvenv.cfg in this checkout." >&2
    else
        echo "$VENV was created by setup_venv.ps1 on Windows (it has Scripts/, not bin/)." >&2
        echo "Creating a POSIX venv there would overwrite its pyvenv.cfg." >&2
    fi
    echo "Either keep one per platform:   VENV=.venv-linux bash setup_venv.sh" >&2
    echo "or remove it and rebuild here:  rm -rf $VENV && bash setup_venv.sh" >&2
    exit 1
fi

# Preferred path: uv (pyproject.toml + uv.lock). `--no-build` is uv's --only-binary: a
# missing cp314t wheel fails loudly instead of starting a source build. Set NO_UV=1 to force
# the pip path below, which installs the same pins from requirements*.txt.
if [ -z "${NO_UV:-}" ] && command -v uv >/dev/null 2>&1; then
    echo "uv found: installing from uv.lock into $VENV"
    uv python install 3.14t
    UV_PROJECT_ENVIRONMENT="$VENV" uv sync --no-build --python 3.14t
    "$VENV/bin/python" -c 'import sys; assert not sys._is_gil_enabled(), "this is not a free-threaded build"'
    "$VENV/bin/python" -m pytest -q
    exit 0
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "$PYTHON not found." >&2
    echo "The free-threaded build is a separate interpreter, not a flag." >&2
    echo "Install it with one of:" >&2
    echo "    uv python install 3.14t" >&2
    echo "    py install 3.14t            # py launcher 3.13+, not the Store shim" >&2
    echo "    the python.org installer, ticking the free-threaded option" >&2
    exit 1
fi

"$PYTHON" -c 'import sys; assert not sys._is_gil_enabled(), "this is not a free-threaded build"'
"$PYTHON" -m venv "$VENV"
# --only-binary=:all: so a missing cp314t wheel fails loudly instead of starting a
# source build (scipy from source needs a Fortran toolchain).
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install --only-binary=:all: -r requirements-dev.txt
"$VENV/bin/python" -m pytest -q
