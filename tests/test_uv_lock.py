"""The two install paths of setup_venv.* install the same versions: uv from pyproject.toml +
uv.lock, pip from requirements.txt + requirements-dev.txt."""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pins(lines) -> dict:
    pins = {}
    for line in lines:
        line = line.split('#')[0].strip()
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)', line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def test_requirements_equal_the_pyproject_pins():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    declared = _pins(project['project']['dependencies'] + project['dependency-groups']['dev'])
    files = _pins((ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines()
                  + (ROOT / 'requirements-dev.txt').read_text(encoding='utf-8').splitlines())
    assert declared == files


def test_the_lock_holds_the_pinned_versions():
    lock = tomllib.loads((ROOT / 'uv.lock').read_text(encoding='utf-8'))
    locked = {p['name'].lower(): p['version'] for p in lock['package'] if 'version' in p}
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    for name, version in _pins(project['project']['dependencies']
                               + project['dependency-groups']['dev']).items():
        assert locked.get(name) == version, (name, locked.get(name), version)


def test_uv_selects_the_free_threaded_interpreter():
    assert (ROOT / '.python-version').read_text(encoding='utf-8').strip() == '3.14t'
