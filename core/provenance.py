"""What has to be logged with every result file.

Seed, git SHA and config travel with the numbers, so a row can be traced back to the
code that produced it.
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(['git', *args], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_sha() -> Optional[str]:
    return _git('rev-parse', 'HEAD')


#: results/ is excluded from the dirtiness check. The requirement it serves is "the
#: CODE that produced this row is committed"; writing the
#: results themselves would otherwise make the tree dirty by the act of writing them,
#: so every run would report DIRTY and the flag would mean nothing.
DIRTY_CHECK_EXCLUDES = (':!results', ':!results/**')


def git_is_dirty() -> Optional[bool]:
    status = _git('status', '--porcelain', '--', '.', *DIRTY_CHECK_EXCLUDES)
    return None if status is None else bool(status)


def provenance() -> dict:
    """Everything needed to reproduce a run, minus the run's own configuration."""
    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git_sha': git_sha(),
        'git_dirty': git_is_dirty(),
        'git_branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'python': platform.python_version(),
        # Free-threaded builds (PEP 703) currently run without the specializing
        # interpreter, so single-threaded numeric code can be slower than on a
        # GIL build. Every timing is therefore tagged with the interpreter it was
        # taken on, and timings from different interpreters must not be compared.
        'python_free_threaded': _free_threaded(),
        'platform': platform.platform(),
        'numpy': _version('numpy'),
        'scipy': _version('scipy'),
    }


def _free_threaded() -> Optional[bool]:
    """True on a PEP 703 free-threaded build (GIL disabled), None on an older Python."""
    is_gil_enabled = getattr(sys, '_is_gil_enabled', None)
    return None if is_gil_enabled is None else not is_gil_enabled()


def _version(module_name: str) -> Optional[str]:
    try:
        return __import__(module_name).__version__
    except Exception:  # noqa: BLE001
        return None


def write_jsonl(path: os.PathLike | str, row: dict) -> Path:
    """Append one JSON object per line: the raw log.

    `results/*.jsonl` is git-ignored; the committed artefacts are the CSV and Markdown
    in `results/`. Note that the `*.json` rule in `.gitignore` does NOT match `.jsonl`,
    which is why that file carries explicit `results/*.jsonl` lines.

    Non-finite floats are written as the strings 'inf', '-inf' and 'nan', because bare
    `Infinity`/`NaN` are accepted by Python's json but are not valid JSON. `r_min_m` is
    genuinely infinite for a straight alignment, so this is a normal case, not an error.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(sanitise(row), sort_keys=True,
                                default=_jsonable, allow_nan=False) + '\n')
    return path


def sanitise(obj):
    """Recursively make `obj` strictly JSON-serialisable.

    Plain Python `float('inf')` is what `core.metrics.r_min_m` returns, and `_jsonable`
    never sees it because `json` handles floats itself -- hence this pass.
    """
    import numpy as np
    if isinstance(obj, dict):
        return {key: sanitise(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitise(value) for value in obj]
    if isinstance(obj, (set, frozenset)):
        return [sanitise(value) for value in sorted(obj)]
    if isinstance(obj, np.ndarray):
        return sanitise(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else str(value)
    return obj


def _jsonable(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if value == value and abs(value) != float('inf') else str(value)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")
