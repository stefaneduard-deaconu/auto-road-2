# Create the project venv on CPython 3.14 FREE-THREADED (python3.14t.exe, PEP 703).
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File setup_venv.ps1
#
# One venv directory per PLATFORM. A checkout shared with WSL (reached as /mnt/c/...)
# cannot hold both in .venv: the second run overwrites pyvenv.cfg and leaves Scripts\+Lib\
# beside bin/+lib64/. Set $env:VENV to keep them apart:
#     $env:VENV = '.venv-win'; .\setup_venv.ps1
$ErrorActionPreference = 'Stop'

$venv = $env:VENV
if (-not $venv) { $venv = '.venv' }

if (Test-Path (Join-Path $venv 'bin')) {
    if (Test-Path (Join-Path $venv 'Scripts')) {
        Write-Host "$venv holds BOTH layouts (Scripts\ and bin/): the two setup scripts have"
        Write-Host "already overwritten each other's pyvenv.cfg in this checkout."
    } else {
        Write-Host "$venv was created by setup_venv.sh on POSIX (it has bin/, not Scripts\)."
        Write-Host "Creating a Windows venv there would overwrite its pyvenv.cfg."
    }
    Write-Host "Either keep one per platform:   `$env:VENV = '.venv-win'; .\setup_venv.ps1"
    Write-Host "or remove it and rebuild here:  Remove-Item -Recurse -Force $venv"
    Write-Error "refusing to overwrite a venv built for another platform"
}

# Preferred path: uv (pyproject.toml + uv.lock). `--no-build` is uv's --only-binary: a
# missing cp314t wheel fails loudly instead of starting a source build. Set $env:NO_UV = '1'
# to force the pip path below, which installs the same pins from requirements*.txt.
if (-not $env:NO_UV -and (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv found: installing from uv.lock into $venv"
    uv python install 3.14t
    if ($LASTEXITCODE -ne 0) { Write-Error "uv python install 3.14t failed" }
    $env:UV_PROJECT_ENVIRONMENT = $venv
    uv sync --no-build --python 3.14t
    if ($LASTEXITCODE -ne 0) { Write-Error "uv sync failed" }
    $py = Join-Path $venv 'Scripts\python.exe'
    & $py -c "import sys; assert not sys._is_gil_enabled(), 'this is not a free-threaded build'"
    & $py -m pytest -q
    exit $LASTEXITCODE
}

# The free-threaded build is a SEPARATE interpreter, not a flag on python.exe.
# Install it with one of:
#   uv python install 3.14t
#   py install 3.14t          # needs the real py launcher, not the Microsoft Store shim
#   the python.org installer, ticking the free-threaded option
$python = $env:PYTHON
if (-not $python) {
    $candidates = @(
        "$env:USERPROFILE\.local\bin\python3.14t.exe",
        'python3.14t.exe'
    )
    $python = $candidates | Where-Object { Get-Command $_ -ErrorAction SilentlyContinue } | Select-Object -First 1
}
if (-not $python) {
    Write-Error "python3.14t.exe not found. Install it with 'uv python install 3.14t' or 'py install 3.14t'."
}

& $python -c "import sys; assert not sys._is_gil_enabled(), 'this is not a free-threaded build'"
& $python -m venv $venv
# --only-binary=:all: so a missing cp314t wheel fails loudly instead of starting a
# source build (scipy from source needs a Fortran toolchain on Windows).
$py = Join-Path $venv 'Scripts\python.exe'
& $py -m pip install --upgrade pip
& $py -m pip install --only-binary=:all: -r requirements-dev.txt
& $py -m pytest -q
