"""ctypes binding for the Mojo Aho-Corasick scan kernel."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = ROOT / "dist" / "libmojo-pyahocorasick.so"
SOURCE_PATH = ROOT / "src" / "pyahocorasick.mojo"

I64 = ctypes.c_int64
SCAN_ARGUMENT_COUNT = 16


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_PYAHOCORASICK_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi")
    if pixi:
        return [pixi, "run", "--manifest-path", str(ROOT / "pixi.toml"), "mojo"]
    raise BuildError(
        "Mojo compiler not found; run inside pixi or set "
        "MOJO_PYAHOCORASICK_MOJO=/path/to/mojo"
    )


def build(force: bool = False) -> Path:
    if (
        not force
        and LIB_PATH.exists()
        and LIB_PATH.stat().st_mtime >= SOURCE_PATH.stat().st_mtime
    ):
        return LIB_PATH
    LIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = _mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        str(SOURCE_PATH),
        "-o",
        str(LIB_PATH),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode or not LIB_PATH.exists():
        message = (result.stderr or result.stdout).strip()
        raise BuildError(message[:8000])
    return LIB_PATH


_LIB: ctypes.CDLL | None = None
_PY_LIB: ctypes.PyDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(build()))
        _LIB.mpac_scan.argtypes = [I64] * SCAN_ARGUMENT_COUNT
        _LIB.mpac_scan.restype = I64
    return _LIB


def py_lib() -> ctypes.PyDLL:
    global _PY_LIB
    if _PY_LIB is None:
        _PY_LIB = ctypes.PyDLL(str(build()))
        _PY_LIB.mpac_materialize.argtypes = [I64] * 4
        _PY_LIB.mpac_materialize.restype = ctypes.py_object
    return _PY_LIB


def scan(*args: int) -> int:
    if len(args) != SCAN_ARGUMENT_COUNT:
        raise TypeError(
            f"mpac_scan expects {SCAN_ARGUMENT_COUNT} arguments, got {len(args)}"
        )
    result = int(lib().mpac_scan(*args))
    if result < 0:
        raise RuntimeError(f"Mojo scan kernel failed with status {result}")
    return result


def materialize(
    end_indices_address: int,
    match_ids_address: int,
    count: int,
    values: tuple[object, ...],
) -> list[tuple[int, object]]:
    return py_lib().mpac_materialize(
        end_indices_address, match_ids_address, count, id(values)
    )
