"""Select a real Bash and construct PATH values for local shell tests."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _find_bash() -> str | None:
    candidates = [os.environ.get("TYPESAFE_TEST_BASH"), shutil.which("bash")]
    if os.name == "nt":
        candidates.extend(
            [
                r"C:\Program Files\Git\bin\bash.exe",
                r"C:\Program Files (x86)\Git\bin\bash.exe",
            ]
        )
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen or not Path(candidate).is_file():
            continue
        seen.add(candidate)
        try:
            result = subprocess.run(
                [candidate, "-c", '[[ -n "${BASH_VERSION:-}" ]] && a=()'],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate
    return None


BASH = _find_bash()
POSIX_SPAWN = getattr(os, "posix_spawn", None)
POSIX_SPAWNP = getattr(os, "posix_spawnp", None)


def allow_local_shell_processes(monkeypatch, popen) -> None:
    """Restore process creation only in tests that run isolated scripts in temporary folders."""
    monkeypatch.setattr(subprocess, "Popen", popen)
    if POSIX_SPAWN is not None:
        monkeypatch.setattr(os, "posix_spawn", POSIX_SPAWN)
    if POSIX_SPAWNP is not None:
        monkeypatch.setattr(os, "posix_spawnp", POSIX_SPAWNP)


def shell_env(*prefix: str | Path, home: str | Path | None = None) -> dict[str, str]:
    """Build an environment that exposes test shims and the selected Bash tools."""
    env = os.environ.copy()
    entries = [str(path) for path in prefix]
    if os.name == "nt" and BASH:
        git_root = Path(BASH).resolve().parent.parent
        entries.extend(str(git_root / path) for path in ("usr/bin", "usr/bin/core_perl", "mingw64/bin", "bin"))
    if env.get("PATH"):
        entries.append(env["PATH"])
    env["PATH"] = os.pathsep.join(entries)
    if home is not None:
        env["HOME"] = str(home)
    return env


def shell_path(path: str | Path) -> str:
    """Convert a Windows path before embedding it inside Bash source text."""
    value = str(path)
    if os.name != "nt" or not BASH:
        return value
    cygpath = Path(BASH).resolve().parent.parent / "usr" / "bin" / "cygpath.exe"
    result = subprocess.run([str(cygpath), "-u", value], capture_output=True, text=True, check=True)
    return result.stdout.strip()
