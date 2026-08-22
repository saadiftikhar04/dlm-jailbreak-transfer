"""_paths.py — single source of truth for repo paths (mirrors repo-root paths.txt).

Convention (portable across this desktop and the HPC):
    <repo>/paths.txt  is git-ignored (one copy per machine), its FIRST bare line
    holds the repo root absolute path, e.g.:

        /home/bc3194/Desktop/dlm-jailbreak-transfer

    Optional KEY=VALUE lines may follow (stored in `overrides`) for any
    per-machine values (e.g. HF_CACHE=..., DATA_DIR=...) the callers choose to
    consume. Nothing is hard-coded in the scripts.

Every runner in scripts/cross-check imports `ROOT` (and, where relevant,
`overrides`) from here instead of embedding a path. On a fresh checkout the
scripts resolve ROOT to paths.txt line 1; if paths.txt is absent they fall back
to locating the repo root structurally (has results/ + scripts/) so the audit
still works before paths.txt is created.
"""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # scripts/cross-check


def _find_root_by_structure() -> Path:
    """Walk upward to a dir that has both results/ and scripts/ (the repo root)."""
    d = _HERE
    for _ in range(8):
        if (d / "results").is_dir() and (d / "scripts").is_dir():
            return d
        nd = d.parent
        if nd == d:
            break
        d = nd
    return _HERE.parent.parent  # fallback: repo root


def repo_root() -> Path:
    env = os.environ.get("DLM_REPO_ROOT")
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p
    # paths.txt (per-machine) wins over structural fallback
    pf = _HERE.parent.parent / "paths.txt"
    if pf.is_file():
        line1 = next((l.strip() for l in pf.read_text(encoding="utf-8").splitlines()
                      if l.strip() and not l.startswith("#")), None)
    else:
        line1 = None
    if line1:
        p = Path(line1).expanduser()
        if p.is_dir():
            return p
    return _find_root_by_structure()


def overrides() -> dict:
    """Return KEY=VALUE pairs following line 1 of paths.txt (lower-cased keys)."""
    pf = _HERE.parent.parent / "paths.txt"
    out = {}
    if not pf.is_file():
        return out
    for l in pf.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if not l or l.startswith("#") or "=" not in l:
            continue
        k, _, v = l.partition("=")
        out[k.strip().upper()] = v.strip()
    return out


def hf_cache() -> str:
    """HF hub cache dir. Resolution order: env -> paths.txt override -> standard."""
    v = (os.environ.get("CROSSCHECK_HF_CACHE")
         or os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
         or overrides().get("HF_CACHE"))
    if v:
        return os.path.expanduser(v)
    home = Path.home()
    for cand in (home / ".cache" / "huggingface" / "hub",
                 home / "Desktop" / "huggingface_cache" / "hub"):
        if cand.is_dir():
            return str(cand)
    return str(home / ".cache" / "huggingface" / "hub")


def data_dir() -> str:
    """Student data dir (weights+dataset mirror). Env -> paths.txt override."""
    return (os.environ.get("CROSSCHECK_DATA_DIR")
            or overrides().get("DATA_DIR")
            or "")


ROOT = repo_root()
OVERRIDES = overrides()


if __name__ == "__main__":
    print("ROOT      =", ROOT)
    print("OVERRIDES =", OVERRIDES)
    print("HF_CACHE  =", hf_cache())
    print("DATA_DIR  =", data_dir() or "(unset)")