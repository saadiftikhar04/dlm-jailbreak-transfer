"""
deepseek_patch.py — place in PiF/ directory, import FIRST in run_pif.py

Three-layer guarantee that every OpenAI() call hits DeepSeek:
  1. Replaces openai.OpenAI class itself
  2. Patches module-level OPENAI_API_KEY in all repo modules
  3. Sets environment variables as fallback
"""
import os, sys

DEEPSEEK_KEY  = os.getenv("DEEPSEEK_API_KEY", "sk-80b9c3e36a374e7489c5ac4438139fdb")
DEEPSEEK_BASE = "https://api.deepseek.com"

# Layer 1: replace the class
from openai import OpenAI as _Orig
class _DS(_Orig):
    def __init__(self, api_key=None, base_url=None, **kw):
        super().__init__(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE, **kw)

import openai as _om
_om.OpenAI = _DS
# patch the symbol so `from openai import OpenAI` also gets our version
if "openai" in sys.modules:
    sys.modules["openai"].OpenAI = _DS

# Layer 3: env vars
os.environ["OPENAI_API_KEY"]  = DEEPSEEK_KEY
os.environ["OPENAI_BASE_URL"] = DEEPSEEK_BASE

def patch_repo_modules():
    """Call after importing repo modules to patch their stored keys."""
    for mod in sys.modules.values():
        if hasattr(mod, "OPENAI_API_KEY"):
            mod.OPENAI_API_KEY = DEEPSEEK_KEY

# Auto-patch already-loaded modules
patch_repo_modules()

def verify():
    """Run this on HPC to confirm patch works before submitting jobs."""
    from openai import OpenAI
    c = OpenAI(api_key="WRONG-KEY-SHOULD-BE-IGNORED")
    base = str(c.base_url).rstrip("/")
    key  = c.api_key
    ok   = "deepseek.com" in base and key == DEEPSEEK_KEY
    print("=" * 50)
    print(f"base_url : {base}")
    print(f"api_key  : {key[:8]}...{key[-4:]}")
    print(f"PATCH    : {'OK' if ok else 'FAILED'}")
    print("=" * 50)
    if not ok:
        raise RuntimeError("Patch failed — check DEEPSEEK_API_KEY")
    return ok
